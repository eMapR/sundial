import ee
import geopandas as gpd
import math
import numpy as np
import regex as re
import time

from numpy.lib import recfunctions as rfn
from shapely.geometry import box

from constants import EE_END_POINT, GEE_REQUEST_LIMIT_MB
from pipeline.utils import ParallelGridAlign
from config_utils import dynamic_import


class Downloader(ParallelGridAlign):
    def __init__(
            self, imagery_path, ee_factory, filter_intersect,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._source_path = imagery_path
        self._ee_factory = dynamic_import(ee_factory, {"projection": self._epsg_str})
        self._filter_intersect = filter_intersect

    def _image_generator(self, chunk: np.ndarray) -> tuple:
        ee.Initialize(opt_url=EE_END_POINT)
        try:
            ty, tx = chunk
            chunk_box = box(tx, ty - self._grid_y_size, tx + self._grid_x_size, ty)
            chunk_coords = list(chunk_box.boundary.coords)
            
            self._report_queue.put(("INFO", f"Creating image payload for chunk... {chunk_coords}"))
            image = self._ee_factory.create_ee_image(chunk_coords)
            image = image.reproject(crs=self._epsg_str, scale=self._scale)

            payload = {
                "expression": image,
                "fileFormat": "NUMPY_NDARRAY"
            }

            translateY = max(c[1] for c in chunk_coords)
            translateX = min(c[0] for c in chunk_coords)
            
            payload["grid"] = {
                'dimensions': {
                    'width': self._chunk_sizes[-1],
                    'height': self._chunk_sizes[-2]
                },
                'affineTransform': {
                    'scaleX': self._scale,
                    'shearX': 0,
                    'translateX': translateX,
                    'shearY': 0,
                    'scaleY': -self._scale,
                    'translateY': translateY
                },
                'crsCode': self._epsg_str,
            }
            return payload, translateY, translateX
        except Exception as e:
            self._report_queue.put(("CRITICAL", f"Failed to create image payload for chunk skipping: {type(e)} {e} {chunk_coords}"))

    def _consumer(self, consumer_index: int) -> None:
        ee.Initialize(opt_url=EE_END_POINT)
        chunk_batch = []

        while (chunk_task := self._chunk_queue.get()) is not None:
            max_retries = 3
            last_error = ""
            chunk = None
            for attempt in range(max_retries):
                try:
                    payload, translateY, translateX = self._image_generator(chunk_task)
                    self._report_queue.put(("INFO", f"Requesting image pixels for chunk... {translateY, translateX}"))
                    chunk = ee.data.computePixels(payload)
                except ee.ee_exception.EEException as e:
                    if "Total request size" in str(e):
                        try:
                            match = re.search(r'(?:Total request size) \((\d+)(?= bytes\) must)', str(e))
                            factor = math.ceil(int(match.group(1)) / GEE_REQUEST_LIMIT_MB)
                            bands = payload['expression'].bandNames().getInfo()
                            band_chunk_size = len(bands) // factor
                            self._report_queue.put(("INFO", f"{str(e)} attempting band subsets of {band_chunk_size} / {len(bands)}. ... {translateY, translateX}"))
                            
                            chunk_arrays = []
                            for i in range(factor):
                                s = band_chunk_size * i
                                e_idx = min(len(bands), s + band_chunk_size)
                                chunk_arrays.append(ee.data.computePixels(payload | {"bandIds": bands[s:e_idx]}))
                            
                            chunk = rfn.merge_arrays(chunk_arrays, flatten=True)
                            chunk = chunk.reshape(self._chunk_sizes[-2:])
                            break
                        except Exception as inner_e:
                            last_error = str(inner_e)
                            self._report_queue.put(("WARNING", f"Attempt {attempt+1}/{max_retries} failed (band split): {type(inner_e)} {inner_e} {translateY, translateX}"))
                    else:
                        last_error = str(e)
                        self._report_queue.put(("WARNING", f"Attempt {attempt+1}/{max_retries} failed (EEException): {type(e)} {e} {translateY, translateX}"))
                
                    if "Too Many Requests" in last_error and attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        break

                except Exception as e:
                    self._report_queue.put(("Error", f"Failed to download chunk: {type(e)} {e} {translateY, translateX}"))
                    break
            
            if chunk is None:
                self._report_queue.put(("ERROR", f"Failed to download chunk: {last_error} {translateY, translateX}"))
                continue

            self._report_queue.put(("INFO", f"Processing chunk array... {translateY, translateX}"))
            try:
                self._report_queue.put(("INFO", f"Appending chunk {chunk.shape} to consumer {consumer_index} ... {translateY, translateX}"))
                chunk_batch.append((chunk, translateY, translateX))
                self._report_queue.put(("INFO", f"Consumer {consumer_index} contains {len(chunk_batch)} chunks..."))

                if len(chunk_batch) == self._io_limit:
                    self._write_array_batch(chunk_batch)
                    chunk_batch.clear()

            except Exception as e:
                self._report_queue.put(("ERROR", f"Failed to process chunks for path {self._source_path}: {type(e)} {e} {translateY, translateX}"))

                # reporting failure to watcher and skipping entire batch
                for _, translateY, translateX in chunk_batch:
                    self._result_queue.put((translateY, translateX))
                chunk_batch.clear()

        if len(chunk_batch) > 0:
            self._write_array_batch(chunk_batch)
            chunk_batch.clear()

        self._report_queue.put(("INFO", f"Consumer {consumer_index} completed. exiting..."))
        self._result_queue.put(None)
