import ee
import geopandas as gpd
import math
import numpy as np
import regex as re
import time

from numpy.lib import recfunctions as rfn
from shapely.geometry import box

from constants import EE_END_POINT, GEE_REQUEST_LIMIT_MB, TOO_MANY_REQUEST_STR
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
        try:
            translateY, translateX = chunk
            chunk_box = box(translateX, translateY - self._grid_y_size, translateX + self._grid_x_size, translateY)
            chunk_coords = list(chunk_box.boundary.coords)
            
            self._report_queue.put(("INFO", f"Creating image payload for chunk... {translateY, translateX}"))
            image = self._ee_factory.create_ee_image(chunk_coords)
            image = image.reproject(crs=self._epsg_str, scale=self._scale)

            payload = {
                "expression": image,
                "fileFormat": "NUMPY_NDARRAY"
            }
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
            self._report_queue.put(("CRITICAL", f"Failed to create image payload for chunk skipping: {type(e)} {e} {translateY, translateX}"))

    def _consumer(self, consumer_index: int) -> None:
        ee.Initialize(opt_url=EE_END_POINT)
        chunk_batch = []
        exp_band_count = np.prod(self._chunk_sizes[:2]).item()
        max_retries = 3
        
        while (chunk_task := self._chunk_queue.get()) is not None:
            attempts = 0
            while attempts < max_retries:
                last_error = ""
                chunk = None
                try:
                    payload, translateY, translateX = self._image_generator(chunk_task)
                    if (band_count := len(payload["expression"].bandNames().getInfo())) != exp_band_count:
                        self._report_queue.put(("WARNING", f"Band count {band_count} != {exp_band_count} mismatch found for chunk skipping... {translateY, translateX}"))
                    else:
                        self._report_queue.put(("INFO", f"Requesting image pixels for chunk... {translateY, translateX}"))
                        chunk = ee.data.computePixels(payload)
                    attempts += max_retries
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
                                subchunk = ee.data.computePixels(payload | {"bandIds": bands[s:e_idx]})
                                chunk_arrays.append(subchunk)
                            
                            chunk = rfn.merge_arrays(chunk_arrays, flatten=True)
                            chunk = chunk.reshape(self._chunk_sizes[-2:])
                            attempts += max_retries
                        except Exception as inner_e:
                            last_error = str(inner_e)
                    else:
                        last_error = str(e)
                
                    if last_error == TOO_MANY_REQUEST_STR and attempts < max_retries:
                        self._report_queue.put(("WARNING", f"Attempt {attempts+1}/{max_retries} failed (EEException): {type(e)} {e} {translateY, translateX}"))
                        time.sleep(2 ** attempts)
                        attempts += 1
                        continue
                    else:
                        self._report_queue.put(("ERROR", f"Attempt {attempts+1}/{max_retries} failed (EEException): {type(e)} {e} {translateY, translateX}"))
                        attempts += max_retries

                except Exception as e:
                    self._report_queue.put(("ERROR", f"Failed to download chunk: {type(e)} {e} {translateY, translateX}"))
                    attempts += max_retries

            if chunk is None:
                continue

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
