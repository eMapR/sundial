import ee
import geopandas as gpd
import logging
import numpy as np
import multiprocessing as mp
import os
import xarray as xr

from pathlib import Path
from shapely.geometry import box
from typing import Any, Literal, Optional

from constants import APPEND_DIM, FILE_EXT_MAP, EE_END_POINT
from pipeline.utils import get_utm_zone


class Downloader:
    """
    A class for downloading images from Google Earth Engine via squares and date filters.

    Args:
        file_type (Literal["GEO_TIFF", "ZARR", "NPY", "NUMPY_NDARRAY"]): The file type to save the image data as.
        overwrite (bool): A flag to overwrite existing image data.
        scale (int): The scale to use for projecting image.
        pixel_edge_size (int): The edge size to use to calculate padding.
        projection (str): A str flag to reproject the image data if set.

        chip_data_path (str): The path to save the image data to.
        meta_data_path (str): The path to the meta data file with coordinates.
        meta_data_parser (callable): A callable to parse the meta data file.
        ee_image_factory (callable): A callable to generate the image expression.
        image_reshaper (callable): A callable to reshape the image data.

        num_workers (int): The number of workers to use for the parallel download process.
        io_limit (int): The number of io requests to make at a time.
        logger (logging.Logger): Instiantiated python logger.

    Methods:
        start(): Starts the parallel download process and performs the necessary checks.
    """

    def __init__(
            self,
            file_type: Literal["GEO_TIFF", "ZARR", "NPY", "NUMPY_NDARRAY"],
            overwrite: bool,
            scale: int,
            pixel_edge_size: int,
            buffer: int,
            projection: bool,

            chip_data_path: str,
            meta_data: gpd.GeoDataFrame,
            meta_data_parser: callable,
            ee_image_factory: callable,
            image_reshaper: callable,

            num_workers: int,
            io_limit: int,
            logger: logging.Logger,

            pixel_grid: Optional[bool] = True,
            reproject: Optional[bool] = True,
            parser_kwargs: Optional[dict] = {},
            factory_kwargs: Optional[dict] = {},
            reshaper_kwargs: Optional[dict] = {},
    ) -> None:
        self._file_type = file_type
        self._overwrite = overwrite
        self._scale = scale
        self._pixel_edge_size = pixel_edge_size
        self._buffer = buffer
        self._projection = projection

        self._chip_data_path = chip_data_path
        self._meta_data = meta_data
        self._meta_data_parser = meta_data_parser
        self._ee_image_factory = ee_image_factory
        self._image_reshaper = image_reshaper

        self._num_workers = num_workers
        self._io_limit = io_limit
        self._logger = logger

        self._pixel_grid = pixel_grid
        self._reproject = reproject
        self._parser_kwargs = parser_kwargs
        self._factory_kwargs = factory_kwargs
        self._reshaper_kwargs = reshaper_kwargs
        self._meta_size = len(self._meta_data)

    def start(self) -> None:
        """
        Starts the parallel download process and performs the necessary checks.
        """
        self._watcher()

    def _watcher(self) -> None:
        # intialize the multiprocessing manager and queues
        manager = mp.Manager()
        payload_queue = manager.Queue()
        image_queue = manager.Queue()
        result_queue = manager.Queue()
        report_queue = manager.Queue()
        chip_lock = manager.Lock() if self._file_type == "ZARR" else None

        # create reporter to aggregate logs
        reporter = mp.Process(
            target=self._reporter,
            args=[report_queue],
            daemon=True)
        reporter.start()

        # filling image queue with GEE image payloads
        generators = set()
        report_queue.put(
            ("INFO", f"Starting generation of {self._meta_size} image payloads..."))
        [payload_queue.put(i) for i in range(self._meta_size)]
        [payload_queue.put(None) for _ in range(self._num_workers)]
        for _ in range(self._num_workers):
            image_generator = mp.Process(
                target=self._image_generator,
                args=(payload_queue,
                      image_queue,
                      report_queue),
                daemon=True)
            image_generator.start()
            generators.add(image_generator)
        [g.join() for g in generators]
        [image_queue.put(None) for _ in range(self._num_workers)]

        # initialize and start parallel downloads
        consumers = set()
        downloads = image_queue.qsize()
        report_queue.put(("INFO",
                          f"Starting download of {downloads-self._num_workers} points of interest..."))
        for consumer_index in range(self._num_workers):
            image_consumer = mp.Process(
                target=self._image_consumer,
                args=(
                    image_queue,
                    result_queue,
                    report_queue,
                    chip_lock,
                    consumer_index),
                daemon=True)
            image_consumer.start()
            consumers.add(image_consumer)

        # watch for results from consumers
        downloads_completed = 0
        consumers_completed = 0
        while consumers_completed < self._num_workers:
            # TODO: perform result checks and monitor gee processes
            result = result_queue.get()
            if result is not None:
                downloads_completed += 1
                report_queue.put(
                    ("INFO", f"{downloads_completed}/{self._meta_size} Downloads completed. {result}"))
            else:
                consumers_completed += 1
                report_queue.put(
                    ("INFO", f"{consumers_completed}/{self._num_workers} Consumers completed."))
        report_queue.put(None)
        [c.join() for c in consumers]
        reporter.join()

    def _reporter(self, report_queue: mp.Queue) -> None:
        while (report := report_queue.get()) is not None:
            level, message = report
            if self._logger is not None:
                match level:
                    case "DEBUG":
                        self._logger.debug(message)
                    case "INFO":
                        self._logger.info(message)
                    case "WARNING":
                        self._logger.warning(message)
                    case "ERROR":
                        self._logger.error(message)
                    case "CRITICAL":
                        self._logger.critical(message)
            else:
                print(level, message)

    def _image_generator(self,
                         payload_queue: mp.Queue,
                         image_queue: mp.Queue,
                         report_queue: mp.Queue,
                         ) -> None:
        ee.Initialize(opt_url=EE_END_POINT)
        file_ext = FILE_EXT_MAP[self._file_type]
        while (index := payload_queue.get()) is not None:
            try:
                # reading meta data from xarray
                square, start_date, end_date = self._meta_data_parser(self._meta_data,
                                                                      index,
                                                                      **self._parser_kwargs)
                
                if self._buffer > 0:
                    expand_distance = self._buffer * self._scale

                    minx, miny, maxx, maxy = square.bounds
                    square = box(minx - expand_distance, miny - expand_distance,
                                maxx + expand_distance, maxy + expand_distance)

                square_coords = list(square.boundary.coords)
                point_coords = list(square.centroid.coords)

                # checking for existing files and skipping if file found
                if self._file_type != "ZARR":
                    chip_data_path = chip_var_path = os.path.join(chip_data_path, f"{index}")
                else:
                    chip_data_path = self._chip_data_path
                    chip_var_path = os.path.join(self._chip_data_path, f"{APPEND_DIM}", f"{index}")
                if not self._overwrite and os.path.exists(chip_var_path):
                    report_queue.put(("INFO", f"Files for chip {index:08d} already exists. Skipping... {square_coords}"))
                    continue

                # getting utm zone and epsg code for reprojection
                match self._projection:
                    case "UTM":
                        epsg_str = get_utm_zone(point_coords)
                    case _:
                        epsg_str = self._projection

                # creating payload for each square to send to GEE
                report_queue.put(("INFO", f"Creating image payload for square {index:08d}... {square_coords}"))
                image = self._ee_image_factory(
                    square_coords,
                    start_date,
                    end_date,
                    epsg_str,
                    **self._factory_kwargs)

                # reprojecting the image if necessary
                if self._reproject and epsg_str is not None:
                    report_queue.put(("INFO", f"Reprojecting image payload square {index:08d} to {epsg_str}... {square_coords}"))
                    image = image.reproject(crs=epsg_str, scale=self._scale)

                # encoding the image for the image consumer
                payload = {
                    "expression": ee.serializer.encode(image),
                    "fileFormat": self._file_type if self._file_type != "ZARR" else "NUMPY_NDARRAY"
                }
                
                if self._pixel_grid:
                    translateX, translateY = max(square_coords, key=lambda coords: (coords[1], -coords[0]))
                    payload["grid"] = {
                        'dimensions': {
                            'width': self._pixel_edge_size + self._buffer*2,
                            'height': self._pixel_edge_size + self._buffer*2
                        },
                        'affineTransform': {
                            'scaleX': self._scale,
                            'shearX': 0,
                            'translateX': translateX,
                            'shearY': 0,
                            'scaleY': -self._scale,
                            'translateY': translateY
                        },
                        'crsCode': epsg_str,
                    }

                # sending expression payload to the image consumer
                image_queue.put((payload, index, square_coords, point_coords, chip_data_path))
            except Exception as e:
                report_queue.put(("CRITICAL", f"Failed to create image payload for square {index:08d} skipping: {type(e)} {e} {square_coords}"))

    def _image_consumer(self,
                        image_queue: mp.Queue,
                        result_queue: mp.Queue,
                        report_queue: mp.Queue,
                        chip_lock: Any,
                        consumer_index: int) -> None:
        ee.Initialize(opt_url=EE_END_POINT)

        if self._file_type == "ZARR":
            square_name_batch = []
            xarr_chip_batch = []
            batch_index = 0
            batch_size = 0

        while (image_task := image_queue.get()) is not None:
            payload, index, square_coords, point_coords, chip_data_path = image_task
            try:
                # google will internally retry the request if it fails
                report_queue.put(("INFO", f"Requesting image pixels for square... {index:08d} {square_coords}"))

                payload["expression"] = ee.deserializer.decode(payload["expression"])
                chip = ee.data.computePixels(payload)
            except Exception as e:
                report_queue.put(("ERROR", f"Failed to download square {index:08d}: {type(e)} {e} {square_coords}"))
                continue

            report_queue.put(
                ("INFO", f"Processing square array for chip format {self._file_type} ... {index:08d} {square_coords}"))
            try:
                match self._file_type:
                    case "NPY" | "GEO_TIFF":
                        # TODO: perform reshaping along times for non zarr file types
                        report_queue.put(("INFO", f"Writing chip {chip.shape} to {self._file_type} file... {index:08d} {square_coords}"))
                        out_file = Path(chip_data_path)
                        out_file.write_bytes(chip)

                    case "NUMPY_NDARRAY":
                        # TODO: perform reshaping along times for non zarr file types
                        report_queue.put(("INFO", f"Writing chip {chip.shape} to {self._file_type} file... {index:08d} {square_coords}"))
                        np.save(chip_data_path, chip)

                    case "ZARR":
                        square_name_batch.append(square_coords)
                        batch_size += 1

                        # reshaping from (D*C, H, W) to (C, D, H, W)
                        report_queue.put(("INFO", f"Reshaping square {chip.shape} for {self._file_type} to pixel size {self._pixel_edge_size+self._buffer*2}... {index:08d} {square_coords}"))
                        xarr_chip = self._image_reshaper(chip,
                                                         index,
                                                         self._pixel_edge_size+self._buffer*2,
                                                         square_coords,
                                                         point_coords,
                                                         **self._reshaper_kwargs)

                        # collecting xr data arrays into list for batch writing into xr dataset
                        report_queue.put(("INFO", f"Appending xarr chip {xarr_chip.shape} to consumer {consumer_index} chip batch {batch_index}... {index:08d} {square_coords}"))
                        xarr_chip_batch.append(xarr_chip)
                        report_queue.put(("INFO", f"Consumer {consumer_index} batch {batch_index} contains {batch_size} chips..."))

                        # attempt to merge batch of dataarrays and write to disk
                        if batch_size == self._io_limit:
                            self._write_array_batch(
                                xarr_chip_batch,
                                square_name_batch,
                                batch_index,
                                batch_size,
                                chip_data_path,
                                report_queue,
                                result_queue,
                                chip_lock,
                                consumer_index)

                            # resetting batch
                            square_name_batch.clear()
                            xarr_chip_batch.clear()
                            batch_index += 1
                            batch_size = 0

            except Exception as e:
                report_queue.put(("ERROR", f"Failed to process chips(s) for path {chip_data_path}: {type(e)} {e} {index:08d} {square_coords}"))

                # reporting failure to watcher and skipping entire batch
                for name in square_name_batch:
                    result_queue.put(name)

                # cleaning potentially corrupted files
                if self._file_type == "NPY":
                    try:
                        out_file.unlink(missing_ok=True)
                    except Exception as e:
                        report_queue.put(("ERROR", f"Failed to clean chip file in {chip_data_path}: {type(e)} {e} {index:08d} {square_coords}"))
                # TODO: clear potential writes to zarr
                if self._file_type == "ZARR":
                    square_name_batch.clear()
                    xarr_chip_batch.clear()
                    batch_size = 0

        # writing any remaining data in batch lists to disk
        if self._file_type == "ZARR" and batch_size > 0:
            self._write_array_batch(
                xarr_chip_batch,
                square_name_batch,
                batch_index,
                batch_size,
                chip_data_path,
                report_queue,
                result_queue,
                chip_lock,
                consumer_index)

        report_queue.put(("INFO", f"Consumer {consumer_index} completed. exiting..."))
        result_queue.put(None)

    def _write_array_batch(self,
                           xarr_chip_batch: list[xr.DataArray],
                           square_name_batch: list[str],
                           batch_index: int,
                           batch_size: int,
                           chip_data_path: str,
                           report_queue: mp.Queue,
                           result_queue: mp.Queue,
                           chip_lock: mp.Queue,
                           consumer_index: int) -> None:
        # merging and writing or appending chip batch as dataset to zarr
        report_queue.put(("INFO", f"Merging and writing consumer {consumer_index} chip batch {batch_index} of size {batch_size} to {chip_data_path}..."))
        
        # TODO: Replace with 'combine_by_coords' method. It's a pain to manage coordinates when the crs can vary. This should probably be done partially in the reshaper.
        #       A potential draw back to this laziness is not being able to select a box from ALL variables since x and y dims only represent coordinates in the single image.
        #       There's also no guarantee that there will not be boundary artifacts from the compute pixels call per bounding box if we do implement stiching.
        xarr_chip_batch = xr.concat(xarr_chip_batch, dim=APPEND_DIM, coords='all')
        with chip_lock:
            if os.path.exists(chip_data_path):
                xarr_chip_batch.to_zarr(store=chip_data_path, append_dim=APPEND_DIM, mode="a")
            else:
                xarr_chip_batch.to_zarr(store=chip_data_path)

        # reporting batch completion to watcher
        for name in square_name_batch:
            result_queue.put(name)
