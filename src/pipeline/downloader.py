import argparse
import ee
import numpy as np
import multiprocessing as mp
import os
import time
import utm
import xarray as xr
import yaml

from datetime import datetime
from pathlib import Path
from typing import Literal

from utils import parse_meta_data, estimate_download_size, lt_image_generator, zarr_reshape
from logger import get_logger
from settings import FILE_EXT_MAP


EE_END_POINT = 'https://earthengine-highvolume.googleapis.com'


class Downloader:
    """
    A class for downloading images from Google Earth Engine via squares and date filters.

    Args:
        start_date (datetime): The start date to filter image collection.
        end_date (datetime): The end date to filter image collection.
        file_type (Literal["GEO_TIFF", "ZARR", "NPY", "NUMPY_NDARRAY"]): The file type to save the image data as.
        overwrite (bool): A flag to overwrite existing image data.
        scale (int): The scale to use for projecting image.
        pixel_edge_size (int): The edge size to use to calculate padding.
        reprojection (str): A str flag to reproject the image data if set.
        overlap_band (bool): A flag to add a band that labels by pixel in the square whether the it overlaps the geometry.
        back_step (int): The number of years to step back from the end date.

        chip_data_path (str): The path to save the image data to.
        anno_data_path (str): The path to the strata map file.
        meta_data_path (str): The path to the meta data file with coordinates.

        num_workers (int): The number of workers to use for the parallel download process.
        retries (int): The number of retries to use for the download process.
        request_limit (int): The number of requests to make at a time.
        ignore_size_limit (bool): A flag to ignore the size limits for the image data.
        io_lock (bool): A flag to use a lock for the io process.

        log_path (str): The path to save the log file to.
        log_name (str): The name of the log file.

    Methods:
        start(): Starts the parallel download process and performs the necessary checks.
    """
    _size_limit: int = 65
    _band_limit: int = 1024
    _pixel_limit: int = 3.2e4

    def __init__(
            self,
            start_date: datetime | None,
            end_date: datetime | None,
            file_type: Literal["GEO_TIFF", "ZARR", "NPY", "NUMPY_NDARRAY"],
            overwrite: bool,
            scale: int,
            pixel_edge_size: int,
            reprojection: bool,
            overlap_band: bool,
            back_step: int,
            chip_data_path: str,
            anno_data_path: str,
            strata_map_path: str,
            meta_data_path: str,
            num_workers: int,
            io_limit: int,
            log_path: str,
            log_name: str,
    ) -> None:
        self._start_date = start_date
        self._end_date = end_date
        self._file_type = file_type
        self._overwrite = overwrite
        self._scale = scale
        self._pixel_edge_size = pixel_edge_size
        self._reprojection = reprojection
        self._overlap_band = overlap_band
        self._back_step = back_step
        self._chip_data_path = chip_data_path
        self._anno_data_path = anno_data_path
        self._strata_map_path = strata_map_path
        self._meta_data_path = meta_data_path
        self._num_workers = num_workers
        self._io_limit = io_limit
        self._log_path = log_path
        self._log_name = log_name

        # TODO: Parameterize the image generator callable
        self._image_gen_callable = lt_image_generator

        # TODO: Perform attribute checks for meta_data files
        self._meta_data = xr.open_zarr(self._meta_data_path)
        self._meta_size = self._meta_data["index"].size

        if not self._overwrite:
            self._existing_chips = os.listdir(self._chip_data_path)
            if self._overlap_band:
                self._existing_annos = os.listdir(self._anno_data_path)

    def start(self) -> None:
        """
        Starts the parallel download process and performs the necessary checks.
        """
        self._watcher()

    def _watcher(self) -> None:
        # intialize the multiprocessing manager and queues
        manager = mp.Manager()
        image_queue = manager.Queue()
        result_queue = manager.Queue()
        report_queue = manager.Queue()
        chip_lock = manager.Lock() if self._file_type == "ZARR" else None
        anno_lock = manager.Lock() if self._file_type == "ZARR" else None
        workers = set()

        # create reporter to aggregate logs
        reporter = mp.Process(
            target=self._reporter,
            args=[report_queue],
            daemon=True)
        workers.add(reporter)

        # initializing image generator to create GEE images
        image_generator = mp.Process(
            target=self._image_generator,
            args=(
                image_queue,
                result_queue,
                report_queue),
            daemon=True)
        workers.add(image_generator)

        # initializing number of parallel downloads
        for consumer_index in range(self._num_workers):
            image_consumer = mp.Process(
                target=self._image_consumer,
                args=(
                    image_queue,
                    result_queue,
                    report_queue,
                    chip_lock,
                    anno_lock,
                    consumer_index),
                daemon=True)
            workers.add(image_consumer)

        # start download and watch for results
        report_queue.put(("INFO",
                          f"Starting download of {self._meta_size} points of interest..."))
        start_time = time.time()
        [w.start() for w in workers]
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
                    ("INFO", f"{consumers_completed}/{self._num_workers} Consumers completed. {result}"))

        end_time = time.time()
        report_queue.put(("INFO",
                          f"Download completed in {(end_time - start_time) / 60:.2} minutes."))
        report_queue.put(None)
        [w.join() for w in workers]

    def _reporter(self, report_queue: mp.Queue) -> None:
        logger = get_logger(self._log_path, self._log_name)
        while (report := report_queue.get()) is not None:
            level, message = report
            match level:
                case "DEBUG":
                    logger.debug(message)
                case "INFO":
                    logger.info(message)
                case "WARNING":
                    logger.warning(message)
                case "ERROR":
                    logger.error(message)
                case "CRITICAL":
                    logger.critical(message)
                case "EXIT":
                    return

    def _image_generator(self,
                         image_queue: mp.Queue,
                         result_queue: mp.Queue,
                         report_queue: mp.Queue
                         ) -> None:
        ee.Initialize(opt_url=EE_END_POINT)
        file_ext = FILE_EXT_MAP[self._file_type]
        for idx in range(self._meta_size):
            try:
                # reading meta data from xarray
                geometry_coords, \
                    point_coords, \
                    point_name, \
                    square_coords, \
                    square_name, \
                    start_date, \
                    end_date, \
                    attributes \
                    = parse_meta_data(self._meta_data, idx, self._back_step)
                if start_date is None:
                    start_date = self._start_date
                if end_date is None:
                    end_date = self._end_date

                # checking for existing files and skipping if file found
                if self._file_type != "ZARR":
                    chip_file_name = f"{square_name}.{file_ext}"
                    chip_data_path = os.path.join(
                        self._chip_data_path, chip_file_name)
                    if self._overlap_band:
                        anno_file_name = f"{square_name}.{file_ext}"
                        anno_data_path = os.path.join(
                            self._anno_data_path, anno_file_name)
                else:
                    chip_file_name = square_name
                    chip_data_path = self._chip_data_path
                    if self._overlap_band:
                        anno_file_name = square_name
                        anno_data_path = self._anno_data_path
                if not self._overwrite:
                    if chip_file_name in self._existing_chips:
                        report_queue.put(("INFO",
                                          f"Files already exists. Skipping... {square_name}"))
                        result_queue.put(square_name)
                        continue

                # creating payload for each square to send to GEE
                report_queue.put(
                    ("INFO", f"Creating image payload for square... {square_name}"))
                image = self._image_gen_callable(
                    start_date,
                    end_date,
                    square_coords,
                    self._scale,
                    self._overlap_band,
                    geometry_coords)

                # getting utm zone and epsg code for reprojection
                match self._reprojection:
                    case "UTM":
                        revserse_point = reversed(point_coords)
                        utm_zone = utm.from_latlon(*revserse_point)[-2:]
                        epsg_prefix = "EPSG:326" if point_coords[1] > 0 else "EPSG:327"
                        epsg_code = f"{epsg_prefix}{utm_zone[0]}"
                    case _:
                        epsg_code = self._reprojection

                # reprojecting the image if necessary
                if epsg_code is not None:
                    report_queue.put(
                        ("INFO", f"Reprojecting image payload square to {epsg_code}... {square_name}"))
                    image = image.reproject(
                        crs=epsg_code, scale=self._scale)

                # encoding the image for the image consumer
                payload = {
                    "expression": ee.serializer.encode(image),
                    "fileFormat": self._file_type if self._file_type != "ZARR" else "NUMPY_NDARRAY",
                }

                # sending expression payload to the image consumer
                image_queue.put(
                    (payload, square_name, point_name, chip_data_path, anno_data_path, attributes))
            except Exception as e:
                report_queue.put(
                    ("CRITICAL", f"Failed to create image payload for square skipping: {type(e)} {e} {square_name}"))
                result_queue.put(square_name)
        [image_queue.put(None) for _ in range(self._num_workers)]

    def _image_consumer(self,
                        image_queue: mp.Queue,
                        result_queue: mp.Queue,
                        report_queue: mp.Queue,
                        chip_lock: mp.Lock,
                        anno_lock: mp.Lock,
                        consumer_index: int) -> None:
        ee.Initialize(opt_url=EE_END_POINT)
        with open(self._strata_map_path, "r") as f:
            strata_map = yaml.safe_load(f)

        if self._file_type == "ZARR":
            square_name_batch = []
            xarr_chip_batch = []
            anno_chip_batch = []
            batch_index = 0
            batch_size = 0

        while (image_task := image_queue.get()) is not None:
            payload, square_name, point_name, chip_data_path, anno_data_path, attributes = image_task
            try:
                # google will internally retry the request if it fails
                report_queue.put(("INFO",
                                  f"Requesting image pixels for square... {square_name}"))

                payload["expression"] = ee.deserializer.decode(
                    payload["expression"])
                array_chip = ee.data.computePixels(payload)
            except Exception as e:
                report_queue.put(
                    ("ERROR", f"Failed to download square: {type(e)} {e} {square_name}"))
                continue

            report_queue.put(
                ("INFO", f"Processing square array for chip format {self._file_type} ... {square_name}"))
            try:
                match self._file_type:
                    case "NPY" | "GEO_TIFF":
                        # TODO: perform reshaping along years for non zarr file types
                        report_queue.put((
                            "INFO", f"Writing chip {array_chip.shape} to {self._file_type} file... {square_name}"))
                        out_file = Path(chip_data_path)
                        out_file.write_bytes(array_chip)

                    case "NUMPY_NDARRAY":
                        # TODO: perform reshaping along years for non zarr file types
                        report_queue.put((
                            "INFO", f"Writing chip {array_chip.shape} to {self._file_type} file... {square_name}"))
                        np.save(chip_data_path, array_chip)

                    case "ZARR":
                        square_name_batch.append(square_name)
                        batch_size += 1

                        # reshaping from (D*C, H, W) to (D, H, W, D)
                        report_queue.put((
                            "INFO", f"Reshaping square {array_chip.shape} for {self._file_type} to pizel size {self._pixel_edge_size}... {square_name}"))
                        xarr_chip, xarr_anno = zarr_reshape(array_chip,
                                                            self._pixel_edge_size,
                                                            square_name,
                                                            point_name,
                                                            attributes,
                                                            strata_map)

                        # collecting xr data arrays into list for batch writing
                        report_queue.put(
                            ("INFO", f"Appending xarr chip {xarr_chip.shape} to consumer {consumer_index} chip batch {batch_index}... {square_name}"))
                        xarr_chip_batch.append(xarr_chip)
                        if xarr_anno is not None:
                            report_queue.put(
                                ("INFO", f"Appending xarr annotations {xarr_anno.shape} to consumer {consumer_index} anno batch {batch_index}... {square_name}"))
                            anno_chip_batch.append(xarr_anno)
                        report_queue.put(
                            ("INFO", f"Consumer {consumer_index} batch {batch_index} contains {batch_size} chips..."))

                        # attempt to merge batch of dataarrays and write to disk
                        if batch_size == self._io_limit:
                            self._write_array_batch(
                                xarr_chip_batch,
                                anno_chip_batch,
                                square_name_batch,
                                batch_index,
                                batch_size,
                                chip_data_path,
                                anno_data_path,
                                report_queue,
                                result_queue,
                                chip_lock,
                                anno_lock,
                                consumer_index,)

                            # resetting batch
                            square_name_batch.clear()
                            xarr_chip_batch.clear()
                            anno_chip_batch.clear()
                            batch_index += 1
                            batch_size = 0

            except Exception as e:
                report_queue.put(
                    ("ERROR", f"Failed to process chips(s) for path {chip_data_path}: {type(e)} {e} {square_name}"))

                # reporting failure to watcher and skipping entire batch
                for name in square_name_batch:
                    result_queue.put(name)

                # cleaning potentially corrupted files
                if self._file_type == "NPY":
                    try:
                        out_file.unlink(missing_ok=True)
                    except Exception as e:
                        report_queue.put(
                            ("ERROR", f"Failed to clean chip file in {chip_data_path}: {type(e)} {e} {square_name}"))
                # TODO: clear potential writes to zarr
                if self._file_type == "ZARR":
                    square_name_batch.clear()
                    xarr_chip_batch.clear()
                    anno_chip_batch.clear()
                    batch_size = 0

        # writing any remaining data in batch lists to disk
        if self._file_type == "ZARR" and batch_size > 0:
            self._write_array_batch(
                xarr_chip_batch,
                anno_chip_batch,
                square_name_batch,
                batch_index,
                batch_size,
                chip_data_path,
                anno_data_path,
                report_queue,
                result_queue,
                chip_lock,
                anno_lock,
                consumer_index)

        report_queue.put(
            ("INFO", f"Consumer {consumer_index} completed. exiting..."))
        result_queue.put(None)

    def _write_array_batch(self,
                           xarr_chip_batch: list[xr.DataArray],
                           anno_chip_batch: list[xr.DataArray],
                           square_name_batch: list[str],
                           batch_index: int,
                           batch_size: int,
                           chip_data_path: str,
                           anno_data_path: str,
                           report_queue: mp.Queue,
                           result_queue: mp.Queue,
                           chip_lock: mp.Queue,
                           anno_lock: mp.Queue,
                           consumer_index: int) -> None:
        # merging and writing or appending chip batch as dataset to zarr
        report_queue.put(
            ("INFO", f"Merging and writing consumer {consumer_index} chip batch {batch_index} of size {batch_size} to {chip_data_path}..."))
        xarr_chip_batch = xr.merge(xarr_chip_batch)
        with chip_lock:
            xarr_chip_batch.to_zarr(
                store=chip_data_path, mode="a")

        # merging and writing or appending anno batch as dataset to zarr if not empty
        if anno_chip_batch[0] is not None:
            report_queue.put(
                ("INFO", f"Merging and writing consumer {consumer_index} annotation batch {batch_index} of size {batch_size} to {anno_data_path}..."))
            xarr_anno_batch = xr.merge(anno_chip_batch)
            with anno_lock:
                xarr_anno_batch.to_zarr(
                    store=anno_data_path, mode="a")

        # reporting batch completion to watcher
        for name in square_name_batch:
            result_queue.put(name)


def parse_args():
    parser = argparse.ArgumentParser(description='Downloader Arguments')
    return vars(parser.parse_args())


def main(**kwargs):
    from settings import DOWNLOADER as config, SAMPLE_PATH
    os.makedirs(SAMPLE_PATH, exist_ok=True)

    downloader = Downloader(**config)
    downloader.start()


if __name__ == "__main__":
    main(**parse_args())
