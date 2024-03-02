import argparse
import ee
import numpy as np
import multiprocessing as mp
import os
import time
import utm
import xarray as xr
import yaml
import zarr

from datetime import datetime
from pathlib import Path
from typing import Literal
from zarr.errors import PathNotFoundError, GroupNotFoundError, ArrayNotFoundError

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
            retries: int,
            ignore_size_limit: bool,
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
        self._retries = retries
        self._ignore_size_limit = ignore_size_limit
        self._io_limit = io_limit
        self._log_path = log_path
        self._log_name = log_name

        # TODO: Parameterize the image generator callable
        # TODO: Perform attribute checks
        self._image_gen_callable = lt_image_generator
        self._meta_data = xr.open_zarr(self._meta_data_path)
        self._meta_size = self._meta_data["index"].size

    def start(self) -> None:
        """
        Starts the parallel download process and performs the necessary checks.
        """
        if not self._ignore_size_limit:
            # this assumes all squares are the same size
            _, _, _, square_coords, start_date, end_date, _ = parse_meta_data(
                self._meta_data, 0, self._back_step)
            test_area = ee.Geometry.Polygon(square_coords)
            if start_date is None:
                start_date = self._start_date
            if end_date is None:
                end_date = self._end_date

            test_image = self._image_gen_callable(
                start_date, end_date, test_area, self._scale)
            test_size, test_pixels, test_bands = estimate_download_size(
                test_image, test_area, self._scale)
            if test_size > self._size_limit:
                raise ValueError(
                    f"Image size of {test_size}MB exceeds size limit of {self._size_limit}MB. Please reduce the size of the image.")
            if test_pixels**.5 > self._pixel_limit:
                raise ValueError(
                    f"Pixel count of {test_pixels} exceeds pixel limit of {self._pixel_limit}. Please reduce the pixels of the image.")
            if test_bands > self._band_limit:
                raise ValueError(
                    f"Band count of {test_bands} exceeds band limit of {self._band_limit}. Please reduce the bands of the image.")
        self._watcher()

    def _watcher(self) -> None:
        # intialize the multiprocessing manager and queues
        manager = mp.Manager()
        image_queue = manager.Queue()
        array_queue = manager.Queue(self._io_limit*self._num_workers)
        result_queue = manager.Queue()
        report_queue = manager.Queue()
        workers = set()

        # create reporter, image generator, and consumer processes
        reporter = mp.Process(
            target=self._reporter,
            args=[report_queue],
            daemon=True)
        workers.add(reporter)

        image_generator = mp.Process(
            target=self._image_generator,
            args=(
                image_queue,
                result_queue,
                report_queue),
            daemon=True)
        workers.add(image_generator)

        for _ in range(self._num_workers):
            image_consumer = mp.Process(
                target=self._image_consumer,
                args=(
                    image_queue,
                    array_queue,
                    result_queue,
                    report_queue),
                daemon=True)
            workers.add(image_consumer)

        writer = mp.Process(
            target=self._writer,
            args=(
                array_queue,
                result_queue,
                report_queue,
            ),
            daemon=True)
        workers.add(writer)

        # start download and watch for results
        report_queue.put(("INFO",
                          f"Starting download of {self._meta_size} points of interest..."))
        start_time = time.time()

        [w.start() for w in workers]
        idx = 0
        while idx < self._meta_size:
            # TODO: perform result checks and monitor gee processes
            result = result_queue.get()
            if result is not None:
                idx += 1
                report_queue.put(
                    ("INFO", f"{idx}/{self._meta_size} Completed. {result}"))
        report_queue.put(None)
        [w.join() for w in workers]

        end_time = time.time()
        report_queue.put(("INFO",
                         f"Download completed in {(end_time - start_time) / 60:.2} minutes."))

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
                    chip_data_path = os.path.join(self._chip_data_path,
                                                  f"{square_name}.{file_ext}")
                    anno_data_path = os.path.join(self._anno_data_path,
                                                  f"{square_name}.{file_ext}")
                    if not self._overwrite and Path(chip_data_path).exists() and Path(anno_data_path).exists():
                        report_queue.put(
                            "INFO", f"File {chip_data_path} already exists. Skipping...")
                        result_queue.put(square_name)
                        continue
                else:
                    chip_data_path = self._chip_data_path
                    anno_data_path = self._anno_data_path
                    if not self._overwrite:
                        try:
                            # opening with read only mode to check for existing zarr groups
                            zarr.open(
                                store=chip_data_path,
                                mode="r")[square_name]
                            zarr.open(
                                store=anno_data_path,
                                mode="r")[square_name]
                            report_queue.put(("INFO",
                                              f"Polygon already exists at path. Skipping... {square_name}"))
                            continue

                        except (PathNotFoundError,
                                GroupNotFoundError,
                                ArrayNotFoundError,
                                KeyError,
                                FileNotFoundError) as e:
                            # capturing valid exceptions and passing to next step
                            report_queue.put(
                                ("INFO", f"Valid exception captured for square: {type(e)}... {square_name}"))
                            pass

                        except Exception as e:
                            # capturing fatal exceptions and skipping to next square
                            report_queue.put(
                                ("CRITICAL", f"Failed to read zarr path {chip_data_path}, zarr group {point_name}, or zarr variable {square_name} skipping: {type(e)} {e}"))
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

                # Reprojecting the image if necessary
                match self._reprojection:
                    case "UTM":
                        revserse_point = reversed(point_coords)
                        utm_zone = utm.from_latlon(*revserse_point)[-2:]
                        epsg_prefix = "EPSG:326" if point_coords[1] > 0 else "EPSG:327"
                        epsg_code = f"{epsg_prefix}{utm_zone[0]}"
                    case _:
                        epsg_code = self._reprojection
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

                # sending payload to the image consumer
                image_queue.put(
                    (payload, square_name, point_name, chip_data_path, anno_data_path, attributes))
            except Exception as e:
                report_queue.put(
                    ("CRITICAL", f"Failed to create image payload for square skipping: {type(e)} {e} {square_name}"))
                result_queue.put(square_name)
        [image_queue.put(None) for i in range(self._num_workers)]

    def _image_consumer(self,
                        image_queue: mp.Queue,
                        array_queue: mp.Queue,
                        result_queue: mp.Queue,
                        report_queue: mp.Queue) -> None:
        ee.Initialize(opt_url=EE_END_POINT)

        while (image_task := image_queue.get()) is not None:
            payload, square_name, point_name, chip_data_path, anno_data_path, attributes = image_task
            attempts = 0

            # attempt to download the image from gee
            while attempts < self._retries:
                attempts += 1
                try:
                    report_queue.put(("INFO",
                                     f"Requesting image pixels for square... {square_name}"))
                    payload["expression"] = ee.deserializer.decode(
                        payload["expression"])
                    array_chip = ee.data.computePixels(payload)
                    break
                except Exception as e:
                    time.sleep(3)
                    report_queue.put(
                        ("WARNING", f"Failed to download square attempt {attempts}/{self._retries}: {type(e)} {e} {square_name}"))
                    if attempts == self._retries:
                        report_queue.put(
                            ("ERROR", f"Max retries reached for square skipping... {square_name}"))
                        result_queue.put(square_name)
                    else:
                        report_queue.put(
                            ("INFO", f"Retrying download for square... {square_name}"))

            # send array to writer if successful
            if array_chip is not None:
                array_queue.put((array_chip, square_name, point_name,
                                chip_data_path, anno_data_path, attributes))

        array_queue.put(None)

    def _writer(self,
                        array_queue: mp.Queue,
                        result_queue: mp.Queue,
                        report_queue: mp.Queue) -> None:
        with open(self._strata_map_path, "r") as f:
            strata_map = yaml.safe_load(f)

        if self._file_type == "ZARR":
            square_name_batch = []
            xarr_chip_batch = []
            anno_chip_batch = []
            batch_index = 0

        completed_consumers = 0
        while completed_consumers < self._num_workers:
            array_task = array_queue.get()
            if array_task is None:
                completed_consumers += 1
                continue
            array_chip, square_name, point_name, chip_data_path, anno_data_path, attributes = array_task
            report_queue.put(
                ("INFO", f"Processing square array to chip format {self._file_type} ... {square_name}"))

            try:
                # TODO: perform reshaping along years for non zarr file types
                match self._file_type:
                    case "NPY" | "GEO_TIFF":
                        report_queue.put((
                            "INFO", f"Writing chip {array_chip.shape} to {self._file_type} file... {square_name}"))
                        out_file = Path(chip_data_path)
                        out_file.write_bytes(array_chip)
                    case "NUMPY_NDARRAY":
                        report_queue.put((
                            "INFO", f"Writing chip {array_chip.shape} to {self._file_type} file... {square_name}"))
                        np.save(chip_data_path, array_chip)
                    case "ZARR":
                        report_queue.put((
                            "INFO", f"Reshaping square {array_chip.shape} for {self._file_type}... {square_name}"))
                        xarr_chip, xarr_anno = zarr_reshape(array_chip,
                                                            self._pixel_edge_size,
                                                            square_name,
                                                            point_name,
                                                            attributes,
                                                            strata_map)

                        # collecting dataarrays for batch writing
                        report_queue.put(
                            ("INFO", f"Appending xarr chip {xarr_chip.shape} to chip batch #{batch_index}... {square_name}"))
                        xarr_chip_batch.append(xarr_chip)
                        if xarr_anno is not None:
                            report_queue.put(
                                ("INFO", f"Appending xarr anno {xarr_anno.shape} to anno batch #{batch_index}... {square_name}"))
                            anno_chip_batch.append(xarr_anno)
                        square_name_batch.append(square_name)
                        batch_size = len(xarr_chip_batch)
                        report_queue.put(
                            ("INFO", f"Current batch #{batch_index} contains {batch_size} chips..."))

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
                                result_queue)

                            # resetting batch
                            square_name_batch.clear()
                            xarr_chip_batch.clear()
                            anno_chip_batch.clear()
                            batch_index += 1

            except Exception as e:
                report_queue.put(
                    ("ERROR", f"Failed to write chip to {chip_data_path}: {type(e)} {e} {square_name}"))
                if self._file_type == "NPY":
                    try:
                        out_file.unlink(missing_ok=True)
                    except Exception as e:
                        report_queue.put(
                            ("ERROR", f"Failed to clean chip file in {chip_data_path}: {type(e)} {e} {square_name}"))

        if self._file_type == "ZARR" and len(xarr_chip_batch) > 0:
            self._write_array_batch(
                xarr_chip_batch,
                anno_chip_batch,
                square_name_batch,
                batch_index,
                batch_size,
                chip_data_path,
                anno_data_path,
                report_queue,
                result_queue)

        result_queue.put(None)

    def _write_array_batch(self,
                           xarr_chip_batch: list[xr.DataArray],
                           anno_chip_batch: list[xr.DataArray],
                           square_name_batch: list[str],
                           batch_index: int,
                           batch_size:int,
                           chip_data_path: str,
                           anno_data_path: str,
                           report_queue: mp.Queue,
                           result_queue: mp.Queue) -> None:
        xarr_chip_batch = xr.merge(xarr_chip_batch)
        report_queue.put(
            ("INFO", f"Writing chip batch #{batch_index} of size {batch_size} to {chip_data_path}..."))
        xarr_chip_batch.to_zarr(
            store=chip_data_path, mode="a")
        if anno_chip_batch[0] is not None:
            xarr_anno_batch = xr.merge(anno_chip_batch)
            report_queue.put(
                ("INFO", f"Writing anno batch #{batch_index} of size {batch_size} to {anno_data_path}..."))
            xarr_anno_batch.to_zarr(
                store=anno_data_path, mode="a")

        # reporting batch completion to watcher
        for name in square_name_batch:
            result_queue.put(name)
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
