import argparse
import ee
import numpy as np
import multiprocessing as mp
import os
import time
import xarray as xr
import yaml
import zarr


# from google.api_core import retry
from datetime import datetime
from pathlib import Path
from typing import Literal
from zarr.errors import PathNotFoundError, GroupNotFoundError, ArrayNotFoundError

from logger import get_logger
from sampler import parse_meta_data
from utils import estimate_download_size, lt_image_generator, zarr_reshape


EE_END_POINT = 'https://earthengine-highvolume.googleapis.com'


class Downloader:
    """
    A class for downloading images from Google Earth Engine via polygons passed in by geojson.

    Args:
        start_date (datetime): The start date to filter image data.
        end_date (datetime): The end date to filter image data.
        file_type (Literal["GEO_TIFF", "ZARR", "NPY", "NUMPY_NDARRAY"]): The file type to save the image data as.
        scale (int): The scale to use for the image data.
        out_path (str): The path to save the image data to.
        meta_data_path (str): The path to the meta data file.
        num_workers (int): The number of workers to use for the parallel download process.
        retries (int): The number of retries to use for the download process.
        request_limit (int): The number of requests to make at a time.
        ignore_size_limit (bool): A flag to ignore the size limits for the image data.
        overwrite (bool): A flag to overwrite existing image data.
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
            start_date: datetime,
            end_date: datetime,
            file_type: Literal["GEO_TIFF", "ZARR", "NPY", "NUMPY_NDARRAY"],
            scale: int,
            out_path: str,
            meta_data_path: str,
            num_workers: int,
            retries: int,
            request_limit: int,
            ignore_size_limit: bool,
            overwrite: bool,
            log_path: str,
            log_name: str,
    ) -> None:
        self._start_date = start_date
        self._end_date = end_date
        self._file_type = file_type
        self._scale = scale
        self._out_path = out_path
        self._meta_data_path = meta_data_path
        self._num_workers = num_workers
        self._retries = retries
        self._request_limit = request_limit
        self._ignore_size_limit = ignore_size_limit
        self._overwrite = overwrite
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
            # this assumes all polygons are the same size
            _, _, polygon_coords, _ = parse_meta_data(self._meta_data, 0)
            test_area = ee.Geometry.Polygon(polygon_coords)
            test_image = self._image_gen_callable(
                self._start_date, self._end_date, test_area, self._scale)
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
        manager = mp.Manager()
        image_queue = manager.Queue()
        result_queue = manager.Queue()
        report_queue = manager.Queue()
        io_lock = manager.Lock()
        request_limiter = manager.Semaphore(self._request_limit)

        workers = set()

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
                    result_queue,
                    report_queue,
                    io_lock,
                    request_limiter),
                daemon=True)
            workers.add(image_consumer)

        start_time = time.time()
        [w.start() for w in workers]
        report_queue.put(("INFO",
                         f"Starting download of {self._meta_size} points of interest..."))
        while (result := result_queue.get()) is not None:
            # TODO: perform result checks and monitor gee processes
            report_queue.put(("INFO", result))
        end_time = time.time()
        report_queue.put(("INFO",
                         f"Download completed in {round((end_time - start_time) / (60))} minutes..."))
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
        match self._file_type:
            case "GEO_TIFF":
                file_ext = "tif"
            case "NPY" | "NUMPY_NDARRAY":
                file_ext = "npy"
        for idx in range(self._meta_size):
            try:
                # creating an outpath for each polygon
                _, point_name, polygon_coords, polygon_name\
                    = parse_meta_data(self._meta_data, idx)

                # checking for existing files
                if self._file_type != "ZARR":
                    out_path = os.path.join(out_path,
                                            point_name,
                                            f"{polygon_name}.{file_ext}")

                    if not self._overwrite and Path(out_path).exists():
                        report_queue.put(
                            "INFO", f"File {out_path} already exists. Skipping...")
                        result_queue.put(polygon_name)
                        continue
                else:
                    if not self._overwrite:
                        try:
                            # opening with read only mode to check for existing zarr groups
                            zarr.open(
                                store=os.path.join(
                                    self._out_path, point_name),
                                mode="r")[polygon_name]
                            report_queue.put(("INFO",
                                              f"{polygon_name} already exists at path. Skipping..."))
                            continue

                        except (PathNotFoundError,
                                GroupNotFoundError,
                                ArrayNotFoundError,
                                KeyError,
                                FileNotFoundError) as e:
                            # capturing valid exceptions and passing to next step
                            report_queue.put(
                                ("INFO", f"Valid exception captured for {polygon_name}: {type(e)}..."))
                            pass

                        except Exception as e:
                            # capturing fatal exceptions and skipping to next polygon
                            report_queue.put(
                                ("CRITICAL", f"Failed to read zarr path {self._out_path}, zarr group {point_name}, or zarr variable {polygon_name} skipping: {type(e)} {e}"))
                            result_queue.put(polygon_name)
                            continue

                # creating payload for each polygon to send to GEE
                report_queue.put(
                    ("INFO", f"Creating image payload {polygon_name}..."))
                area_of_interest = ee.Geometry.Polygon(polygon_coords)
                image = self._image_gen_callable(
                    self._start_date, self._end_date, area_of_interest, self._scale)
                payload = {
                    "expression": ee.serializer.encode(image),
                    "fileFormat": self._file_type if self._file_type != "ZARR" else "NUMPY_NDARRAY",
                }

                # sending payload to the image consumer
                image_queue.put(
                    (payload, polygon_name, point_name, self._out_path))
            except Exception as e:
                report_queue.put(
                    ("CRITICAL", f"Failed to create image for {polygon_name} skipping: {e}"))
                result_queue.put(polygon_name)
        image_queue.put(None)

    def _xr_image_generator(self,
                            image_queue: mp.Queue,
                            result_queue: mp.Queue
                            ) -> None:
        """
        # with the xee backend for the Earth Engine API. The following code is a placeholder.
        # SR is a collection but the output for open_dataset is NaN. If an image is used, the output
        # error with limit (10MB) exceeded instead of 48MB. The issue is likely with the xee backend.
        # ds = xr.open_dataset(
        #     sr_collection,
        #     scale=30,
        #     geometry=p_sml,
        #     engine=xee.EarthEngineBackendEntrypoint,
        # )
        """
        pass

    def _image_consumer(self,
                        image_queue: mp.Queue,
                        result_queue: mp.Queue,
                        report_queue: mp.Queue,
                        io_lock: mp.Lock,
                        request_limiter: mp.Semaphore) -> None:
        ee.Initialize(opt_url=EE_END_POINT)
        while (image_task := image_queue.get()) is not None:
            payload, polygon_name, point_name, out_path = image_task
            attempts = 0

            # attempt to download the image
            while attempts < self._retries:
                attempts += 1
                try:
                    report_queue.put(("INFO",
                                     f"Requesting Image pixels for {polygon_name}..."))
                    with request_limiter:
                        # TODO: implement retry.Retry decorator
                        payload["expression"] = ee.deserializer.decode(
                            payload["expression"])
                        arr = ee.data.computePixels(payload)
                        break
                except Exception as e:
                    time.sleep(3)
                    report_queue.put(
                        ("WARNING", f"Failed to download {polygon_name} attempt {attempts}/{self._retries}: {type(e)} {e}"))
                    if attempts == self._retries:
                        report_queue.put(
                            ("ERROR", f"Max retries reached for {polygon_name} skipping..."))
                        result_queue.put(polygon_name)
                    else:
                        report_queue.put(
                            ("INFO", f"Retrying download for {polygon_name}..."))

            # write the image to disk
            if arr is not None:
                report_queue.put(
                    ("INFO", f"Attempting to save {polygon_name} pixels to {out_path}..."))
                try:
                    # TODO: perform reshaping along years for non zarr file types
                    match self._file_type:
                        case "NPY" | "GEO_TIFF":
                            out_file = Path(out_path)
                            out_file.write_bytes(arr)
                        case "NUMPY_NDARRAY":
                            np.save(out_path, arr)
                        case "ZARR":
                            report_queue.put((
                                "INFO", f"Reshaping image {arr.shape} to zarr..."))
                            xarr = zarr_reshape(arr,
                                                polygon_name,
                                                point_name,
                                                self._start_date.year,
                                                self._end_date.year)

                            report_queue.put((
                                "INFO", f"Writing image of shape {xarr.shape}..."))
                            with io_lock:
                                xarr.to_zarr(
                                    store=out_path,
                                    group=point_name,
                                    mode="a")
                except Exception as e:
                    report_queue.put(
                        ("ERROR", f"Failed to write to {out_path}: {type(e)} {e}"))
                    if self._file_type == "NPY":
                        try:
                            out_file.unlink(missing_ok=True)
                        except Exception as e:
                            report_queue.put(
                                ("ERROR", f"Failed to clean file {out_path}: {type(e)} {e}"))
            result_queue.put(polygon_name)

        result_queue.put(None)


def parse_args():
    parser = argparse.ArgumentParser(description='Sampler Arguments')
    parser.add_argument('--config_path', type=str)
    return vars(parser.parse_args())


def main(**kwargs):
    # TODO: add additional kwargs checks
    if (config_path := kwargs["config_path"]) is not None:
        with open(config_path, "r") as f:
            configs = yaml.safe_load(f)
    else:
        from settings import DOWNLOADER as configs

    downloader = Downloader(**configs)
    downloader.start()


if __name__ == "__main__":
    main(**parse_args())
