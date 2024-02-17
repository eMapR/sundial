import ee
import os
import zarr
import time
import yaml
import argparse
import numpy as np
import xarray as xr
import multiprocessing as mp

# from google.api_core import retry
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Literal

from logger import get_logger
from sampler import parse_meta_data
from utils import estimate_download_size, lt_image_generator, zarr_reshape


class Downloader:
    """
    A class for downloading images from Google Earth Engine via polygons passed in by geojson.

    Args:
        start_date (datetime): The start date to filter image data.
        end_date (datetime): The end date to filter image data.
        out_path (str, optional): The output path for downloaded images. Defaults to "data" folder in the user's home directory.
        log_path (str, optional): The path for log files. Defaults to "logs" folder in the user's home directory.
        file_type (str, optional): The file type of the downloaded images. Defaults to "geo_tiff".
        meta_data_path (str, optional): The path to the meta data file. Defaults to "meta_data.json".
        num_workers (int, optional): The number of worker processes for downloading. Defaults to 64.
        image_generator (callable, optional): The image generator function. Defaults to lt_image_generator.
        retries (int, optional): The number of retries for failed downloads. Defaults to 5.
        request_limit (int, optional): The maximum number of concurrent requests. Defaults to 40.
        size_limit (int, optional): The maximum size limit for downloaded images in MB. Defaults to 48MB.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

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
            out_path: str,
            meta_data_path: str,
            num_workers: int,
            retries: int,
            request_limit: int,
            overwrite: bool,
            log_path: str,
            log_name: str,
            verbose: bool = False,
    ) -> None:
        self._start_date = start_date
        self._end_date = end_date
        self._file_type = file_type
        self._out_path = out_path
        self._meta_data_path = meta_data_path
        self._num_workers = num_workers
        self._retries = retries
        self._request_limit = request_limit
        self._overwrite = overwrite
        self._log_path = log_path
        self._log_name = log_name
        self._verbose = verbose

        # TODO: Parameterize the image generator callable
        self._image_gen_callable = lt_image_generator
        self.meta_data = xr.open_zarr(self._meta_data_path)
        self.meta_size = self.meta_data["index"].sizes
        # TODO: Perform attribute checks

    def start(self) -> None:
        """
        Starts the parallel download process and performs the necessary checks.
        """
        mp.set_start_method('fork')
        if self._verbose:
            print(
                f"Starting download of {self.meta_size} points of interest...")

        # this assumes all polygons are the same size
        _, _, polygon_coords, _ = parse_meta_data(self.meta_data, 0)
        test_area = ee.Geometry.Polygon(polygon_coords)
        test_image = self._image_gen_callable(
            self._start_date, self._end_date, test_area)
        test_size, test_pixels, test_bands = estimate_download_size(
            test_image, test_area, 30, self.end_date.year - self.start_date.year + 1)
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
        report_queue = manager.Queue()
        result_queue = manager.Queue()
        request_limiter = manager.Semaphore(self._request_limit)

        if self._verbose:
            progress = tqdm(total=self.meta_size, desc="DATA")

        workers = set()
        reporter = mp.Process(
            target=self._reporter, args=(
                report_queue,))
        workers.add(reporter)
        image_generator = mp.Process(
            target=self._ex_image_generator, args=(
                image_queue,
                report_queue,
                result_queue))
        workers.add(image_generator)

        for _ in range(self._num_workers):
            image_consumer = mp.Process(
                target=self._image_consumer, args=(
                    image_queue,
                    report_queue,
                    result_queue,
                    request_limiter))
            workers.add(image_consumer)

        [w.start() for w in workers]
        while (result := result_queue.get()) is not None:
            # TODO: perform result checks and monitor gee processes
            if self._verbose:
                if isinstance(result, str):
                    progress.write(
                        f"Download succeeded: {result}")
                else:
                    progress.write(
                        f"Download Failed: {result}")
                progress.update()
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
                         report_queue: mp.Queue,
                         result_queue: mp.Queue
                         ) -> None:
        for idx in range(self.meta_size):
            try:
                # creating an outpath for each polygon
                _, point_name, polygon_coords, polygon_name = parse_meta_data(
                    self.meta_data, idx)
                out_path = self._out_path

                # checking for existing files
                if self._file_type != "zarr":
                    out_path = os.path.join(self._out_path,
                                            point_name,
                                            f"{polygon_name}.npy")

                    if not self._overwrite and Path(out_path).exists():
                        report_queue.put("INFO",
                                         f"File {out_path} already exists. Skipping...")
                        result_queue.put(f"{polygon_name} (skipped)")
                        continue
                else:
                    if not self._overwrite:
                        try:
                            xr.open_zarr(
                                store=out_path,
                                group=point_name,
                                mode="r")[polygon_name]
                            report_queue.put("INFO",
                                             f"Zarr path {out_path}, zarr group {point_name}, or zarr variable {polygon_name} already exists. Skipping...")
                            continue
                        except zarr.errors.PathNotFoundError as e:
                            pass
                        except KeyError as e:
                            pass
                        except Exception as e:
                            report_queue.put("CRITICAL",
                                             f"Fatal error in reading zarr path {out_path}, zarr group {point_name}, or zarr variable {polygon_name}: {e}")
                            report_queue.put("INFO",
                                             f"Skipping {polygon_name}...")
                            result_queue.put(f"{polygon_name} (failed)")
                            continue

                # creating payload for each polygon
                report_queue.put("INFO",
                                 f"Creating image payload {polygon_name}...")
                area_of_interest = ee.Geometry.Polygon(polygon_coords)
                image = self._image_gen_callable(
                    self._start_date, self._end_date, area_of_interest)
                payload = {
                    "expression": ee.serializer.encode(image),
                    "fileFormat": self._file_type if self._file_type != "ZARR" else "NPY",
                }

                # sending payload to the image consumer
                image_queue.put((payload, polygon_name, point_name, out_path))
            except Exception as e:
                report_queue.put("CRITICAL",
                                 f"Failed to create image for {polygon_name}: {e}")
        image_queue.put(None)

    def _xr_image_generator(self,
                            image_queue: mp.Queue,
                            report_queue: mp.Queue,
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
        # TODO: Implement xarray image generator and consumer for now, there seems to be a gee issue

    def _image_consumer(self,
                        image_queue: mp.Queue,
                        report_queue: mp.Queue,
                        result_queue: mp.Queue,
                        request_limiter: mp.Semaphore) -> None:
        while (image_task := image_queue.get()) is not None:
            payload, polygon_name, point_name, out_path = image_task
            attempts = 0

            # attempt to download the image
            while attempts < self._retries:
                attempts += 1
                try:
                    report_queue.put("INFO",
                                     f"Requesting Image pixels for {polygon_name}...")
                    with request_limiter:
                        # TODO: implement retry.Retry decorator
                        payload["expression"] = ee.deserializer.decode(
                            payload["expression"])
                        arr = ee.data.computePixels(payload)
                        break
                except Exception as e:
                    time.sleep(3)
                    report_queue.put(
                        "WARNING", f"Failed to download {polygon_name}: {e}")
                    if attempts == self._retries:
                        report_queue.put(
                            "ERROR",
                            f"Max retries reached for {polygon_name}: Skipping...")
                        result_queue.put((polygon_name, e))
                    else:
                        report_queue.put(
                            "INFO",
                            f"Retrying download for {polygon_name}...")

            # write the image to disk
            if arr is not None:
                try:
                    report_queue.put(
                        "INFO",
                        f"Writing Image pixels to {out_path}...")
                    match self._file_type:
                        case "NPY":
                            # TODO: perform npy reshaping along years
                            out_file = Path(out_path)
                            out_file.write_bytes(arr)
                            result_queue.put(polygon_name)
                        case "NUMPY_NDARRAY":
                            # TODO: perform npy reshaping along years
                            np.save(out_path, arr)
                        case "ZARR":
                            xarr = zarr_reshape(arr,
                                                polygon_name,
                                                self._start_date.year,
                                                self._end_date.year)
                            xarr.to_zarr(
                                store=out_path,
                                group=point_name,
                                mode="a",)

                except Exception as e:
                    report_queue.put(
                        "ERROR",
                        f"Failed to write to {out_path}: {e}")
                    if self._file_type == "NPY":
                        try:
                            out_file.unlink(missing_ok=True)
                        except Exception as e:
                            report_queue.put(
                                "ERROR",
                                f"Failed to clean file {out_path}: {e}")

        result_queue.put(None)


def parse_args():
    parser = argparse.ArgumentParser(description='Sampler Arguments')
    parser.add_argument('--config_path', type=str)
    return parser.parse_args()


def main(**kwargs):
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    # TODO: add additional kwargs checks
    if (configs_path := kwargs["configs_path"]) is not None:
        with open(configs_path, "r") as f:
            configs = yaml.safe_load(f)
    else:
        from settings import DOWNLOADER as configs

    downloader = Downloader(**configs)
    downloader.start()


if __name__ == "__main__":
    main(**parse_args())
