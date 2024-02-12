import ee
import os
import argparse
import time
import multiprocessing as mp

from logger import get_logger
from ltgee import LandTrendr
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from datetime import datetime
from sampler import load_geojson

IMAGE_PIXEL_URL = f"https://earthengine-highvolume.googleapis.com/v1/{
    os.getenv('GEE_PROJECT')}:computePixels"


def estimate_download_size(image: ee.Image, geometry: ee.Geometry, scale: int) -> float:
    """
    Estimates the download size of an image based on its pixel count and band dtype.
    The function currently only supports array images.

    Args:
        image (ee.Image): The image to estimate the download size for.
        geometry (ee.Geometry): The geometry to reduce the image over.
        scale (int): The scale to use for the reduction.

    Returns:
        int: The estimated download size in megabytes.
    """

    # TODO: add support for other image shapes
    pixel_count = image\
        .arrayLength(0)\
        .reduceRegion(ee.Reducer.sum(), geometry, scale=scale, maxPixels=1e13)\
        .values()\
        .getInfo()\
        / 1e6
    match image.bandTypes().values().getInfo()[0]["precision"]:
        case "int16":
            return round(pixel_count * 2, 2)
        case "int32" | "int":
            return round(pixel_count * 4, 2)
        case "int64" | "double":
            return round(pixel_count * 8, 2)


def lt_image_generator(start_date: datetime, end_date: datetime, area_of_interest: ee.Geometry, mask_labels: list[str] = ["cloud"]) -> ee.Image:
    lt = LandTrendr(
        start_date=start_date,
        end_date=end_date,
        area_of_interest=area_of_interest,
        mask_labels=mask_labels,
        run=False
    )
    return lt.build_sr_collection().toArrayBands().clip(area_of_interest)


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
        meta_index_key (str, optional): The key for the index in the meta data. Defaults to "id".
        meta_shape_key (str, optional): The key for the polygon to download in the meta data. Defaults to "squares".
        meta_count_key (str, optional): The key for the polygon count in the meta data. Defaults to "square_count".
        num_workers (int, optional): The number of worker processes for downloading. Defaults to 64.
        image_generator (callable, optional): The image generator function. Defaults to lt_image_generator.
        retries (int, optional): The number of retries for failed downloads. Defaults to 5.
        request_limit (int, optional): The maximum number of concurrent requests. Defaults to 40.
        size_limit (int, optional): The maximum size limit for downloaded images in MB. Defaults to 48MB.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Methods:
        start(): Starts the download process.
    """

    def __init__(
            self,
            start_date: datetime,
            end_date: datetime,
            out_path: str = os.join(Path.home(), "data"),
            log_path: str = os.join(Path.home(), "logs"),
            file_type: str = "geo_tiff",
            meta_data_path: str = "meta_data.json",
            meta_index_key: str = "id",
            meta_shape_key: str = "squares",
            meta_count_key: str = "square_count",
            num_workers: int = 64,
            image_generator: callable = lt_image_generator,
            retries: int = 5,
            request_limit: int = 40,
            size_limit: int = 48,
            overwrite: bool = False,
            verbose: bool = False
    ) -> None:
        self._start_date = start_date
        self._end_date = end_date
        self._out_path = out_path
        self._log_path = log_path
        self._file_type = file_type
        self._meta_index_key = meta_index_key
        self._meta_shape_key = meta_shape_key
        self._meta_count_key = meta_count_key
        self._num_workers = num_workers
        self._image_generator = image_generator
        self._retries = retries
        self._request_limit = request_limit
        self._size_limit = size_limit
        self._overwrite = overwrite
        self._verbose = verbose

        self._init_meta_data_gcloud(meta_data_path)

    def start(self) -> None:
        """
        Starts the download process and performs the necessary checks.
        """
        mp.set_start_method('fork')
        if self._verbose:
            print(
                f"Starting download of {META_DATA.size} points of interest...")

        # this assumes all polygons are the same size
        test_area = ee.Geometry.Polygon(
            list(META_DATA[self._meta_shape_key][0].exterior.coords))
        test_image = self._image_generator(
            self._start_date, self._end_date, test_area)
        test_size = estimate_download_size(test_image, test_area, 30)
        if test_size > self._size_limit:
            raise ValueError(
                f"Image size of {test_size}MB exceeds size limit of {self._size_limit}MB. Please reduce the size of the image.")

        self._watcher()

    def _watcher(self) -> None:
        manager = mp.Manager()
        image_queue = manager.Queue()
        report_queue = manager.Queue()
        result_queue = manager.Queue()
        request_limiter = manager.Semaphore(self._request_limit)

        if self._verbose:
            size = META_DATA[self._meta_count_key].sum()
            progress = tqdm(total=size, desc="DATA")

        workers = set()
        reporter = mp.Process(
            target=self._reporter, args=(
                report_queue,))
        workers.add(reporter)

        image_generator = mp.Process(
            target=self._image_generator, args=(
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
        logger = get_logger(self._out_path, "dft.downloader")
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
        for i in range(META_DATA.size):
            try:
                index = META_DATA[self._meta_index_key][i]
                idx = 0
                for square in META_DATA[self._meta_shape_key][i].geoms:
                    centroid = list(square.centroid).join("_")
                    name = str(idx) + "_" + centroid
                    out_path = os.join(self._out_path,
                                       index,
                                       name + "." + self._file_type)

                    idx += 1
                    if not self._overwrite and Path(out_path).exists():
                        report_queue.put("INFO",
                                         f"File {out_path} already exists. Skipping...")
                        result_queue.put(name + " (skipped)")
                        continue

                    report_queue.put("INFO",
                                     f"Creating image {name}...")

                    area_of_interest = ee.Geometry.Polygon(
                        list(square.exterior.coords))
                    image = self._image_generator(
                        self._start_date, self._end_date, area_of_interest)
                    image_queue.put(
                        (image.serialize(), name, out_path))
            except Exception as e:
                report_queue.put("CRITICAL",
                                 f"Failed to create image for {index}: {e}")
        image_queue.put(None)

    def _image_consumer(self,
                        image_queue: mp.Queue,
                        report_queue: mp.Queue,
                        result_queue: mp.Queue,
                        request_limiter: mp.Semaphore) -> None:
        while (image_task := image_queue.get()) is not None:
            image, name, out_path = image_task
            attempts = 0
            while attempts < self._retries:
                attempts += 1
                try:
                    report_queue.put("INFO",
                                     f"Requesting Image pixels for {name}...")
                    with request_limiter:
                        payload = {
                            "expression": image,
                            "fileFormat": self._file_type
                        }
                        response = SESSION.post(
                            IMAGE_PIXEL_URL,
                            payload
                        )
                        # Google's authorized session retries internally based on the status code
                        response.raise_for_status()
                        break
                except Exception as e:
                    report_queue.put(
                        "WARNING", f"Failed to download {name}: {e}")
                    if attempts == self._retries:
                        report_queue.put(
                            "ERROR",
                            f"Max retries reached for {name}: Skipping...")
                        result_queue.put((name, e))
                    else:
                        report_queue.put(
                            "INFO",
                            f"Retrying download for {name}...")
                    response = None

            if response is not None:
                try:
                    report_queue.put(
                        "INFO",
                        f"Writing Image pixels to {out_path}...")
                    out_file = Path(out_path)
                    out_file.write_bytes(response.body)
                    result_queue.put(name)
                except Exception as e:
                    report_queue.put(
                        "ERROR",
                        f"Failed to write to {out_path}: {e}")
                    try:
                        out_file.unlink(missing_ok=True)
                    except Exception as e:
                        report_queue.put(
                            "ERROR",
                            f"Failed to clean file {out_path}: {e}")

        result_queue.put(None)

    def _init_meta_data_gcloud(self, meta_data_path: str) -> None:
        # must be run on a posix compliant system to avoid copy. On windows, forked process do not inheret globals in the same way
        global META_DATA
        global CREDENTIALS
        global SESSION
        META_DATA = load_geojson(meta_data_path)
        CREDENTIALS = service_account.Credentials.from_service_account_file(
            os.getenv("GEE_SERVICE_KEY_PATH"))
        SESSION = AuthorizedSession(CREDENTIALS)


def parse_args():
    parser = argparse.ArgumentParser(description='Downloader Arguments')
    parser.add_argument('--start_date', type=str, required=True,
                        default="1985-06-01", help='Start date')
    parser.add_argument('--end_date', type=str, required=True,
                        default="2017-09-01", help='End date')
    parser.add_argument('--out_path', type=str,
                        help='Output path')
    parser.add_argument('--log_path', type=str,
                        help='Path to save logs')
    parser.add_argument('--file_type', type=str,
                        help='File type to save to')
    parser.add_argument('--meta_data_path', type=str,
                        help='Path to meta data')
    parser.add_argument('--num_workers', type=int,
                        help='Number of workers')
    parser.add_argument('--retries', type=int,
                        help='Number of retries for failed downloads')
    parser.add_argument('--request_limit', type=int,
                        help='Number of requests')
    parser.add_argument('--size_limit', type=int,
                        help='Size limit in MB')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    return parser.parse_args()


def main(**kwargs):
    ee.Initialize()
    # TODO: add additional kwargs checks
    kwargs["start_date"] = datetime.strptime(kwargs["start_date"], "%Y-%m-%d")
    kwargs["end_date"] = datetime.strptime(kwargs["end_date"], "%Y-%m-%d")
    watcher = Downloader(**kwargs)
    watcher.start()


if __name__ == "__main__":
    main(**parse_args())
