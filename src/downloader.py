import ee
import os
import argparse
import json
import numpy as np
import multiprocessing as mp

from queue import Queue
from tqdm import tqdm
from pathlib import Path
from threading import Thread
from datetime import datetime
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from datetime import datetime

COLLECTION_URL = f"https://earthengine-highvolume.googleapis.com/v1/{os.getenv('GEE_PROJECT')}/imageCollection:computeImages"
IMAGE_PIXEL_URL = f"https://earthengine-highvolume.googleapis.com/v1/{os.getenv('GEE_PROJECT')}:getPixels"


class EEESRCollection:
    def __init__(
            self,
            centroid: str,
            area_of_interest: list,
            start_date: datetime,
            end_date: datetime
    ) -> None:
        self.centroid = centroid
        self.area_of_interest = area_of_interest
        self.start_date = start_date
        self.end_date = end_date
        self.expression = self._build_sr_collection().serialize()

    def _flatten_collection(self, collection: ee.ImageCollection) -> ee.Image:
        pass

    def _build_sr_collection(self) -> ee.ImageCollection:
        dummy_collection = ee.ImageCollection(
            [ee.Image([0, 0, 0, 0, 0, 0]).mask(ee.Image(0))])
        return ee.ImageCollection(
            [self._build_medoid_mosaic(year, dummy_collection) for year in range(self.start_date.year, self.end_date.year + 1)])

    # shamelessly stolen from lt-gee-py with minor tweaks to avoid any client side operations
    def _build_medoid_mosaic(self, year: int, dummy_collection: ee.ImageCollection) -> ee.ImageCollection:
        collection = self._get_combined_sr_collection(year)
        image_count = collection.size()
        final_collection = ee.ImageCollection(ee.Algorithms.If(
            image_count.gt(0), collection, dummy_collection))
        median = final_collection.median()
        med_diff_collection = final_collection.map(
            lambda image: self.calculate_median_diff(image, median))
        return med_diff_collection\
            .reduce(ee.Reducer.min(7))\
            .select([1, 2, 3, 4, 5, 6], ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'])\
            .set('system:time_start', self.start_date.millis())\
            .toUint16()

    def _get_combined_sr_collection(self, year: int) -> ee.ImageCollection:
        lt5 = self._get_sr_collection(year, 'LT05')
        le7 = self._get_sr_collection(year, 'LE07')
        lc8 = self._get_sr_collection(year, 'LC08')
        lc9 = self._get_sr_collection(year, 'LC09')
        return lt5.merge(le7).merge(lc8).merge(lc9)

    def _get_sr_collection(self, year, sensor: str) -> ee.ImageCollection:
        if self.start_date.month > self.end_date.month:
            start_date = ee.Date.fromYMD(
                year - 1, self.start_date.month, self.start_date.day)
            end_date = ee.Date.fromYMD(
                year, self.end_date.month, self.end_date.day)
        else:
            start_date = ee.Date.fromYMD(
                year, self.start_date.month, self.start_date.day)
            end_date = ee.Date.fromYMD(
                year, self.end_date.month, self.end_date.day)
        return ee.ImageCollection('LANDSAT/' + sensor + '/C02/T1_L2')\
            .filterBounds(ee.Geometry.Polygon(self.area_of_interest))\
            .filterDate(start_date, end_date)\
            .map(lambda image: self._preprocess_image(image, sensor))\
            .set("system:time_start", self.start_date.millis())

    def _preprocess_image(self, image: ee.Image, sensor: ee.Image) -> ee.Image:
        if sensor == 'LC08' or sensor == 'LC09':
            dat = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'], [
                'B1', 'B2', 'B3', 'B4', 'B5', 'B7'])
        else:
            dat = image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'], [
                'B1', 'B2', 'B3', 'B4', 'B5', 'B7'])
        dat = dat.multiply(
            0.0000275).add(-0.2).multiply(10000).toUint16().unmask()
        return self._apply_masks(image.select('QA_PIXEL'), dat)

    def _apply_masks(self, qa: ee.image, image: ee.image) -> ee.image:
        self.collection.toList()
        image = self._mask_clouds(qa, image)
        return image

    @staticmethod
    def _mask_clouds(qa: ee.image, image: ee.image) -> ee.image:
        mask = qa.bitwiseAnd(1 << 3).eq(0)
        return image.updateMask(mask)

    @staticmethod
    def calculate_median_diff(image: ee.Image, median: ee.Image) -> ee.Image:
        diff = image.subtract(median).pow(ee.Image.constant(2))
        return diff.reduce('sum').addBands(image)


class Downloader:
    def __init__(
            self,
            out_path: str,
            file_type: str,
            meta_data_path: str,
            num_workers: int,
            start_date: datetime,
            end_date: datetime,
            retries: int,
            bounding_polygon: bool,
            request_limit: int,
            verbose: bool
    ) -> None:
        self.out_path = out_path
        self.file_type = file_type
        self.num_workers = num_workers
        self.start_date = start_date
        self.end_date = end_date
        self.retries = retries
        self.bounding_polygon = bounding_polygon
        self.request_limit = request_limit
        self.verbose = verbose

        self._init_meta_data_gcloud(meta_data_path)

    def start(self) -> None:
        mp.set_start_method('fork')
        if self.verbose:
            print(
                f"Starting download of {META_DATA.size} points of interest...")
        self._watcher()

    def _watcher(self) -> None:
        manager = mp.Manager()
        collection_queue = manager.Queue()
        report_queue = manager.Queue()
        result_queue = manager.Queue()
        request_limiter = manager.Semaphore(self.request_limit)

        if self.verbose:
            if self.bounding_polygon:
                size = META_DATA.size
            else:
                size = META_DATA[0]["polygons"].size
            progress = tqdm(total=size, desc="DATA")

        workers = set()
        # Spawning additional processes to start download before coordinates are fully parsed
        collection_generator = mp.Process(
            target=self._collection_generator, args=(
                collection_queue,
                report_queue))
        workers.add(collection_generator)

        for _ in range(self.num_workers):
            collection_consumer = mp.Process(
                target=self._collection_consumer, args=(
                    collection_queue,
                    report_queue,
                    result_queue,
                    request_limiter))
            workers.add(collection_consumer)

        [w.start() for w in workers]
        while (result := result_queue.get()) is not None:
            if self.verbose:
                while (report := report_queue.get()) is not None:
                    progress.write(report)
                if isinstance(result, str):
                    progress.write(
                        f"Download succeeded: {result}")
                else:
                    progress.write(
                        f"Download Failed: {result}")
                progress.update()
            # TODO: gracefully handle failed downloads via result rather than skip POI
            # TODO: record completed downloads in case of failure for resuming
        [w.join() for w in workers]

    def _collection_generator(self,
                              collection_queue: mp.Queue,
                              report_queue: mp.Queue) -> None:
        for i in range(META_DATA.size):
            centroid = META_DATA[i]["centroid"]\
                .toList()\
                .join("_")
            if self.bounding_polygon:
                area_of_interest = META_DATA[i]["bounding_polygon"]\
                    .toList()
                out_path = os.join(self.out_path, centroid)
                if self.verbose:
                    report_queue.put(
                        f"Creating Collection {centroid}...")
                collection = EEESRCollection(
                    area_of_interest,
                    self.start_date,
                    self.end_date)

                collection_queue.put((collection, centroid, out_path))
            else:
                for j in range(META_DATA[i]["polygons"].size):
                    centroid_idx = centroid + "_" + str(j)
                    area_of_interest = META_DATA[i]["polygons"][j]\
                        .toList()
                    out_path = os.join(self.out_path,
                                       centroid,
                                       centroid_idx)

                    if self.verbose:
                        report_queue.put(
                            f"Creating Collection {centroid_idx}...")
                    collection = EEESRCollection(
                        area_of_interest,
                        self.start_date,
                        self.end_date)

                    collection_queue.put((collection, centroid_idx, out_path))

        collection_queue.put(None)

    def _collection_consumer(self,
                             collection_queue: mp.Queue,
                             report_queue: mp.Queue,
                             result_queue: mp.Queue,
                             request_limiter: mp.Semaphore) -> None:
        while (collection_task := collection_queue.get()) is not None:
            attempts = 0
            collection, name, out_path = collection_task
            while attempts < self.retries:
                attempts += 1
                try:
                    if self.verbose:
                        report_queue.put(
                            f"Requesting Image references for Collection {name}...")

                    with request_limiter:
                        # TODO: parse response errors
                        payload = {
                            "expression": collection.expression
                        }
                        response = SESSION.send_request(
                            COLLECTION_URL, payload)

                    if self.verbose:
                        report_queue.put(
                            f"Downloading Image files for Collection {name}...")
                    thread_queue = Queue()
                    threads = [Thread(
                        target=self._downloader,
                        args=(image, out_path, thread_queue, request_limiter))
                        for image in json.loads(response.body)["images"]]
                    [t.join() for t in threads]

                    results = [thread_queue.get() for _ in threads]
                    if all([not isinstance(r, Exception) for r in results]):
                        result_queue.put(name)
                    else:
                        result_queue.put(results)
                    break
                except Exception as e:
                    if attempts == self.retries:
                        # TODO: parse exceptions
                        result_queue.put(e)
                        break

        result_queue.put(None)

    def _downloader(self,
                    image: dict,
                    out_path: str,
                    result_queue: Queue,
                    request_limiter: mp.Semaphore) -> None:
        attempts = 0
        while attempts < self.retries:
            attempts += 1
            try:
                # Someday, google will increase the concurrent request limit
                # TODO: Perform image size check before download
                with request_limiter:
                    # TODO: parse response error
                    payload = {
                        "name": image["name"],
                        "fileFormat": self.file_type
                    }
                    response = SESSION.send_request(
                        IMAGE_PIXEL_URL, payload)

                year = self.get_year_from_rfc3339(image["startTime"])
                out_path = os.join(
                    out_path,
                    year + image["id"] + "." + self.file_type
                )
                outfile = Path(outfile)
                outfile.write_bytes(response.body)
                result_queue.put(None)
                break
            except Exception as e:
                # TODO: parse exceptions
                if attempts == self.retries:
                    result_queue.put(e)
                    break

    def get_year_from_rfc3339(date_str: str) -> str:
        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        year = str(date.year)
        return year

    def _init_meta_data_gcloud(self, meta_data_path: str) -> None:
        # must be run on a posix compliant system to avoid copy. On windows, forked process do not inheret globals in the same way
        global META_DATA
        global CREDENTIALS
        global SESSION
        META_DATA = np.load(meta_data_path)
        CREDENTIALS = service_account.Credentials.from_service_account_file(
            os.getenv("GEE_SERVICE_KEY_PATH"))
        SESSION = AuthorizedSession(CREDENTIALS)


def parse_args():
    parser = argparse.ArgumentParser(description='Downloader Arguments')
    parser.add_argument('--out_path', type=str,
                        default=os.getcwd(), help='Output path')
    parser.add_argument('--file_type', type=str,
                        default='zarr', help='File type to save to')
    parser.add_argument('--meta_data_path', type=str,
                        default="meta_data.npy", help='Path to meta data')
    parser.add_argument('--num_workers', type=int,
                        default=mp.cpu_count(), help='Number of workers')
    parser.add_argument('--start_date', type=str,
                        default="2017-09-01", help='Start date')
    parser.add_argument('--end_date', type=str,
                        default="1985-06-01", help='End date')
    parser.add_argument('--retries', type=int, default=5,
                        help='Number of retries for failed downloads')
    parser.add_argument('--bounding_polygon', action='store_true',
                        help='Download area bounding_polygon instead of subareas.')
    parser.add_argument('--request_limit', type=int,
                        default=40, help='Number of requests')
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
