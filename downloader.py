import ee
import os
import argparse
import asyncio
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from datetime import datetime


class EarthEngineSRCollection:
    def __init__(
            self,
            area_of_interest: ee.Geometry,
            start_date: datetime,
            end_date: datetime
    ) -> None:
        self.area_of_interest = area_of_interest
        self.start_date = start_date
        self.end_date = end_date
        self.collection = self._build_sr_collection()

    def save_to_file(self, file_type, out_path: str) -> None:
        # TODO: do some image size checking and warn if it's too large
        pass

    def _build_sr_collection(self) -> ee.ImageCollection:
        dummy_collection = ee.ImageCollection(
            [ee.Image([0, 0, 0, 0, 0, 0]).mask(ee.Image(0))])
        return ee.ImageCollection(
            [self._build_medoid_mosaic(year, dummy_collection) for year in range(self.start_date.year, self.end_date.year + 1)])

    # shamelessly stolen from lt-gee-py with minor tweaks
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
            .filterBounds(self.area_of_interest)\
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


class DownloaderWatcher:
    def __init__(
            self,
            out_path: str,
            file_type: str,
            meta_data_path: str,
            num_workers: int,
            start_date: datetime,
            end_date: datetime,
            retries: int,
            chunk: bool,
            verbose: bool
    ) -> None:
        self.out_path = out_path
        self.file_type = file_type
        self.num_workers = num_workers
        self.start_date = start_date
        self.end_date = end_date
        self.retries = retries
        self.chunk = chunk
        self.verbose = verbose
        self.meta_data = None
        self._load_meta_data(meta_data_path)

    def start(self) -> None:
        mp.set_start_method('fork')
        with mp.Pool(self.num_workers) as pool:
            if self.verbose:
                print(
                    f"Starting download of {self.meta_data.size} points of interest...")
            result_queue = mp.Manager().Queue()
            watcher_process = mp.Process(target=self._watcher, args=(
                result_queue, self.meta_data.size))
            watcher_process.start()
            results = pool.imap_unordered(
                self._download, range(self.meta_data.size))
            for result in results:
                result_queue.put(result)
            watcher_process.join()

    def _watcher(self, result_queue: mp.Queue, total_tasks: int) -> None:
        completed_tasks = 0
        if self.verbose:
            self.progress = tqdm(
                total=total_tasks, desc="POI Download", leave=False)
        while completed_tasks < total_tasks:
            result = result_queue.get()
            self.progress.update()
            completed_tasks += 1
            # TODO: gracefully handle failed downloads via result rather than skip POI
        if self.verbose:
            self.progress.close()

    def _download(self, index: int) -> None:
        point_of_interest = self.meta_data[index]["point_of_interest"].toList().join(
            "_")
        if self.verbose:
            self.progress.write(f"Downloading {point_of_interest}...")
        if self.chunk:
            asyncio.run(self._async_download_handler(
                point_of_interest, self.meta_data[index]))
        else:
            area_of_interest = self.meta_data[index]["bounding_box"].toList()
            out_path = os.join(self.out_path, point_of_interest.toList().join(
                "_") + "." + self.file_type)
            return self._downloader(area_of_interest, out_path)

    async def _async_download_handler(self, point_of_interest: str, area_of_interest: np.array) -> bool:
        total_tasks = area_of_interest["area_centroids"].size
        if self.verbose:
            area_prog = tqdm(
                total=total_tasks, desc=f"{point_of_interest}", leave=False)

        tasks = set()
        async with asyncio.TaskGroup() as tg:
            for i in range(total_tasks):
                out_path = area_of_interest["area_centroids"][i] + \
                    "." + self.file_type
                area_of_interest = area_of_interest["area_vertices"][i]\
                    .toList()
                task = tg.create_task(self._async_downloader(
                    area_of_interest["area_vertices"][i].toList(),
                    out_path),
                    name=area_of_interest,
                    callback=lambda _: area_prog.update() if self.verbose else None)
                tasks.add(task)

        # TODO: Record failed downloads and progress
        return all([task.result() for task in tasks])

    async def _async_downloader(self, *args: dict) -> bool:
        return self._downloader(*args)

    def _downloader(self, area_of_interest: list[int], out_path: str) -> bool:
        attempts = 0
        while attempts < self.retries:
            attempts += 1
            try:
                geometry = ee.Geometry.Polygon(area_of_interest)
                collection = EarthEngineSRCollection(
                    geometry, self.start_date, self.end_date)
                collection.save_to_file(self.file_type, out_path)
                break
            except Exception as e:
                if self.verbose:
                    self.progress.write(
                        f"Error downloading. Attempt: {attempts} AOI: {area_of_interest} : {e}")
                if attempts == self.retries:
                    if self.verbose:
                        self.progress.write(f"Max retries reached. Exiting.")
                    return False
                else:
                    if self.verbose:
                        self.progress.write(f"Retrying...")
        return True

    def _load_meta_data(self, meta_data_path: str) -> None:
        self.meta_data = np.load(meta_data_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Downloader Arguments')
    parser.add_argument('--out_path', type=str,
                        default=os.getcwd(), help='Output path')
    parser.add_argument('--file_type', type=str,
                        default='zarr', help='File type to save to')
    parser.add_argument('--meta_data_path', type=str,
                        default=os.join("meta_data.npy"), help='Path to meta data')
    parser.add_argument('--num_workers', type=int,
                        default=mp.cpu_count(), help='Number of workers')
    parser.add_argument('--start_date', type=str,
                        default="2017-09-01", help='Start date')
    parser.add_argument('--end_date', type=str,
                        default="1985-06-01", help='End date')
    parser.add_argument('--retries', type=int, default=3,
                        help='Number of retries for failed downloads')
    parser.add_argument('--chunk', action='store_true',
                        help='Download area chunks instead of entire bounding box.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    return parser.parse_args()


def main(**kwargs):
    ee.Initialize()
    # TODO: add additional kwargs checks
    kwargs["start_date"] = datetime.strptime(kwargs["start_date"], "%Y-%m-%d")
    kwargs["end_date"] = datetime.strptime(kwargs["end_date"], "%Y-%m-%d")
    watcher = DownloaderWatcher(**kwargs)
    watcher.start()


if __name__ == "__main__":
    main(parse_args())
