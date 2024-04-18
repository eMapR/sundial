import ee
import geopandas as gpd
import multiprocessing as mp
import numpy as np
import operator
import os
import random
import xarray as xr

from datetime import datetime
from typing import Literal
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely import unary_union
from shapely.ops import transform
from sklearn.model_selection import train_test_split
from typing import Optional

from .downloader import Downloader
from .logger import get_logger
from .settings import (
    load_yaml,
    save_yaml,
    update_yaml,
    ALL_SAMPLE_PATH,
    ANNO_DATA_PATH,
    CHIP_DATA_PATH,
    DATETIME_LABEL,
    GEO_PRE_PATH,
    GEO_RAW_PATH,
    LOG_PATH,
    META_DATA_PATH,
    METHOD,
    NO_DATA_VALUE,
    PREDICT_SAMPLE_PATH,
    RANDOM_SEED,
    SAMPLER_CONFIG,
    STAT_DATA_PATH,
    STRATA_LABEL,
    STRATA_LIST_PATH,
    TEST_SAMPLE_PATH,
    TRAIN_SAMPLE_PATH,
    VALIDATE_SAMPLE_PATH,
)
from .utils import (
    clip_xy_xarray,
    function_timer,
    get_class_weights,
    get_xarr_anno_mean_std,
    get_xarr_chip_mean_std,
    lt_image_generator,
    pad_xy_xarray,
    parse_meta_data,
    test_non_zero_sum,
    zarr_reshape,
)

LOGGER = get_logger(LOG_PATH, METHOD)


def gee_get_ads_score_image(
        area_of_interest: ee.Geometry) -> ee.Image:
    return ee.Image(os.getenv("ADS_SCORE_IMAGE_LINK")).clip(area_of_interest)


def gee_get_elevation_image(
        area_of_interest: ee.Geometry) -> ee.Image:
    return ee.Image('USGS/SRTMGL1_003').clip(area_of_interest)


def gee_get_prism_image(
        area_of_interest: ee.Geometry,
        start_date: datetime,
        end_date: datetime) -> ee.Image:
    collection = ee.ImageCollection("OREGONSTATE/PRISM/AN81m")\
        .filterBounds(area_of_interest)\
        .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    return collection.reduce(ee.Reducer.mean()).select(["ppt_mean"], ["ppt"])


def gee_get_percentile_ranges(
        single_band_image: ee.Image,
        area_of_interest: ee.Geometry,
        percentiles: ee.List) -> list[int]:
    return sorted(single_band_image.reduceRegion(
        reducer=ee.Reducer.percentile(percentiles),
        geometry=area_of_interest,
        maxPixels=1e13
    ).values().getInfo())


@function_timer
def gee_stratify_by_percentile(
        single_band_image: ee.Image,
        percentiles: list[int],
        out_band_name: str = None) -> ee.Image:
    result = ee.Image(0)
    for idx in range(len(percentiles) - 1):
        mask = single_band_image.gte(percentiles[idx]).And(
            single_band_image.lt(percentiles[idx+1]))
        result = result.where(mask, ee.Image(idx+1))
    if out_band_name is not None:
        result = result.select(["constant"], [out_band_name])
    return result


@function_timer
def gee_generate_random_points(
        feature: ee.Feature,
        radius: int,
        num_points: int,
) -> ee.FeatureCollection:
    geometry = feature.geometry().buffer(distance=radius)
    return ee.FeatureCollection.randomPoints(
        region=geometry,
        points=num_points,
        seed=RANDOM_SEED,
    )


@function_timer
def gee_stratified_sampling(
        num_points: int,
        num_strata: int,
        scale: int,
        start_date: datetime,
        end_date: datetime,
        sources: Literal["prism", "elevation", "ads_score"],
        area_of_interest: ee.Geometry,
        projection: str) -> ee.FeatureCollection:
    # creating percentiles for stratification
    num_images = len(sources)
    percentiles = ee.List.sequence(0, 100, count=num_strata+1)

    # Getting data images for stratification
    raw_images = []
    for source in sources:
        match source:
            case "prism":
                raw_images.append(
                    gee_get_prism_image(area_of_interest, start_date, end_date))
            case "elevation":
                raw_images.append(
                    gee_get_elevation_image(area_of_interest))
            case "ads_score":
                raw_images.append(
                    gee_get_ads_score_image(area_of_interest))
            case _:
                raise ValueError(f"Invalid source: {source}")

    # stratify by percentile
    stratified_images = []
    for image in raw_images:
        percentile_ranges = gee_get_percentile_ranges(
            image, area_of_interest, percentiles)
        stratified_images.append(
            gee_stratify_by_percentile(image, percentile_ranges))

    # concatenate stratified images
    if num_images == 1:
        population = stratified_images[0]
    else:
        combined = ee.Image.cat(stratified_images)
        num_bands = num_images
        concatenate_expression = " + ".join(
            [f"(b({i})*(100**{i}))" for i in range(num_bands)])
        population = combined.expression(concatenate_expression).toInt()

    # get stratified random sample experession for compute features laters
    return population.stratifiedSample(
        num_points,
        region=area_of_interest,
        scale=scale,
        projection=projection,
        geometries=True)


@function_timer
def gee_download_features(
        features: ee.FeatureCollection) -> gpd.GeoDataFrame:
    try:
        return ee.data.computeFeatures({
            "expression": features,
            "fileFormat": "GEOPANDAS_GEODATAFRAME"})
    except Exception as e:
        LOGGER.critical(
            f"Failed to convert feature collection to GeoDataFrame: {e}")
        raise e


@function_timer
def preprocess_data(
        geo_dataframe: gpd.GeoDataFrame,
        preprocess_actions: list[dict[
            "column": str,
            "action": Literal[">", "<", "==", "!=", "replace"],
            "targets": int, float, str, list]],
        projection: Optional[str] = None,
        strata_list_path: Optional[str] = None,
        strata_columns: Optional[list[str]] = None,
        datetime_column: Optional[str] = None) -> gpd.GeoDataFrame:
    epsg = f"EPSG:{geo_dataframe.crs.to_epsg()}"
    if projection is not None and epsg != projection:
        LOGGER.info(f"Reprojecting geo dataframe to {projection}...")
        geo_dataframe = geo_dataframe.to_crs(projection)
        epsg = projection

    for preprocess in preprocess_actions:
        column = preprocess.get("column")
        action = preprocess.get("action")
        target = preprocess.get("targets")
        match action:
            case ">":
                geo_dataframe = geo_dataframe.loc[geo_dataframe[column] > target]
            case "<":
                geo_dataframe = geo_dataframe.loc[geo_dataframe[column] < target]
            case "==":
                if isinstance(target, str) or isinstance(target, int):
                    geo_dataframe = geo_dataframe.loc[geo_dataframe[column] == target]
                elif isinstance(target, list):
                    geo_dataframe = geo_dataframe.loc[geo_dataframe[column].isin(
                        target)]
                else:
                    raise ValueError(
                        f"Invalid filter target type: {type(target)}")
            case "!=":
                if isinstance(target, str) or isinstance(target, int):
                    geo_dataframe = geo_dataframe.loc[geo_dataframe[column] != target]
                elif isinstance(target, list):
                    geo_dataframe = geo_dataframe.loc[~geo_dataframe[column].isin(
                        target)]
                else:
                    raise ValueError(
                        f"Invalid filter target type: {type(target)}")
            case "replace":
                ogs, new = target
                geo_dataframe.loc[:, column] = geo_dataframe\
                    .loc[:, column].replace(ogs, new)
            case "unique":
                geo_dataframe = geo_dataframe.loc[
                    ~geo_dataframe[column].duplicated(keep=False)]
            case _:
                raise ValueError(f"Invalid filter operator: {action}")

    if datetime_column is not None:
        geo_dataframe.loc[:, DATETIME_LABEL] = geo_dataframe\
            .loc[:, datetime_column]

    if strata_columns is not None and strata_list_path is not None:
        LOGGER.info(f"Saving strata list to {strata_list_path}...")
        geo_dataframe.loc[:, STRATA_LABEL] = geo_dataframe.loc[:, strata_columns].astype(
            str).apply(lambda x: '__'.join(x).replace(" ", "_"), axis=1)
        strata_list = geo_dataframe.groupby(STRATA_LABEL).size().to_dict()
        save_yaml(strata_list, strata_list_path)
    return geo_dataframe


@function_timer
def postprocess_data(
    chip_data: xr.Dataset,
    anno_data: xr.Dataset | None,
    postprocess_actions: list[dict[
        "data": Literal["chip", "anno"],
        "action": Literal["sum"],
        "operator": Literal[">", "<", ">=", "<=", "==", "!="],
        "targets": int, float, str, list]]):
    pre_calc = {"chip": {}, "anno": {}}
    data_map = {"chip": chip_data, "anno": anno_data}
    ops_map = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    for postprocess in postprocess_actions:
        data = postprocess["data"]
        action = postprocess["action"]
        compare = ops_map[postprocess["operator"]]
        target = postprocess["targets"]
        match action:
            case "sum":
                sums_ds = pre_calc[data].setdefault(
                    action, data_map[data].sum())
                to_drop = [v for v in sums_ds.data_vars if not compare(
                    sums_ds[v].values, target)]
                data_map[data] = data_map[data].drop_vars(
                    to_drop, errors="ignore")
            case _:
                raise ValueError(f"Invalid action: {action}")
    chip_vars = np.array(data_map["chip"].data_vars, dtype=int)
    anno_vars = np.array(data_map["anno"].data_vars, dtype=int)
    return np.intersect1d(chip_vars, anno_vars)


@function_timer
def stratified_sample(
        geo_dataframe: gpd.GeoDataFrame,
        fraction: Optional[float] = None,
        num_points: Optional[int] = None):
    if fraction is not None:
        groupby = geo_dataframe.groupby(STRATA_LABEL)
        sample = groupby.sample(frac=fraction)
    elif num_points is not None:
        groupby = geo_dataframe.groupby(STRATA_LABEL)
        sample = groupby.sample(n=num_points)
    else:
        sample = geo_dataframe
    sample = sample.reset_index().rename(columns={'index': 'meta_index'})
    return sample


@function_timer
def generate_centroid_squares(
        geo_dataframe: gpd.GeoDataFrame,
        meter_edge_size: int) -> gpd.GeoDataFrame:
    geo_dataframe.loc[:, "geometry"] = geo_dataframe.loc[:, "geometry"]\
        .apply(lambda p: p.centroid.buffer(meter_edge_size//2).envelope)
    return geo_dataframe


def centroid_shift_helper(
        polygon: Polygon,
        meter_edge_size: int) -> Polygon:
    centroid = polygon.centroid
    d = random.randint(-meter_edge_size//2, meter_edge_size//2)
    centroid_shift = transform(lambda x, y, z=None: (x+d, y+d), centroid)
    centroid_shift = centroid_shift.buffer(meter_edge_size//2).envelope
    return centroid_shift


@function_timer
def generate_centroid_shift_squares(
        geo_dataframe: gpd.GeoDataFrame,
        meter_edge_size: int) -> gpd.GeoDataFrame:
    geo_dataframe.loc[:, "geometry"] = geo_dataframe.loc[:, "geometry"]\
        .apply(lambda p: centroid_shift_helper(p, meter_edge_size))
    return geo_dataframe


@function_timer
def generate_squares(
        geo_dataframe: gpd.GeoDataFrame,
        method: Literal["convering_grid", "random", "gee_stratified", "centroid"],
        meta_data_path: str,
        meter_edge_size: int | float,
        gee_stratafied_config: Optional[dict] = None) -> tuple[gpd.GeoDataFrame, int]:
    LOGGER.info(f"Generating squares from sample points via {method}...")
    match method:
        case "convering_grid":
            raise NotImplementedError
        case "random":
            raise NotImplementedError
        case "gee_stratified":
            ee.Initialize()
            even_odd = True if epsg == "EPSG:4326" else False
            epsg = f"EPSG:{geo_dataframe.crs.to_epsg()}"
            area_of_interest = unary_union(geo_dataframe.geometry)
            area_of_interest = ee.Geometry.Polygon(
                list(area_of_interest.exterior.coords), proj=epsg, evenOdd=even_odd)
            gee_stratafied_config["area_of_interest"] = area_of_interest
            gee_stratafied_config["projection"] = epsg

            points = gee_stratified_sampling(**gee_stratafied_config)
            points = gee_download_features(points)\
                .set_crs("EPSG:4326")\
                .to_crs(epsg)
            gdf = generate_centroid_squares(points, meter_edge_size)
            gdf.loc[:, DATETIME_LABEL] = gee_stratafied_config["end_date"].r
        case "centroid":
            gdf = generate_centroid_squares(
                geo_dataframe,
                meter_edge_size)
        case "centroid_shift":
            gdf = generate_centroid_shift_squares(
                geo_dataframe,
                meter_edge_size)
        case _:
            raise ValueError(f"Invalid method: {method}")

    LOGGER.info(f"Saving squares geo dataframe to {meta_data_path}...")
    gdf.to_file(meta_data_path)

    return len(gdf), gdf


@function_timer
def generate_time_combinations(
        samples: np.array,
        num_times: int,
        time_step: int) -> np.ndarray:
    start = num_times % time_step
    time_arr = np.arange(num_times)[start::time_step]

    sample_arr, time_arr = np.meshgrid(samples, time_arr)
    sample_arr = sample_arr.flatten()
    time_arr = time_arr.flatten()
    time_arr = np.column_stack((sample_arr, time_arr))

    return time_arr


def rasterizer(polygons: gpd.GeoSeries,
               square: Polygon,
               scale: int,
               fill: int | float,
               default_value: int | float):
    cols = int((square.bounds[2] - square.bounds[0]) / scale)
    rows = int((square.bounds[3] - square.bounds[1]) / scale)

    transform = from_bounds(*square.bounds, rows, cols)
    raster = rasterize(
        shapes=polygons,
        out_shape=(rows, cols),
        fill=fill,
        transform=transform,
        default_value=default_value)

    return raster


def annotator(population_gdf: gpd.GeoDataFrame,
              sample_gdf: gpd.GeoDataFrame,
              groupby_columns: list[str | int],
              strata_list: dict[str, int],
              pixel_edge_size: int,
              scale: int,
              anno_data_path: str,
              io_limit: int,
              io_lock: Optional[mp.Lock] = None,
              index_queue: Optional[mp.Queue] = None):
    batch = []
    while (index := index_queue.get()) is not None:

        # getting annotation information from sample
        LOGGER.info(f"Rasterizing sample {index}...")
        target = sample_gdf.iloc[index]
        group = target[groupby_columns]
        square = target.geometry
        default_value = strata_list.index(target[STRATA_LABEL]) + 1

        # rasterizing multipolygon and clipping to square
        xarr_anno_list = []
        LOGGER.info(
            f"Creating annotations for sample {index} from strata list...")

        # strata list should already be in order of index value
        for strata in strata_list:
            try:
                mask = (population_gdf[groupby_columns] == group).all(axis=1) & \
                    (population_gdf[STRATA_LABEL] == strata)
                mp = population_gdf[mask].geometry
                if len(mp) == 0:
                    annotation = np.zeros((pixel_edge_size, pixel_edge_size))
                else:
                    annotation = rasterizer(
                        mp, square, scale, NO_DATA_VALUE, default_value)
            except Exception as e:
                raise e
            annotation = xr.DataArray(annotation, dims=["y", "x"])

            # clipping or padding to pixel_edge_size
            if pixel_edge_size > min(annotation["x"].size,  annotation["y"].size):
                annotation = pad_xy_xarray(annotation, pixel_edge_size)
            if pixel_edge_size < max(annotation["x"].size,  annotation["y"].size):
                annotation = clip_xy_xarray(annotation, pixel_edge_size)
            xarr_anno_list.append(annotation)
        xarr_anno = xr.concat(xarr_anno_list, dim=STRATA_LABEL)

        # writing in batches to avoid io bottleneck
        LOGGER.info(f"Appending rasterized sample {index} to batch...")
        xarr_anno.name = str(index)
        batch.append(xarr_anno)
        if len(batch) == io_limit:
            xarr_anno = xr.merge(batch)
            with io_lock:
                xarr_anno.to_zarr(store=anno_data_path, mode="a")
            batch.clear()
            indices = list(xarr_anno.data_vars)
            LOGGER.info(f"Rasterized sample batch submitted... {indices}")

    # writing remaining batch
    if len(batch) > 0:
        with io_lock:
            xarr_anno = xr.merge(batch)
            xarr_anno.to_zarr(store=anno_data_path, mode="a")


@function_timer
def generate_annotation_data(
        population_gdf: gpd.GeoDataFrame,
        sample_gdf: gpd.GeoDataFrame,
        groupby_columns: list[str],
        strata_list_path: str,
        pixel_edge_size: int,
        scale: int,
        anno_data_path: str,
        num_workers: int,
        io_limit: int,):
    num_samples = len(sample_gdf)
    strata_list = sorted(load_yaml(strata_list_path))

    manager = mp.Manager()
    index_queue = manager.Queue()
    io_lock = manager.Lock()

    LOGGER.info(
        f"Starting parallel process for {num_samples} samples using {num_workers}...")
    [index_queue.put(i) for i in range(num_samples)]
    [index_queue.put(None) for _ in range(num_workers)]
    annotators = set()
    for _ in range(num_workers):
        p = mp.Process(
            target=annotator,
            args=(population_gdf,
                  sample_gdf,
                  groupby_columns,
                  strata_list,
                  pixel_edge_size,
                  scale,
                  anno_data_path,
                  io_limit,
                  io_lock,
                  index_queue),
            daemon=True)
        p.start()
        annotators.add(p)
    [r.join() for r in annotators]


@function_timer
def generate_image_chip_data(downloader_kwargs: dict):
    downloader = Downloader(**downloader_kwargs)
    downloader.start()


@function_timer
def sample():
    try:
        LOGGER.info("Loading geo file into GeoDataFrame...")
        geo_dataframe = gpd.read_file(GEO_RAW_PATH)

        if SAMPLER_CONFIG["sample_toggles"]["preprocess_data"]:
            LOGGER.info("Preprocessing data in geo file...")
            sample_config = {
                "geo_dataframe": geo_dataframe,
                "preprocess_actions": SAMPLER_CONFIG["preprocess_actions"],
                "projection": SAMPLER_CONFIG["projection"],
                "strata_list_path": STRATA_LIST_PATH,
                "strata_columns": SAMPLER_CONFIG["strata_columns"],
                "datetime_column": SAMPLER_CONFIG["datetime_column"],
            }
            geo_dataframe = preprocess_data(**sample_config)
            LOGGER.info(f"Saving processed geo dataframe to {GEO_PRE_PATH}...")
            geo_dataframe.to_file(GEO_PRE_PATH)

        if SAMPLER_CONFIG["sample_toggles"]["stratified_sample"]:
            LOGGER.info("Stratifying data in geo file...")
            group_config = {
                "geo_dataframe": geo_dataframe,
                "fraction": SAMPLER_CONFIG["fraction"],
                "num_points": SAMPLER_CONFIG["num_points"],
            }
            geo_dataframe = stratified_sample(**group_config)

        if SAMPLER_CONFIG["sample_toggles"]["generate_squares"]:
            LOGGER.info("Generating squares...")
            square_config = {
                "geo_dataframe": geo_dataframe,
                "method": SAMPLER_CONFIG["method"],
                "meta_data_path": META_DATA_PATH,
                "meter_edge_size": SAMPLER_CONFIG["scale"] * SAMPLER_CONFIG["pixel_edge_size"],
                "gee_stratafied_config": SAMPLER_CONFIG["gee_stratafied_config"],
            }
            _, _ = generate_squares(**square_config)

    except Exception as e:
        LOGGER.critical(f"Failed to generate sample: {type(e)} {e}")
        raise e


@function_timer
def annotate():
    try:
        LOGGER.info("Loading geo files into GeoDataFrame...")
        if os.path.exists(GEO_PRE_PATH):
            population_gdf = gpd.read_file(GEO_PRE_PATH)
        else:
            population_gdf = gpd.read_file(GEO_RAW_PATH)
        sample_gdf = gpd.read_file(META_DATA_PATH)

        LOGGER.info("Generating annotations...")
        annotation_config = {
            "population_gdf": population_gdf,
            "sample_gdf": sample_gdf,
            "groupby_columns": SAMPLER_CONFIG["groupby_columns"],
            "strata_list_path": STRATA_LIST_PATH,
            "pixel_edge_size": SAMPLER_CONFIG["pixel_edge_size"],
            "scale": SAMPLER_CONFIG["scale"],
            "anno_data_path": ANNO_DATA_PATH,
            "num_workers": SAMPLER_CONFIG["num_workers"],
            "io_limit": SAMPLER_CONFIG["io_limit"],
        }
        generate_annotation_data(**annotation_config)
    except Exception as e:
        LOGGER.critical(f"Failed to generate annotations: {type(e)} {e}")
        raise e


@function_timer
def download():
    try:
        LOGGER.info("Loading geo file into GeoDataFrame...")
        gdf = gpd.read_file(META_DATA_PATH)

        LOGGER.info("Generating image chips...")
        parser_kwargs = SAMPLER_CONFIG["medoid_config"]
        parser_kwargs["look_range"] = SAMPLER_CONFIG["look_range"]
        downloader_kwargs = {
            "file_type": SAMPLER_CONFIG["file_type"],
            "overwrite": SAMPLER_CONFIG["overwrite"],
            "scale": SAMPLER_CONFIG["scale"],
            "pixel_edge_size": SAMPLER_CONFIG["pixel_edge_size"],
            "projection": SAMPLER_CONFIG["projection"],
            "chip_data_path": CHIP_DATA_PATH,
            "meta_data": gdf,
            "meta_data_parser": parse_meta_data,
            "image_expr_generator": lt_image_generator,
            "image_reshaper": zarr_reshape,
            "num_workers": SAMPLER_CONFIG["gee_workers"],
            "io_limit": SAMPLER_CONFIG["io_limit"],
            "logger": LOGGER,
            "parser_kwargs": parser_kwargs,
        }
        generate_image_chip_data(downloader_kwargs)
    except Exception as e:
        LOGGER.critical(f"Failed to download images: {type(e)} {e}")
        raise e


@function_timer
def calculate():
    try:
        LOGGER.info("Calculating sample statistics...")
        stat = {}
        match SAMPLER_CONFIG["file_type"]:
            case "ZARR":
                LOGGER.info(f"Verifying chip data...")
                chip_data = xr.open_zarr(CHIP_DATA_PATH)
                if SAMPLER_CONFIG["sample_toggles"]["postprocess_data"]:
                    if os.path.exists(ALL_SAMPLE_PATH):
                        all_samples = [str(i)
                                       for i in np.load(ALL_SAMPLE_PATH)]
                        chip_data = chip_data[all_samples]
                stat["chip_means"], stat["chip_stds"] = get_xarr_chip_mean_std(
                    chip_data)
                stat["chip_count"] = len(chip_data.variables)
                stat["chip_verify"] = {
                    i: s for i, s in test_non_zero_sum(chip_data, RANDOM_SEED)}
                if os.path.isdir(ANNO_DATA_PATH):
                    LOGGER.info(f"Verify annotation data...")
                    anno_data = xr.open_zarr(ANNO_DATA_PATH)
                    if SAMPLER_CONFIG["sample_toggles"]["postprocess_data"]:
                        if os.path.exists(ALL_SAMPLE_PATH):
                            anno_data = anno_data[all_samples]
                    stat["anno_totals"], stat["anno_weights"] = get_class_weights(
                        anno_data)
                    stat["anno_means"], stat["anno_stds"] = get_xarr_anno_mean_std(
                        anno_data)
                    stat["anno_verify"] = {
                        i: s for i, s in test_non_zero_sum(anno_data, RANDOM_SEED)}
            case _:
                raise ValueError(
                    f"Invalid file type: {SAMPLER_CONFIG['file_type']}")
        update_yaml(stat, STAT_DATA_PATH)
    except Exception as e:
        LOGGER.critical(
            f"Failed to calculate sample statistics: {type(e)} {e}")
        raise e


@function_timer
def index():
    chip_data = xr.open_zarr(CHIP_DATA_PATH)
    anno_data = xr.open_zarr(ANNO_DATA_PATH) if \
        os.path.exists(ANNO_DATA_PATH) else None
    num_samples = len(chip_data)
    samples = np.arange(num_samples)

    if SAMPLER_CONFIG["sample_toggles"]["postprocess_data"]:
        LOGGER.info("Postprocessing data in geo file...")
        sample_config = {
            "chip_data": chip_data,
            "anno_data": anno_data,
            "postprocess_actions": SAMPLER_CONFIG["postprocess_actions"],
        }
        # TODO: resample data so sample ratio stays consistent postprocessing
        samples = postprocess_data(**sample_config)
    np.save(ALL_SAMPLE_PATH, samples)

    if SAMPLER_CONFIG["sample_toggles"]["generate_time_combinations"]:
        LOGGER.info("Generating time combinations...")
        num_times = SAMPLER_CONFIG["look_range"] + 1
        time_sample_config = {
            "samples": samples,
            "num_times": num_times,
            "time_step": SAMPLER_CONFIG["time_step"]
        }
        samples = generate_time_combinations(**time_sample_config)

    if SAMPLER_CONFIG["sample_toggles"]["generate_train_test_splits"]:
        LOGGER.info(
            "Splitting sample data into training, validation, test, and predict sets...")
        train, validate = train_test_split(
            samples, test_size=SAMPLER_CONFIG["validate_ratio"])
        validate, test = train_test_split(
            validate, test_size=SAMPLER_CONFIG["test_ratio"])
        test, predict = train_test_split(
            test, test_size=SAMPLER_CONFIG["predict_ratio"])

        LOGGER.info("Saving sample data splits to paths...")
        idx_lsts = [train, validate, test, predict]
        paths = [TRAIN_SAMPLE_PATH,
                 VALIDATE_SAMPLE_PATH,
                 TEST_SAMPLE_PATH,
                 PREDICT_SAMPLE_PATH]
        for path, idx_lst in zip(paths, idx_lsts):
            np.save(path, idx_lst)
        update_yaml(
            {
                "train_count": len(train),
                "validate_count": len(validate),
                "test_count": len(test),
                "predict_count": len(predict),
            },
            STAT_DATA_PATH)
    else:
        LOGGER.info("Saving sample data indices to paths...")
        np.save(PREDICT_SAMPLE_PATH, samples)
        update_yaml({"train_count": len(samples)}, STAT_DATA_PATH)
