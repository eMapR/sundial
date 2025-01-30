import ee

from ltgee import LandsatComposite
from datetime import datetime


def lt_medoid_image_factory(
        square_coords: list[tuple[float, float]],
        start_date: datetime,
        end_date: datetime,
        # pixel_edge_size: int,
        # scale: int,
        projection: str,
        mask_labels: list[str] = ["snow", "cloud", "shadow"]) -> ee.Image:
    # TODO: actually parse the projection string
    even_odd = (projection == "EPSG:4326")
    square = ee.Geometry.Polygon(square_coords, proj=projection, evenOdd=even_odd)
    collection = LandsatComposite(
        start_date=start_date,
        end_date=end_date,
        area_of_interest=square,
        mask_labels=mask_labels,
        exclude={"slcOff": True}
    )
    size = 1 + end_date.year - start_date.year

    old_band_names = [f"{str(i)}_{band}" for i in range(size)
                      for band in collection._band_names]
    new_band_names = [f"{str(start_date.year + i)}_{band}" for i in range(size)
                      for band in collection._band_names]

    # TODO: fix hacky filter bounds to reprojections
    image = collection\
        .toBands()\
        .select(old_band_names, new_band_names)\
        .divide(10000)

    return image


def lt_medoid_image_factory_forward(*args, **kwargs):
    """factory to increment end year by one"""
    square_coords, start_date, end_date, epsg_str = args
    end_date = end_date.replace(year=end_date.year + 1)
    return lt_medoid_image_factory(square_coords, start_date, end_date, epsg_str)