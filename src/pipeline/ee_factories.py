import ee

from ltgee import LandsatComposite
from datetime import datetime


class LTMedoidImage:
    def __init__(self, start_date, end_date, projection, mask_labels=["cloud", "shadow", "water"]):
        self.start_date = start_date
        self.end_date = end_date
        self.projection = projection
        self.mask_labels = mask_labels
        
        format_string = "%Y-%m-%d"
        if type(self.start_date) == str:
            self.start_date = datetime.strptime(self.start_date, format_string)
        if type(self.end_date) == str:
            self.end_date = datetime.strptime(self.end_date, format_string)
        
    def create_ee_image(self, coordinates):
        even_odd = (self.projection == "EPSG:4326")
        polygon = ee.Geometry.Polygon(coordinates, proj=self.projection, evenOdd=even_odd)
        collection = LandsatComposite(
            start_date=self.start_date,
            end_date=self.end_date,
            area_of_interest=polygon,
            mask_labels=self.mask_labels,
            exclude={"slcOff": True}
        )
        size = 1 + self.end_date.year - self.start_date.year

        old_band_names = [f"{str(i)}_{band}" for i in range(size)
                        for band in collection._band_names]
        new_band_names = [f"{str(self.start_date.year + i)}_{band}" for i in range(size)
                        for band in collection._band_names]

        image = collection\
            .toBands()\
            .select(old_band_names, new_band_names)\
            .divide(10000)

        return image
