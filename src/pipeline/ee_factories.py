import ee

from ltgee import LandsatComposite
from datetime import datetime


LBANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
SBANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]


class EEBase:
    format_string = "%Y-%m-%d"
    def __init__(self, start_date, end_date, projection, start_date_lo=None, end_date_lo=None, **kwargs):
        self.start_date = start_date
        self.end_date = end_date
        self.projection = projection
        self.start_date_lo = start_date_lo
        self.end_date_lo = end_date_lo

        if type(self.start_date) == str:
            self.start_date = datetime.strptime(self.start_date, self.format_string)
        if type(self.end_date) == str:
            self.end_date = datetime.strptime(self.end_date, self.format_string)
        if type(self.start_date_lo) == str:
            self.start_date_lo = datetime.strptime(self.start_date_lo, self.format_string)
        if type(self.end_date_lo) == str:
            self.end_date_lo = datetime.strptime(self.end_date_lo, self.format_string)
    
    def create_ee_image(self):
        raise NotImplementedError
    

class LTMedoidImages(EEBase):
    def __init__(self, mask_labels=["cloud", "shadow", "water"], **kwargs):
        super().__init__(**kwargs)
        self.mask_labels = mask_labels
        
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
    

class LSMedianImage(EEBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def create_ee_image(self, coordinates):
        even_odd = (self.projection == "EPSG:4326")
        polygon = ee.Geometry.Polygon(coordinates, proj=self.projection, evenOdd=even_odd)

        image = None
        for year in range(self.start_date.year, self.end_date.year+1):
            start_date = ee.Date.fromYMD(
                year = year,
                month = self.start_date.month,
                day = self.start_date.day)
            end_date = ee.Date.fromYMD(
                year = year,
                month = self.end_date.month,
                day = self.end_date.day)
            yimage = self.get_ls_combined_sr_collection()\
                            .filterBounds(polygon)\
                            .filter(ee.Filter.date(start_date, end_date))\
                            .median()
                            
            if self.start_date_lo is not None and self.end_date_lo is not None:
                start_date = ee.Date.fromYMD(
                    year = year-1,
                    month = self.start_date_lo.month,
                    day = self.start_date_lo.day)
                end_date = ee.Date.fromYMD(
                    year = year,
                    month = self.end_date_lo.month,
                    day = self.end_date_lo.day)        
                loyimage = self.get_ls_combined_sr_collection()\
                                .filterBounds(polygon)\
                                .filter(ee.Filter.date(start_date, end_date))\
                                .median()
                yimage = loyimage.addBands(yimage)
            
            if image is None:
                image = yimage
            else:
                image = image.addBands(yimage)

        return image

    def get_ls_combined_sr_collection(self):
        lt5 = self.get_ls_sr_collection('LT05')
        le7 = self.get_ls_sr_collection('LE07')
        # SLC-OFF for Landsat 7 after 2003-05-31
        le7 = le7.filter(ee.Filter.lte('system:time_start', 1054425600000))
        lc8 = self.get_ls_sr_collection('LC08')
        lc9 = self.get_ls_sr_collection('LC09')
        return lt5.merge(le7).merge(lc8).merge(lc9)

    def get_ls_sr_collection(self, sensor: str):
        ls_col = ee.ImageCollection(f'LANDSAT/{sensor}/C02/T1_L2')
        ls_col = ls_col.map(lambda image: self.preprocess_ls_image(image, sensor))
        return ls_col

    def preprocess_ls_image(self, image: ee.Image, sensor: str):
        qa = image.select('QA_PIXEL')
        cloud = qa.bitwiseAnd(1 << 3).eq(0)
        shadow = qa.bitwiseAnd(1 << 4).eq(0)
        mask = cloud.multiply(shadow)
        image = image.updateMask(mask)

        if sensor == 'LC08' or sensor == 'LC09':
            image = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
                                LBANDS)
        else:
            image = image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
                                LBANDS)
        return image
        

class Sentinel2MedianImage(EEBase):
    _sr_band_scale = 1e4

    def __init__(self,
                 cloud_filter: int = 100,
                 cloud_prb_thresh: int = 65,
                 nir_drk_thresh: float = 0.2,
                 cloud_prj_dist: float = 1.5,
                 buffer: int = 50,
                 **kwargs):
        super().__init__(**kwargs)
        self.cloud_filter = cloud_filter
        self.cloud_prb_thresh = cloud_prb_thresh
        self.nir_drk_thresh = nir_drk_thresh
        self.cloud_prj_dist = cloud_prj_dist
        self.buffer = buffer

    def create_ee_image(self, coordinates):
        even_odd = (self.projection == "EPSG:4326")
        polygon = ee.Geometry.Polygon(coordinates, proj=self.projection, evenOdd=even_odd)

        image = None
        for year in range(self.start_date.year, self.end_date.year+1):
            start_date = ee.Date.fromYMD(
                year = year,
                month = self.start_date.month,
                day = self.start_date.day)
            end_date = ee.Date.fromYMD(
                year = year,
                month = self.end_date.month,
                day = self.end_date.day)
            yimage = self._build_median_mosaic(polygon, start_date, end_date)
            
            if self.start_date_lo is not None and self.end_date_lo is not None:
                start_date = ee.Date.fromYMD(
                    year = year-1,
                    month = self.start_date_lo.month,
                    day = self.start_date_lo.day)
                end_date = ee.Date.fromYMD(
                    year = year,
                    month = self.end_date_lo.month,
                    day = self.end_date_lo.day)        
                loyimage = self._build_median_mosaic(polygon, start_date, end_date)
                yimage = loyimage.addBands(yimage)

            if image is None:
                image = yimage
            else:
                image = image.addBands(yimage)

        return image

    def _build_median_mosaic(self,
                             area_of_interest, start_date, end_date):
        collection = self._get_sr_collection(area_of_interest, start_date, end_date)
        collection = collection.select(SBANDS)
        median = collection.median()
        
        return median

    def _get_sr_collection(self, area_of_interest, start_date, end_date):
        s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.cloud_filter))\
                    .filterBounds(area_of_interest)\
                    .filterDate(start_date, end_date)

        s2_cloud_prob = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')\
                        .filterBounds(area_of_interest)\
                        .filterDate(start_date, end_date)
                    
        collection = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(
            primary=s2_sr,
            secondary=s2_cloud_prob,
            condition=ee.Filter.equals(
                leftField='system:index', rightField='system:index'
            )
        ))
        collection = collection.map(self._preprocess_image)
        
        return collection

    def _preprocess_image(self, image):
        image = self._add_cloud_shdw_mask(image)
        image = self._apply_cloud_shdw_mask(image)
        return image

    def _index_join(self, collectionA, collectionB, property_name):
        joined = ee.ImageCollection(ee.Join.saveFirst(property_name).apply(
            primary=collectionA,
            secondary=collectionB,
            condition=ee.Filter.equals(
                leftField='system:index', rightField='system:index'
            )
        ))
        return joined.map(lambda image: image.addBands(ee.Image(image.get(property_name))))

    def _add_cloud_bands(self, image):
        cloud_prb = ee.Image(image.get('s2cloudless')).select('probability')
        is_cloud = cloud_prb.gt(self.cloud_prb_thresh).rename('clouds')
        return image.addBands(ee.Image([cloud_prb, is_cloud]))

    def _add_shadow_bands(self, image):
        not_water = image.select('SCL').neq(6)
        dark_pixels = image.select('B8').lt(self.nir_drk_thresh * self._sr_band_scale).multiply(not_water).rename('dark_pixels')

        shadow_azimuth = ee.Number(90).subtract(ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
        cloud_proj = (image.select('clouds').directionalDistanceTransform(shadow_azimuth, self.cloud_prj_dist * 10)
                    .reproject(crs=image.select(0).projection(), scale=100)
                    .select('distance')
                    .mask()
                    .rename('cloud_transform'))

        shadows = cloud_proj.multiply(dark_pixels).rename('shadows')
        return image.addBands(ee.Image([dark_pixels, cloud_proj, shadows]))

    def _add_cloud_shdw_mask(self, image):
        image = self._add_cloud_bands(image)
        image = self._add_shadow_bands(image)

        is_cloud_shdw = image.select('clouds').add(image.select('shadows')).gt(0)
        is_cloud_shdw = (is_cloud_shdw.focalMin(2).focalMax(self.buffer * 2 / 20)
                       .reproject(crs=image.select([0]).projection(), scale=10)
                       .rename('cloudshadowmask'))

        return image.addBands(is_cloud_shdw)

    def _apply_cloud_shdw_mask(self, image):
        not_cloud_shdw = image.select('cloudshadowmask').Not()
        return image.select('B.*').updateMask(not_cloud_shdw)
    

class NLCD2019(EEBase):
    def create_ee_image(self, coordinates):
        nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD')
        nlcd2019 = nlcd.filter(ee.Filter.eq('system:index', '2019')).first()
        landcover = nlcd2019.select('landcover')
        return landcover 


class NLCD2019LS(LSMedianImage):
    def create_ee_image(self, coordinates):
        ls_image = super().create_ee_image(coordinates)
        nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD')
        nlcd2019 = nlcd.filter(ee.Filter.eq('system:index', '2019')).first()
        landcover = nlcd2019.select('landcover')
        return landcover.addBands(ls_image)

    
class NLCD2019S2(Sentinel2MedianImage):
    def create_ee_image(self, coordinates):
        s2_image = super().create_ee_image(coordinates)
        nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD')
        nlcd2019 = nlcd.filter(ee.Filter.eq('system:index', '2019')).first()
        landcover = nlcd2019.select('landcover')
        return landcover.addBands(s2_image)