import rasterio
import numpy as np
from abc import ABC, abstractmethod
import ast


class DataPostProcessing(ABC):

    def __init__(self, exportpath: str):
        self.exportpath = exportpath

    @abstractmethod
    def export2file(self, data: list):
        return


class RasterioPostProcessing(DataPostProcessing):

    def __init__(self, exportpath):
        super().__init__(exportpath)

    def export2file(self, data: list):
        predictions, filenames, profiles = data[0], data[2], data[3]
        if (len(predictions) != len(filenames)):
            raise Exception(f"{__class__}: len of predictions({len(predictions)}) != filesnames({len(filenames)})")
        if (len(profiles) != 1 and (len(profiles) != len(predictions))):
            raise Exception(f"{__class__}: len of profiles({len(profiles)}) must be either 1 or equal to predictions({len(predictions)})")
        profiles = [profiles[0]] * len(predictions) if (len(profiles) == 1) else profiles
        for i, p in enumerate(predictions):
            profile = eval(profiles[i])
            profile['crs'] = rasterio.crs.CRS.from_dict(profile['crs'])
            profile['transform'] = rasterio.Affine.from_gdal(*profile['transform'])
            profile['nodata'] = None if (profile['nodata'] == 'None') else profile['nodata']
            profile['nodata'] = float("nan") if (profile['nodata'] == 'nan') else profile['nodata']
            with rasterio.open(f'{self.exportpath}/{filenames[i]}', 'w', **profile) as dst:
                [dst.write(p[b], b + 1) for b in range(p.shape[0])]
