import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box


def region_assign(lon: float, lat: float, shape_df: gpd.GeoDataFrame,
                  col_name: str, res: float) -> str:
    """

    Parameters
    ----------
    lon
        Longitudinal coordinate in the geometry of shape_df
    lat
        Latitudinal coordinate in the geometry of shape_df
    shape_df
        GeoPandas DataFrame containing an assigned region for each exclusive geometry
    col_name
        Column name
    res

    Returns
    -------

    """

    cell_box = box(lon - res / 2, lat - res / 2, lon + res / 2, lat + res / 2)

    intersecting_regions = (shape_df
                           .loc[lambda dd: dd.geometry.apply(lambda x: cell_box.intersects(x))]
        )
    if intersecting_regions.shape[0] == 0:
        return np.NaN
    elif intersecting_regions.shape[0] == 1:
        return intersecting_regions[col_name].values[0]
    else:
        intersecting_regions = (
            intersecting_regions
                .assign(area=lambda dd: dd.geometry.apply(lambda x: x.intersection(cell_box).area))
                .loc[lambda dd: dd['area'] == dd['area'].max()])
        return intersecting_regions[col_name].values[0]
