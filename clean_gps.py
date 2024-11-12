import pandas as pd
from pandas.tseries.offsets import DateOffset
from shapely.geometry import Polygon
import numpy as np
import geopandas as gpd
import os
from points_to_boxes import points_to_boxes


"""
OPTIONS FOR PARAMETERS:
temporal_res = ['daily', 'weekly', 'biweekly', 'monthly']
box_length_m: numeric, meters
years_cut_from_back: numeric, yrs --> As we know, the dataset's GPS trackers started dying in 2014. The 
    following parameter allows control over how many years are cut since 2018. 
    No more than 4 years is recommended.
keep_geometry_col: bool
"""

# defining date range
DATE_RANGE_TRANSLATOR = {  
    'daily': 'D',
    'weekly': 'W',
    'biweekly': '2W',
    'monthly': 'M'
}
# how much temporal buffer to give based on resolution
DATE_OFFSET_TRANSLATOR = {  
    'daily': 1,
    'weekly': 7,
    'biweekly': 14,
    'monthly': 30
}
# naming the temporal column
DATE_NAME_TRANSLATOR = {  
    'daily': 'day',
    'weekly': 'week',
    'biweekly': 'biweek',
    'monthly': 'month'
}

METERS_PER_DEGREE = 111111

# Function to generate a grid of boxes
def generate_grid(bbox, spacing, crs):
    """
    Generate box grid based on min x, min y, max x, and max y (LONG/LAT)
    Spacing: Space between each box in degrees
    Crs: Coordinate reference system
    """
    METERS_PER_DEGREE = 111111

    if crs.to_string() == 'EPSG:26914':
        spacing = spacing * METERS_PER_DEGREE

    minx, miny, maxx, maxy = bbox
    x_coords = np.arange(minx, maxx, spacing)
    y_coords = np.arange(miny, maxy, spacing)

    grid = []
    for x in x_coords:
        for y in y_coords:
            grid.append(Polygon([(x, y), (x + spacing, y), (x + spacing, y + spacing), (x, y + spacing), (x, y)]))
    return gpd.GeoDataFrame({'geometry': grid}, crs=crs)



def read_gps(years_cut_from_back=0, temporal_res='weekly', keep_geometry_col=True):
    # set gps gdf and CRS
    gps_gdf = pd.read_csv('raw-data/WHCR_locations_gps.csv')
    gps_gdf = gpd.GeoDataFrame(gps_gdf, geometry=gpd.points_from_xy(gps_gdf.Long, gps_gdf.Lat), crs='EPSG:4326')
    gps_gdf = gps_gdf.set_crs('EPSG:4326', allow_override=True)
    gps_gdf = gps_gdf.to_crs('EPSG:26914')
    gps_gdf['Date'] = pd.to_datetime(gps_gdf['Date'])
    gps_gdf['Year'] = gps_gdf['Date'].dt.year
    valid_years = gps_gdf['Year'].unique()[:len(gps_gdf['Year'].unique()) - years_cut_from_back]
    gps_gdf = gps_gdf[gps_gdf['Year'].isin(valid_years)]

    all_dates = pd.date_range(start=gps_gdf['Date'].min() - DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), end=gps_gdf['Date'].max() + DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), freq=DATE_RANGE_TRANSLATOR[temporal_res])
    gps_gdf[DATE_NAME_TRANSLATOR[temporal_res]] = np.searchsorted(all_dates, gps_gdf['Date'])  

    # add names for weeks for data clarity
    bin_names = {i + 1: f'{all_dates[i].date()}_to_{all_dates[i + 1].date()}' for i in range(len(all_dates) - 1)}
    gps_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] = gps_gdf[DATE_NAME_TRANSLATOR[temporal_res]].map(bin_names)

    gps_gdf = gps_gdf.rename(columns={'Date': 'date', 'Long': 'X', 'Lat': 'Y'})
    gps_gdf['count'] = 1

    columns_of_interest = ['date', 'week', 'week_name', 'X', 'Y', 'count', 'geometry']
    
    return gps_gdf[columns_of_interest]



if __name__ == '__main__':
    
    # BEGIN PARAMETERS
    box_length_m = 500
    temporal_res = 'weekly'
    years_cut_from_back = 4
    keep_geometry_col = False  # Must keep this true for now so data can have crs
    complete_idx_square = False  # I believe square is already completed for this dataset as all possible tsteps are included
    # END PARAMETERS

    # set gps gdf and CRS
    gps_gdf = read_gps(years_cut_from_back, temporal_res, keep_geometry_col)

    file_id = f"2009_to_{2018 - years_cut_from_back}_{temporal_res}_{box_length_m}M"
    filename = f"gps_{file_id}_RawPts"
    os.makedirs(f"gps/{file_id}", exist_ok=True)
    gps_gdf.to_csv(f"gps/{file_id}/{filename}.csv")

    points_to_boxes(gps_gdf, study='gps', temporal_res=temporal_res, box_length_m=box_length_m, 
    keep_geometry_col=keep_geometry_col, complete_idx_square=complete_idx_square, years_cut_from_back=years_cut_from_back)
