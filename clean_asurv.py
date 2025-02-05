import pandas as pd
from pandas.tseries.offsets import DateOffset
from shapely.geometry import Polygon
import numpy as np
import geopandas as gpd
import os
from points_to_boxes import points_to_boxes
import argparse

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

# meters per degree lat or long
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


def read_asurv(years_through_2011=10, temporal_res='weekly', keep_geometry_col=True):
    """
    Reads raw asurv data
    Assigns each observation into a temporal bucket based on temporal resolution
    """
    
    # read and turn into a geopandas dataframe
    df = pd.read_csv('raw-data/asurv_1950_to_2011/WHCR_Aerial_Observations_1950_2011.txt', encoding='latin1', sep='\t')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs='EPSG:26914')
    
    # cut years based on function parameter
    gdf = gdf[gdf['Year'].isin(gdf['Year'].unique()[-years_through_2011:])]

    # add time resolution
    gdf['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    all_dates = pd.date_range(start=gdf['date'].min() - DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), end=gdf['date'].max() + DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), freq=DATE_RANGE_TRANSLATOR[temporal_res])
    gdf[DATE_NAME_TRANSLATOR[temporal_res]] = np.searchsorted(all_dates, gdf['date'])  

    # add names for weeks for data clarity
    bin_names = {i + 1: f'{all_dates[i].date()}_to_{all_dates[i + 1].date()}' for i in range(len(all_dates) - 1)}
    gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] = gdf[DATE_NAME_TRANSLATOR[temporal_res]].map(bin_names)

    gdf['count'] = gdf['WHITE'].fillna(0) + gdf['JUVE'].fillna(0)  + gdf['UNK'].fillna(0) 

    if keep_geometry_col:
        columns_of_interest = ['date', 'week', 'week_name', 'X', 'Y', 'count', 'geometry']
    else:
        columns_of_interest = ['date', 'week', 'week_name', 'X', 'Y', 'count']

    return gdf[columns_of_interest]


if __name__ == '__main__':


    """
    OPTIONS FOR PARAMETERS:
    years_through_2011: numeric, integer
    temporal_res = ['weekly']
    box_length_m: numeric, meters
    complete_idx_square: bool. --> this dataset does not come with the square completed, 
        so can switch this to true to complete square. Default is false.
    keep_geometry_col: bool. --> saves a lot of space if this is set to false. default is true. 
    """
    # argparse, add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--years_through_2011', type=int, choices=range(10, 51), default=40, help='number of years before 2011 to use')
    parser.add_argument('--temporal_res', choices=['weekly'], default='weekly', help='Temporal resolution for timesteps')
    parser.add_argument('--box_length_m', type=int, default=500, help='Length of box in meters')
    parser.add_argument('--complete_idx_square', type=bool, default=True, help='this dataset does not come with the square completed, so can switch this to true to complete square. Default is false.')
    parser.add_argument('--keep_geometry_col', type=bool, default=True, help='saves a lot of space if this is set to false. default is true. ')
    
    args = parser.parse_args()
    years_through_2011 = args.years_through_2011
    temporal_res = args.temporal_res
    box_length_m = args.box_length_m
    complete_idx_square = args.complete_idx_square
    keep_geometry_col = args.keep_geometry_col
    # END PARAMS

    asurv_gdf = read_asurv(years_through_2011, temporal_res=temporal_res)
    file_id = f"{2011 - years_through_2011 + 1}_to_2011_{temporal_res}_{box_length_m}M"
    filename = f"asurv_{file_id}_RawPts"
    os.makedirs(f"asurv/{file_id}", exist_ok=True)
    asurv_gdf.to_csv(f"asurv/{file_id}/{filename}.csv")

    points_to_boxes(asurv_gdf, study='asurv', temporal_res=temporal_res, box_length_m=box_length_m, 
    keep_geometry_col=keep_geometry_col, complete_idx_square=complete_idx_square, years_through_2011=years_through_2011)