import pandas as pd
from pandas.tseries.offsets import DateOffset
from shapely.geometry import Polygon
import numpy as np
import geopandas as gpd
import os
import ast

"""
OPTIONS FOR PARAMETERS:
years_through_2011: numeric, integer
temporal_res = ['daily', 'weekly', 'biweekly', 'monthly']
box_length_m: numeric, meters
complete_idx_square: bool. --> this dataset does not come with the square completed, 
    so can switch this to true to complete square. Default is false.
keep_geometry_col: bool. --> saves a lot of space if this is set to false. default is true. 
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

# meters per degree lat or long
METERS_PER_DEGREE = 111111

def get_aransas_box_bounds():
    """
    If we don't have access to aransas box bounds recreate them
    """

    # TODO untested
    # Set spacing and generate the grid
    aransas_df = pd.read_csv('raw-data/asurv_1950_to_2011/WHCR_Aerial_Observations_1950_2011.txt', encoding='latin1', sep='\t')
    aransas_gdf = gpd.GeoDataFrame(aransas_df, geometry=gpd.points_from_xy(aransas_df.X, aransas_df.Y), crs='EPSG:26914')
    bbox = aransas_gdf.total_bounds
    box_length_degrees = box_length_m / METERS_PER_DEGREE
    grid_gdf = generate_grid(bbox, box_length_degrees, crs=aransas_gdf.crs)
    return grid_gdf.total_bounds

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


def points_to_boxes(gdf, study, temporal_res, box_length_m, keep_geometry_col, complete_idx_square, **kwargs):
    
    """
    Given gdf with x;y;count;tstep, assign counts to boxes 
    """
    if 'years_through_2011' in kwargs.keys():
        years_through_2011 = kwargs['years_through_2011']
    if 'years_cut_from_back' in kwargs.keys():
        years_cut_from_back = kwargs['years_cut_from_back']
    if 'save_shp_folder' in kwargs.keys():
        save_shp_folder = kwargs['save_shp_folder']

    all_gdfs = []
    for tstep in np.sort(gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'].unique()):
        
        filtered_gdf = gdf[gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] == tstep]
        # Read in bounding box from data folder
        with open("raw-data/boxes_total_bounds.txt", "r") as file:
            bounds = file.read()
        
        min_x, min_y, max_x, max_y = ast.literal_eval(bounds)

        # MAKE SURE we are in the CRS that measures by meters, not lat/long
        assert (filtered_gdf.crs.to_string() == 'EPSG:26914')
        grid_cells = []
        for x in np.arange(min_x, max_x, box_length_m):
            for y in np.arange(min_y, max_y, box_length_m):
                grid_cells.append(Polygon([
                    (x, y),
                    (x + box_length_m, y),
                    (x + box_length_m, y + box_length_m),
                    (x, y + box_length_m)
                ]))

        # Create a GeoDataFrame for the grid
        full_grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=filtered_gdf.crs)

        # Perform a spatial join to count the number of points in each grid cell
        joined = gpd.sjoin(filtered_gdf, full_grid, how='left', predicate='within')
        counts = joined.groupby('index_right').agg({'count': 'sum'})

        # Add the counts to the grid GeoDataFrame
        full_grid['counts'] = counts
        full_grid[f'{DATE_NAME_TRANSLATOR[temporal_res]}_id'] = filtered_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}'].unique()[0]
        full_grid[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] = tstep

        full_grid['counts'].fillna(0, inplace=True)
        print(f"unique counts for {tstep}: {full_grid['counts'].unique()}")
        all_gdfs.append(full_grid)
    
    combined_gdf = pd.concat(all_gdfs)
    print('shape of combined GDF', combined_gdf.shape[0])
    
    # create lat long columns and save to CSV
    combined_gdf.set_index([f'{DATE_NAME_TRANSLATOR[temporal_res]}_id', 'geometry'], drop=True, inplace=True)
    
    combined_gdf['geometry_col'] = combined_gdf.index.get_level_values(1)
    combined_gdf = combined_gdf.set_geometry('geometry_col')
    
    centers = combined_gdf.geometry.centroid
    centers_latlong = centers.to_crs('EPSG:4326')
    combined_gdf['lat'] = centers_latlong.y
    combined_gdf['long'] = centers_latlong.x

    if study == 'asurv':
        file_id = f"{2011 - years_through_2011 + 1}_to_2011_{temporal_res}_{box_length_m}M"
        filename = f"asurv_{file_id}"
        os.makedirs(f"asurv/{file_id}", exist_ok=True)
        EDA_file = f'asurv/{file_id}/{filename}_EDA.txt'

    elif study == 'gps':
        file_id = f"2009_to_{2018 - years_cut_from_back}_{temporal_res}_{box_length_m}M"
        filename = f"gps_{file_id}"
        os.makedirs(f"gps/{file_id}", exist_ok=True)
        EDA_file = f'gps/{file_id}/{filename}_EDA.txt'

    # Make "README" of sorts for each dataset
    with open(EDA_file, 'w') as file:
        
        file.write(f'This file contains basic information about the dataset {filename}.\n\n')
        
        # basic info
        file.write(f"Is the square completed for time-space indices? {complete_idx_square}\n")
        file.write(f"Is the geometry column present? {keep_geometry_col}\n")

        # total number of boxes
        file.write(f"TOTAL # OF BOXES: {len(combined_gdf['geometry_col'].unique())}\n")

        print(combined_gdf.columns)
        # total number of timesteps + how many years
        file.write(f"TOTAL # OF TIMESTEPS: {len(combined_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'])}\n")

        # min date, max date
        file.write(f"MIN DATE: {np.sort(combined_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'])[0]}\n")
        file.write(f"MAX DATE: {np.sort(combined_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'])[-1]}\n")

        # total pct by counts
        file.write("DISTRIBUTION OF COUNTS, WHOLE DATASET: ")
        unique_vals, counts = np.unique(combined_gdf['counts'], return_counts=True)
        unique_vals = unique_vals.astype(int)
        counts = counts.astype(int)
        file.write(str({val: ct for val, ct in zip(unique_vals, counts)}))
        file.write("\n\n")

        # unique timesteps
        file.write('UNIQUE TIMESTEPS AND THEIR COUNTS:\n')
        
        for tstep in np.sort(combined_gdf[f"{DATE_NAME_TRANSLATOR[temporal_res]}_name"].unique()):
            filtered_gdf = combined_gdf[combined_gdf[f"{DATE_NAME_TRANSLATOR[temporal_res]}_name"] == tstep]
            unique_vals, counts = np.unique(filtered_gdf['counts'], return_counts=True)
            unique_vals = unique_vals.astype(int)
            counts = counts.astype(int)
            count_str = str({val: ct for val, ct in zip(unique_vals, counts)})
            file.write(f"{tstep}: {count_str}\n")
        
        # counts 

    if complete_idx_square:
        tstep_ids = combined_gdf.index.get_level_values(f'{DATE_NAME_TRANSLATOR[temporal_res]}_id').unique()
        geometries = combined_gdf.index.get_level_values('geometry').unique()
        full_index = pd.MultiIndex.from_product([tstep_ids, geometries], names=[f'{DATE_NAME_TRANSLATOR[temporal_res]}_id', 'geometry'])
        combined_gdf = combined_gdf.reindex(full_index)

    if keep_geometry_col:
        combined_gdf.drop(columns=['geometry_col']).to_csv(f'{study}/{file_id}/{filename}_final.csv')
        if save_shp_folder: gpd.GeoDataFrame(combined_gdf.droplevel(1)).to_file(f'{study}/{file_id}/{filename}_info')  # folder with GPD info
    else:
        combined_gdf.drop(columns=['geometry_col']).droplevel(1).to_csv(f'{study}/{file_id}/{filename}_final.csv')
        if save_shp_folder: gpd.GeoDataFrame(combined_gdf.droplevel(1)).to_file(f'{study}/{file_id}/{filename}_info')  # folder with GPD info
