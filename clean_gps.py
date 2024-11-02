import pandas as pd
from pandas.tseries.offsets import DateOffset
from shapely.geometry import Polygon
import numpy as np
import geopandas as gpd
import os


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


if __name__ == '__main__':
    
    # BEGIN PARAMETERS
    box_length_m = 250
    temporal_res = 'weekly'
    years_cut_from_back = 4
    keep_geometry_col = False
    # END PARAMETERS

    # set gps gdf and CRS
    gps_gdf = pd.read_csv('raw-data/WHCR_locations_gps.csv')
    gps_gdf = gpd.GeoDataFrame(gps_gdf, geometry=gpd.points_from_xy(gps_gdf.Long, gps_gdf.Lat), crs='EPSG:4326')
    gps_gdf = gps_gdf.set_crs('EPSG:4326', allow_override=True)
    gps_gdf = gps_gdf.to_crs('EPSG:26914')
    gps_gdf['Date'] = pd.to_datetime(gps_gdf['Date'])
    gps_gdf['Year'] = gps_gdf['Date'].dt.year
    valid_years = gps_gdf['Year'].unique()[:len(gps_gdf['Year'].unique()) - years_cut_from_back]
    gps_gdf = gps_gdf[gps_gdf['Year'].isin(valid_years)]

    aransas_df = pd.read_csv('raw-data/asurv_1950_to_2011/WHCR_Aerial_Observations_1950_2011.txt', encoding='latin1', sep='\t')
    aransas_gdf = gpd.GeoDataFrame(aransas_df, geometry=gpd.points_from_xy(aransas_df.X, aransas_df.Y), crs='EPSG:26914')

    # Set spacing and generate the grid
    bbox = aransas_gdf.total_bounds
    box_length_degrees = box_length_m / METERS_PER_DEGREE
    grid_gdf = generate_grid(bbox, box_length_degrees, crs=aransas_gdf.crs)

    all_dates = pd.date_range(start=gps_gdf['Date'].min() - DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), end=gps_gdf['Date'].max() + DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), freq=DATE_RANGE_TRANSLATOR[temporal_res])
    gps_gdf[DATE_NAME_TRANSLATOR[temporal_res]] = np.searchsorted(all_dates, gps_gdf['Date'])  
    
    # add names for weeks for data clarity
    bin_names = {i + 1: f'{all_dates[i].date()}_to_{all_dates[i + 1].date()}' for i in range(len(all_dates) - 1)}
    gps_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] = gps_gdf[DATE_NAME_TRANSLATOR[temporal_res]].map(bin_names)

    # match the GPS data to only fit in the aransas boxes
    boxes_in_aransas = gpd.sjoin(gps_gdf, grid_gdf, how='left', predicate='within')
    boxes_only_aransas = boxes_in_aransas[~boxes_in_aransas['index_right'].isna()]

    gps_all_gdfs = []

    for tstep in np.sort(gps_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'].unique()):
        
        filtered_gdf = gps_gdf[gps_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] == tstep]
        
        # Calculate the bounding box of the data
        min_x, min_y, max_x, max_y = boxes_only_aransas.total_bounds

        # Create a 1x1 km grid
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
        # Changed 'op' to 'predicate' for compatibility with newer geopandas versions
        joined = gpd.sjoin(filtered_gdf, full_grid, how='left', predicate='within')
        counts = joined.groupby('index_right').size()

        # Add the counts to the grid GeoDataFrame
        full_grid['counts'] = counts
        full_grid[f'{DATE_NAME_TRANSLATOR[temporal_res]}_id'] = filtered_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}'].unique()[0]
        full_grid[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] = tstep
        full_grid['counts'].fillna(0, inplace=True)
        print(f"unique counts for {tstep}: {full_grid['counts'].unique()}")
        gps_all_gdfs.append(full_grid)

    
    combined_gdf = pd.concat(gps_all_gdfs)
    print('shape of combined GDF', combined_gdf.shape[0])
    
    # create lat long columns and save to CSV
    combined_gdf.set_index([f'{DATE_NAME_TRANSLATOR[temporal_res]}_id', 'geometry'], drop=True, inplace=True)
    
    combined_gdf['geometry_col'] = combined_gdf.index.get_level_values(1)
    combined_gdf = combined_gdf.set_geometry('geometry_col')
    
    centers = combined_gdf.geometry.centroid
    centers_latlong = centers.to_crs('EPSG:4326')
    combined_gdf['lat'] = centers_latlong.y
    combined_gdf['long'] = centers_latlong.x


    file_id = f"2009_to_{2018 - years_cut_from_back}_{temporal_res}_{box_length_m}M"
    filename = f"gps_{file_id}"
    os.makedirs(f"gps/{file_id}", exist_ok=True)
    # Make "README" of sorts for each dataset
    with open(f'gps/{file_id}/{filename}_EDA.txt', 'w') as file:
        
        file.write(f'This file contains basic information about the dataset {filename}.\n\n')
        
        # basic info
        file.write(f"Is the square completed for time-space indices? True\n")
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

    if keep_geometry_col:
        combined_gdf.drop(columns=['geometry_col']).to_csv(f'gps/{file_id}/{filename}.csv')
        gpd.GeoDataFrame(combined_gdf.droplevel(1)).to_file(f'gps/{file_id}/{filename}_info')  # folder with GPD info
    else:
        combined_gdf.drop(columns=['geometry_col']).droplevel(1).to_csv(f'gps/{file_id}/{filename}.csv')
        gpd.GeoDataFrame(combined_gdf.droplevel(1)).to_file(f'gps/{file_id}/{filename}_info')  # folder with GPD info

        