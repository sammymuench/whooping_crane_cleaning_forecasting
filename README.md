Sammy Muench
10/26/24

PROJECT TITLE: Spatiotemporal Predictions of Aransas Whooping Crane Population using
flyover data and GPS data

The purpose of this directory is to generate spatiotemporal maps of the aransas whooping crane population, grouped by boxes with preset sizes and preset temporal binning.

Sources: 
- asurv: https://data.amerigeoss.org/nl/dataset/observations-of-whooping-cranes-during-winter-aerial-surveys-19502011
- gps: https://www.sciencebase.gov/catalog/item/5ea3071582cefae35a19349a

Directories & Files:
- raw-data: contains all raw data listed in sources
    - WHCR_locations_gps.csv: locations from the gps tracking data
    - asurv_1950_to_2011: directory with asurv data and info (DOESNT EXIST IN GITHUB REPO, MUST ADD)
        - WHCR_Aerial_Observations_1950_2011.txt: flyover data
        - WHCR_Aerial_Observations_1950-2011_metadata.pdf: pdf of metadata
- clean_asurv.py: file to get asurv data. Contains function that converts raw data to point-based event data (with cols x, y, count, timestep)
- clean_gps.py: file to clean gps data. Contains function that converts raw data to point-based event data (with cols x, y, count, timestep)
- points_in_boxes.py: contains function - used in both clean_asurv and clean_gps - that converts raw point data to binned box data

How to run program:
- First, add the data to raw-data. Add the folder from the asurv download link, and the file in the gps link (both in "sources" section) to the raw-data folder. You should have a file for gps and a folder containing the data and metadata for
asurv (aerial survey data).
- clean_asurv.py: at the top of __main__, you will find a list of parameters you can set and change. Change them in __main__ to your liking, and run the script.
    - Running this file with your specifications will output a directory inside "asurv" named by yuor specifications
        ({start_year}_to_{end_year}_{box_length_meters}: so it will look like 2009_to_2014_500M, or the like). 
        This directory will contain:
        - filename_EDA.txt: some basic but necessary exploratory analysis of the file. Useful information including:
            - Whether the square is completed (preset parameter)
            - If geometry column is present (preset parameter)
            - Total # of boxes
            - Total # of timesteps
            - Min date, max date
            - Distribution of counts over the whole dataset
            - Distribution of counts per-timestep
        - the dataset itself (.csv)
        - info directory with information about the geometry (shapefiles and other files related)
- clean_gps.py: same clean_asurv, there is a description of parameters you can manually set in __main__. Manually set them and then run the program with any compiler.


Description of output data:
- asurv: data from whenever specified in parameter until 2011. Data will be sparse relative to gps data. 
- gps: data from 2009 until whenever specified in parameter. Less sparse, as observations are GPS-based and
    are taken every 12 hours. Worth trying daily binning.

Known Issues: 
- If you run the code for a daily temporal resolution, the dates will look something like this: "2001_1_1_to_2001_1_2". This temporal bin will contain observations from 2001-1-2, and none from 2001_1_1.