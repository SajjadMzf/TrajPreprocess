# Data Preprocessing Framework for Trajectory Datasets


## :gear: Installation
You may create a conda environment for this project using:
```shell
conda env create -f environment.yml
```
or create a docker container based on instructions here: (https://github.com/SajjadMzf/TPL)[https://github.com/SajjadMzf/TPL]
## :wave: Intro
This is a framework for pre-processing trajectory datasets. The framework works based on preprocess functions defined  on grouped data based on track_id or frame. Following are some pre-implemented preprocessing functions of the framework:

- Coordinate Conversion (Frenet to Cartesian, Cartesian to Frenet).
- Trajectory smoothing using digital filters. (e.g., Savitzky–Golay filter)
- Estimating Velocity, and Acceleration from trajectory data.
- Estimating lane marking locations (where map data is not available).
- Calculating Surrounding Vehicles IDs (e.g., Right/Left Preceding/Alongside/Following vehicles).
- Visualising estimated states (e.g. position, velocity, acceleration)
- Visualising trajectory data
- Importing, exporting, and updating Dataframes, track_groups, and frame_groups.
- Exporting statics and metadata in highD format (few columns are populated in the current version).



## :rocket: How to use
### 1. Prepare your dataset
As a bare mininum, a trajectory dataset, saved as .csv file, is expected to have following columns:
| Frame Number | TRACK ID | X Coordinate | Y Coordinate

### 2. Write a config file:
Similar to *configs/ngsim_preprocess.yaml*, create a config file for the trajectory dataset of your choice. 

| Name                         | Type       | Description                                                                                                                                                                                                                        |
|------------------------------|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset                      | dictionary | Contains some common parameters for all datasets (e.g., name, description,  import_dir, export_dir, dataset_fps, desired_fps). You may add other parameters specific to your dataset here.                                         |
| ordered_preprocess_functions | list       | ordered list of preprocess functions. Each line has the following format: - ["FUNCTION_NAME", "TYPE"] where "TYPE" is "all" or "one", depending on if the function is written for all  csv files or one csv file of your dataset.  |
| columns                      | dictionary | Contains columns of the output dataset in following format: "COLUMN_IN_OUTPUT": "MATCHING_COLUMN_IN_INPUT" or None                                                                                                                 |
### 3. Write preprocess functions
A preprocess function should have following default arguments:
| Argument    | Description                                                            |
|-------------|------------------------------------------------------------------------|
| configs     | Contains configs dictionary of your dataset                            |
| df_itr      | itr of dataframe or None if function type is "all".                    |
| df_data     | dataframe data (list of data for "all" function type).                 |
| tracks_data | grouped data based on track_id (list of data for "all" function type). |
| frames_data | grouped_data based on frame. (list of data for "all" fucntion type).   |
and should return a dictionary with updated "configs", "df_data", "trackes_data", and "frames_data".

Example: A function to relocate the tracking point of vehicles from front bumper centre to centre of vehicle
```shell
def relocate_tracking_point(configs,itr, df_data, tracks_data = None, frames_data = None):
    df_data[p.X] = df_data[p.X]- df_data[p.WIDTH]/2
    return {'configs': None, 'df': df_data, 'tracks_data': None,'frames_data': None}
```

The framework comes with some pre-defined common preproces functions in PreprocessTraj.py and some dataset specific functions in dataset_func folder.
### 4. Run!
Run the framework using:
```shell
python3 PreprocessTraj.py CONFIG_FILE_DIR
```

## :books: References:
1. highD dataset format: https://www.highd-dataset.com/format
2. NGSIM dataset: https://datahub.transportation.gov/stories/s/i5zb-xe34#trajectory-data
3. exiD dataset: https://www.exid-dataset.com/
