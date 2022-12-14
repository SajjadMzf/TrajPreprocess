# Data Preprocessing Framework for Trajectory Datasets (highD dataset format)
This is a framework for pre-processing trajectory datasets using frame and id grouping. Here, we used the framework to convert custom trajectory datasets (proprietary or public) to highD dataset format.

The framework already includes conversions among the following public datasets:

- NGSIM to highD format conversion.
- exiD to highD format conversion (soon).

## How to use for converting custom trajectory dataset to highD format:
Similar to *configs/ngsim_preprocess.yaml*, create a config file for the trajectory dataset of your choice. Write down the order of preprocessing functions to be applied to your dataset (*ordered_preprocess_functions* in config file). You may use preprocessing functions of your own or use the functions already included in the framework. Each preprocessing function can modify any of (1) config files, (2) dataframe objects (3) id_grouped data (4) frame_grouped data. Finally, you can run *PreprocessTraj.py" to perform the conversion/preprocessing.

## Some included preprocessing functions:
Following are some pre-implemented preprocessing functions of the framework:

- Trajectory smoothing using digital filters. (e.g., Savitzky–Golay filter)
- Estimating Velocity, and Acceleration from trajectory data.
- Estimating lane marking locations (where map data is not available).
- Calculating Surrounding Vehicles IDs (e.g., Right/Left Preceding/Alongside/Following vehicles).
- Visualising estimated states (e.g. position, velocity, acceleration)
- Visualising trajectory data
- Importing, exporting, and updating Dataframes, track_groups, and frame_groups.
- Exporting statics and metadata in highD format (few columns are populated in the current version).

## Reference:
1. highD dataset format: https://www.highd-dataset.com/format
2. NGSIM dataset: https://datahub.transportation.gov/stories/s/i5zb-xe34#trajectory-data
3. exiD dataset: https://www.exid-dataset.com/
