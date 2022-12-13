# Data Preprocessing Framework for Trajectory Datasets (highD dataset format)
This is a framework for pre-processing trajectory datasets using frame and id grouping. Here, we used the framework to convert custom trajectory datasets (proprietary or public) to highD dataset format.

The framework already includes conversions among the following public datasets:

- NGSIM to highD format conversion.
- exiD to highD format conversion (soon).

## How to use for converting custom trajectory dataset to highD format:
Similar to *configs/ngsim_preprocess.yaml*, create a config file for the trajectory dataset of your choice. Write down the order of preprocessing functions to be applied to your dataset (*ordered_preprocess_functions* in config file). You may use preprocessing functions of your own or use the functions already included in the framework. Each preprocessing function can modify any of (1) config files, (2) dataframe objects (3) id_grouped data (4) frame_grouped data. Finally, you can run *PreprocessTraj.py" to perform the conversion/preprocessing.

## Some included preprocessing functions:
Following are some pre-implemented preprocessing functions of the framework:

1. Trajectory smoothing using digital filters. (e.g., Savitzky–Golay filter)
2. Estimating Velocity, and Acceleration from trajectory data.
3. Estimating lane marking locations (where map data is not available).
4. Calculating Surrounding Vehicles IDs (e.g., Right/Left Preceding/Alongside/Following vehicles).
5. Visualising estimated states (e.g. position, velocity, acceleration)
6. Visualising trajectory data
7. Importing, exporting, and updating Dataframes, track_groups, and frame_groups.
8. Exporting statics and metadata in highD format (few columns are populated in the current version).

## Reference:
highD dataset format: https://www.highd-dataset.com/format
NGSIM dataset: https://datahub.transportation.gov/stories/s/i5zb-xe34#trajectory-data
exiD dataset: https://www.exid-dataset.com/
