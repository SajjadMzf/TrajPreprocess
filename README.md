# Pre-process Framework for Trajectory Datasets (highD dataset format)
This is a framework to pre-process trajectory datasets by grouping them based on Track ID or Frame. Here, we used the framework to convert custom trajectory datasets to highD dataset format.

The framework is tested on:

- NGSIM to highD format conversion.
- ExiD to highD format conversion.

## How to use for converting custom trajectory dataset to highD:
Similar to *configs/ngsim_preprocess.yaml*, create a config file for trajectory dataset of your choice. Write down the order of preprocess functions to be applied to your dataset (*ordered_preprocess_functions*). You may use preprocess function of your own or use the functions already create in framework or covered dataset. Note that there is specific input/output arguments for each defined preprocess function. Then you can run *PreprocessTraj.py" to perform the conversion/preprocessing.

## Some included preprocessing functions:
Following preprocessing functions are already implemented:

1. Trajectory smoothing using digital filters. (e.g., Savitzky–Golay filter)
2. Estimating Velocity, Acceleration from trajectory data.
3. Estimating lane boundries (where map data is not available).
4. Calculating Surrounding Vehicles IDs (e.g., Right/Left Preceding/Alongside/Following vehicles).
5. Visualising estimated states (e.g. position, velocity, acceleration)
6. Visualising trajectory data
7. Importing, exporting and updating Dataframes, track_groups, and frame_groups.
8. Exporting statics and meta data in highD format (few columns are populated in current version).
