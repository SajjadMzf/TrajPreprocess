


VISUALISE_FRENET = True
#Visualisation parameters:
VISUALISATION_COUNT = 1
X_SCALE = 10
Y_SCALE = 10
FONT_SCALE = 0.25

measurement_dir = 'visualisations/measurements'
tracks_dir = 'visualisations/tracks'


DF_SAVE_DIR = 'Tracks'
TRACK_SAVE_DIR = 'Pickles'
FRAME_SAVE_DIR = 'Pickles'
META_SAVE_DIR = 'Metas'
STATICS_SAVE_DIR = 'Statics'

MAX_PLOTTED_FRAME =10000
LINE_THICKNESS = 1


# Meta
#LOWER_LANE_MARKINGS = "lowerLaneMarkings"
# Statics
#DRIVING_DIRECTION = "drivingDirection"
# Tracking
Y2LANE = 'y2lane'
FRAME = "frame"
TRACK_ID = "id"
X = "x"
Y = "y"
X_RAW = 'x_raw'
Y_RAW = 'y_raw'
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
WIDTH = "width"
HEIGHT = "height"
PRECEDING_ID = "precedingId"
FOLLOWING_ID = "followingId"
LEFT_PRECEDING_ID = "leftPrecedingId"
LEFT_ALONGSIDE_ID = "leftAlongsideId"
LEFT_FOLLOWING_ID = "leftFollowingId"
RIGHT_PRECEDING_ID = "rightPrecedingId"
RIGHT_ALONGSIDE_ID = "rightAlongsideId"
RIGHT_FOLLOWING_ID = "rightFollowingId"
LANE_ID = "laneId"
LANE_WIDTH = 'laneWidth'
DRIVING_DIR = 'drivingDir'
RV_IDs = [
    RIGHT_FOLLOWING_ID,
    RIGHT_ALONGSIDE_ID,
    RIGHT_PRECEDING_ID
]
LV_IDs = [
    LEFT_FOLLOWING_ID,
    LEFT_ALONGSIDE_ID,
    LEFT_PRECEDING_ID
]

SV_IDs = [
        PRECEDING_ID, 
        FOLLOWING_ID,
        LEFT_PRECEDING_ID, 
        LEFT_ALONGSIDE_ID,
        LEFT_FOLLOWING_ID, 
        RIGHT_PRECEDING_ID, 
        RIGHT_ALONGSIDE_ID, 
        RIGHT_FOLLOWING_ID
        ]
SV_IDS_ABBR = [
    'PV',
    'FV',
    'LPV',
    'LAV',
    'LFV',
    'RPV',
    'RAV',
    'RFV'
]

metas_columns = ['id','frameRate','locationId','speedLimit','month','weekDay','startTime',
                    'duration','totalDrivenDistance','totalDrivenTime','numVehicles','numCars','numTrucks','upperLaneMarkings','lowerLaneMarkings']
statics_columns = ['width','height','initialFrame','finalFrame','numFrames','class',
                    'traveledDistance','minXVelocity','maxXVelocity','meanXVelocity','minDHW','minTHW','minTTC','numLaneChanges'] #except 'id' and 'drivingDirection' 
