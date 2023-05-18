DEBUG_MODE = False

DELETE_PREV_VIS = False
VISUALISE_FRENET = True
#Visualisation parameters:
VISUALISATION_COUNT = 5
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

S = "s"
D = "d"
S_VELOCITY = "sVelocity"
D_VELOCITY = "dVelocity"
S_ACCELERATION = "sAcceleration"
D_ACCELERATION = "dAcceleration"

WIDTH = "width"
HEIGHT = "height"
HEADING = 'heading'
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

# NEW SVs MMnTP:
RIGHT_VEHICLE1_ID = 'rightVehicle1Id'
RIGHT_VEHICLE2_ID = 'rightVehicle2Id'
RIGHT_VEHICLE3_ID = 'rightVehicle3Id'
LEFT_VEHICLE1_ID = 'leftVehicle1Id'
LEFT_VEHICLE2_ID = 'leftVehicle2Id'
LEFT_VEHICLE3_ID = 'leftVehicle3Id'

RV_IDs = [
    RIGHT_VEHICLE1_ID,
    RIGHT_VEHICLE2_ID,
    RIGHT_VEHICLE3_ID
]
LV_IDs = [
    LEFT_VEHICLE1_ID,
    LEFT_VEHICLE2_ID,
    LEFT_VEHICLE3_ID
]

# NEW SVs POVL:
LEFT_CLOSE_PRECEDING_ID= 'leftClosePrecedingId'
LEFT_FAR_PRECEDING_ID= 'leftFarPrecedingId'
LEFT_CLOSE_FOLLOWING_ID= 'leftCloseFollowingId'
LEFT_FAR_FOLLOWING_ID= 'leftFarFollowingId'
RIGHT_CLOSE_PRECEDING_ID= 'rightClosePrecedingId'
RIGHT_FAR_PRECEDING_ID= 'rightFarPrecedingId'
RIGHT_CLOSE_FOLLOWING_ID= 'rightCloseFollowingId'
RIGHT_FAR_FOLLOWING_ID= 'rightFarFollowingId'

RP_IDs = [
    RIGHT_CLOSE_PRECEDING_ID,
    RIGHT_FAR_PRECEDING_ID
]

RF_IDs = [
    RIGHT_CLOSE_FOLLOWING_ID,
    RIGHT_FAR_FOLLOWING_ID
]

LP_IDs = [
    LEFT_CLOSE_PRECEDING_ID,
    LEFT_FAR_PRECEDING_ID
]

LF_IDs = [
    LEFT_CLOSE_FOLLOWING_ID,
    LEFT_FAR_FOLLOWING_ID
]

#modify this for visualisation of other SV format
#SVs for POVL
SV_IDs = [PRECEDING_ID,
            FOLLOWING_ID,
            RIGHT_CLOSE_PRECEDING_ID,
            RIGHT_CLOSE_FOLLOWING_ID,
            RIGHT_FAR_FOLLOWING_ID,
            RIGHT_FAR_PRECEDING_ID,
            LEFT_CLOSE_PRECEDING_ID,
            LEFT_CLOSE_FOLLOWING_ID,
            LEFT_FAR_FOLLOWING_ID,
            LEFT_FAR_PRECEDING_ID
            ]
SV_IDS_ABBR = [
    'pv',
    'fv',
    'rcp',
    'rcf',
    'rff',
    'rfp',
    'lcp',
    'lcf',
    'lff',
    'lfp',
]
metas_columns = ['id','frameRate','locationId','speedLimit','month','weekDay','startTime',
                    'duration','totalDrivenDistance','totalDrivenTime','numVehicles','numCars','numTrucks','upperLaneMarkings','lowerLaneMarkings']
statics_columns = ['width','height','initialFrame','finalFrame','numFrames','class',
                    'traveledDistance','minXVelocity','maxXVelocity','meanXVelocity','minDHW','minTHW','minTTC','numLaneChanges'] #except 'id' and 'drivingDirection' 
