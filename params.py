DEBUG_FLAG = False
#Visualisation parameters:
X_SCALE = 4
Y_SCALE = 4
IMAGE_Y = 200 * Y_SCALE
IMAGE_X = 500 * X_SCALE
IMAGE_D = 400 * X_SCALE
IMAGE_S = 50 * Y_SCALE
D_BIAS = -600 * X_SCALE
S_BIAS = 0 * Y_SCALE
X_BIAS = 0 * X_SCALE
Y_BIAS = 0 * Y_SCALE

#DATA_FILES = ['../../Dataset/Autoplex/Raw/M40draft2.csv']

DATA_FILES = ['M40_h06.csv',
                'M40_h07.csv', 
                'M40_h08.csv',
                'M40_h09.csv', 
                'M40_h10.csv', 
                'M40_h11.csv', 
                'M40_h12.csv', 
                'M40_h13.csv', 
                'M40_h14.csv', 
                'M40_h15.csv', 
                'M40_h16.csv', 
                'M40_h17.csv', 
                'M40_h18.csv', 
                'M40_h19.csv'] #h10 for testing, rest for training

DF_LOAD_DIR = '../../Dataset/Autoplex/Raw/' #Change to Raws for preprocessing from raw data
TRACK_LOAD_DIR = '../../Dataset/Autoplex/Pickles'
FRAME_LOAD_DIR = '../../Dataset/Autoplex/Pickles'

SAVE_DIR = '../../Dataset/Autoplex/'
DF_SAVE_DIR = '../../Dataset/Autoplex/Tracks'
TRACK_SAVE_DIR = '../../Dataset/Autoplex/Pickles'
FRAME_SAVE_DIR = '../../Dataset/Autoplex/Pickles'


LANE_MARKINGS_FILE = './LaneMarkingsM40.csv'
XY_IMAGE_SAVE_DIR = './images_XY'
DS_IMAGE_SAVE_DIR = './images_DS'

MAX_PLOTTED_FRAME =10000

PROCESSED_DATASET_DIR = 'Autoplex_CPM'
META_DIR = 'Metas'
STATIC_DIR = 'Statics'
TRACK_DIR = 'Tracks'
PICKLE_DIR = 'Pickles'


AVG_WIDTH = 2
AVG_LENGTH = 5



LANE_ID = 'laneId'
FRAME = 'frame'
ID = 'id'
X = 'x'
Y = 'y'
S = 's'
D = 'd'
S_S = 's_smooth'
D_S = 'd_smooth'
X_VELOCITY = 'xVelocity'
Y_VELOCITY = 'yVelocity'
YAW = 'yaw'
lon_1 = 'Inner_lon'
lat_1 = 'Inner_lat'
lon_2 = 'Middle_lon'
lat_2 = 'Middle_lat'
lon_3 = 'Outer_lon'
lat_3 = 'Outer_lat'

LM = {
    lon_1:[0,0],
    lat_1:[0,1],
    lon_2:[1,0],
    lat_2:[1,1],
    lon_3:[2,0],
    lat_3:[2,1],
}


LINE_THICKNESS = 1

ORIGIN_LON = -1.6101445
ORIGIN_LAT = 52.2590840

MERGE_LON = -1.607064506925382 
MERGE_LAT = 52.258202079012321


#TTLC
MAX_TTLC = 5.2
PTTRLC = "PredictedTimeToRightLaneChange"
PTTLLC = "PredictedTimeToLeftLaneChange"
TTRLC = "TimeToRightLaneChange"
TTLLC = "TimeToLeftLaneChange"
FPS = 10


# Meta
#LOWER_LANE_MARKINGS = "lowerLaneMarkings"
# Statics
#DRIVING_DIRECTION = "drivingDirection"
# Tracking
FRAME = "frame"
TRACK_ID = "id"
X = "x"
Y = "y"
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


column_list = [FRAME, TRACK_ID, X, Y, S, D, S_S, D_S, WIDTH, HEIGHT, 
                X_VELOCITY, Y_VELOCITY, X_ACCELERATION, Y_ACCELERATION,
                PRECEDING_ID, FOLLOWING_ID, LEFT_PRECEDING_ID, LEFT_ALONGSIDE_ID, LEFT_FOLLOWING_ID,
                RIGHT_PRECEDING_ID, RIGHT_ALONGSIDE_ID, RIGHT_FOLLOWING_ID, LANE_ID ]