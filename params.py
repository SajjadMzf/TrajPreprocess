DATA_FILES = ['./M40draft2.csv']
'''
DATA_FILES = ['./Data/M40_h06.csv',
                './Data/M40_h07.csv', 
                './Data/M40_h08.csv',
                './Data/M40_h09.csv', 
                './Data/M40_h10.csv', 
                './Data/M40_h11.csv', 
                './Data/M40_h12.csv', 
                './Data/M40_h13.csv', 
                './Data/M40_h14.csv', 
                './Data/M40_h15.csv', 
                './Data/M40_h16.csv', 
                './Data/M40_h17.csv', 
                './Data/M40_h18.csv', 
                './Data/M40_h19.csv'] #h10 for testing, rest for training
'''         
LANE_MARKINGS_FILE = './LaneMarkingsM40.csv'
XY_IMAGE_SAVE_DIR = './images_XY'
DS_IMAGE_SAVE_DIR = './images_DS'

MAX_PLOTTED_FRAME =200


X_SCALE = 4
Y_SCALE = 4
IMAGE_Y = 200 * Y_SCALE
IMAGE_X = 500 * X_SCALE
IMAGE_D = 400 * X_SCALE
IMAGE_S = 50 * Y_SCALE
D_BIAS = -600 * X_SCALE
S_BIAS = 0 * Y_SCALE
VEHICLE_WIDTH = 2 * Y_SCALE
VEHICLE_LENGTH = 5 * X_SCALE
X_BIAS = 0 * X_SCALE
Y_BIAS = 0 * Y_SCALE

LANE_ID = 'laneId'
FRAME = 'frame'
ID = 'id'
X = 'x'
Y = 'y'
S = 's'
D = 'd'
XVELOCITY = 'xVelocity'
YVELOCITY = 'yVelocity'

lon_1 = 'Inner_lon'
lat_1 = 'Inner_lat'
lon_2 = 'Middle_lon'
lat_2 = 'Middle_lat'
lon_3 = 'Outer_lon'
lat_3 = 'Outer_lat'
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