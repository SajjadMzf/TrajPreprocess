import os
import pandas as pd
import numpy as np
import params as p
import matplotlib.pyplot as plt
import cv2
import math
from numpy.linalg import norm
class convert_format():
    def __init__(self, data_files, lane_markings_file, preprocess_df):
        self.data_dfs = []
        
        lane_markings = pd.read_csv(lane_markings_file)
        self.lane_markings = lane_markings.to_dict(orient='list')
        self.lane_markings_xy, self.lane_centers_xy = self.global2local_coor(self.lane_markings)
        self.lane_markings_ds, self.lane_centers_ds = self.lanes2frenet(self.lane_markings_xy, self.lane_centers_xy)    
        # stright lane markings
        self.lane_markings_s = np.mean(self.lane_markings_ds[:,:,1], axis=1)
        self.lane_centers_s = np.mean(self.lane_centers_ds[:,:,1], axis=1)
        
        self.vehicle_data_list = []
        self.frame_data_list = []
        for data_file in data_files:
        
            df, vehicle_data_list, frame_data_list= self.read_df(data_file, preprocess = preprocess_df) 
            self.vehicle_data_list.append(vehicle_data_list)
            self.frame_data_list.append(frame_data_list)
            self.data_dfs.append(df)

    def get_lane_id(self, vehicle_data_list):
        olv_c = 0
        ilv_c = 0
        for id, vehicle_data in enumerate(vehicle_data_list):
            total_frames = len(vehicle_data[p.X])
            lane_id = np.zeros((total_frames))
            for fr in range(total_frames):
                for i in range(3):
                    if vehicle_data[p.S][fr]<=self.lane_markings_s[i+1] and vehicle_data[p.S][fr]>=self.lane_markings_s[i]:
                        lane_id[fr] = i+1
                if vehicle_data[p.S][fr]<self.lane_markings_s[0]:
                    ilv_c +=1
                
                if vehicle_data[p.S][fr]>self.lane_markings_s[3]:
                    olv_c +=1
                    lane_id[fr] = 3
                
            vehicle_data_list[id][p.LANE_ID] = lane_id
        print('Outer lane violation counts (classified as lane 3): {}, Inner lane violation counts (classified as lane 0): {}'.format(olv_c, ilv_c))
        return vehicle_data_list
    def lanes2frenet(self, markings, centers):
        markings_ds = np.zeros_like(markings)
        centers_ds = np.zeros_like(centers)
        ref = markings[0]
        for i in range(markings.shape[0]):
            if i == 0:
                continue
            markings_ds[i], ref_frenet = self.cart2frenet(markings[i],ref)
        markings_ds[0] = ref_frenet
        for i in range(centers.shape[0]):
            centers_ds[i], _ = self.cart2frenet(centers[i],ref)
        return markings_ds, centers_ds
    def read_df(self, data_file, preprocess = True): 
        self.max_lane_x = np.amax(self.lane_markings_xy[:,:,0])-2
        self.max_lane_y = np.amax(self.lane_markings_xy[:,:,1])
        
        pr_data_file = data_file.split('.csv')[0] + '_processed' + '.csv'
        print('Preprocessing: ', data_file)
        df = pd.read_csv(data_file)
        print("################### Initial DF ########################")
        print('min: ',df.min())
        print('mean: ',df.mean())
        print('max: ',df.max())
        if preprocess == True:
            
            
            # convert to image coordinate
            df[p.Y] = df[p.Y].apply(lambda x: x*-1)
            
            # drop samples with x larger than max_lane_x and y less than min_lane_y
            df.drop(df[df[p.X]>self.max_lane_x].index, inplace=True)
            df.drop(df[df[p.Y]>self.max_lane_y].index, inplace=True)
            
            # convert to frenet frame
            vehicle_data_list = self.group_df(df, by = p.ID)
            ref = self.lane_markings_xy[0]
            for id, vehicle_data in enumerate(vehicle_data_list):
                num_frames = len(vehicle_data[p.X])
                traj = np.zeros((num_frames,2))
                traj[:,0] = vehicle_data[p.X]
                traj[:,1] = vehicle_data[p.Y]
                traj_frenet, _ = self.cart2frenet(traj,ref)
                vehicle_data_list[id][p.D] = traj_frenet[:,0]
                vehicle_data_list[id][p.S] = traj_frenet[:,1]
            
            '''
            for id in range(len(vehicle_data_list)):
                for k,v in vehicle_data_list[id].items():
                    if k == p.ID:
                        vehicle_data_list[id][k] *= np.ones_like(vehicle_data_list[id][p.X])
                    vehicle_data_list[id][k] = list(vehicle_data_list[id][k])
            '''
            vehicle_data_list = self.get_lane_id(vehicle_data_list)
            df = pd.concat(pd.DataFrame(vehicle_data) for vehicle_data in vehicle_data_list)
            df.to_csv(pr_data_file, index = False)
            #print(df.head())
            # ammend vehicle data and frame data lists
        else:
            print('Loading: ', pr_data_file)
            df = pd.read_csv(pr_data_file)
        
        print("################### Processed DF ########################")
        print('min: ',df.min())
        print('max: ',df.max())
        frame_data_list = self.group_df(df, by = p.FRAME)
        vehicle_data_list = self.group_df(df, by = p.ID)
        return df, vehicle_data_list, frame_data_list

    def cart2frenet(self, traj, ref):
        '''
        traj = np array of size [T,2]
        ref = np array of size [L,2]
        '''
        L = ref.shape[0]
        T = traj.shape[0]
        gamma = np.zeros((L))
        for i in range(L-1):
            gamma[i] = norm(ref[i+1]-ref[i])
        gamma[L-1] = gamma[L-2]
        gamma = np.cumsum(gamma)
        ref_frenet = np.zeros((L, 2))
        ref_frenet[:,0] = gamma
        traj_frenet = np.zeros((T, 2))
        for i in range(T):
            min2itr = np.argpartition(norm(ref-traj[i], axis=1), 3)[0:2]
            it = np.min(min2itr)
            it1 = np.max(min2itr)
            traj_frenet[i,0] =  gamma[it] + norm(np.dot(ref[it1]-ref[it], ref[it]-traj[i]))/norm(ref[it1]-ref[it])
            traj_frenet[i,1] = norm(np.cross(ref[it1]-ref[it], ref[it]-traj[i]))/norm(ref[it1]-ref[it])
        return traj_frenet, ref_frenet 
    
    def global2local_coor(self, lane_markings):
        lon1 = np.array(lane_markings[p.lon_1]) #+ p.X_BIAS
        lat1 = np.array(lane_markings[p.lat_1]) #+ p.Y_BIAS
        lon2 = np.array(lane_markings[p.lon_2]) #+ p.X_BIAS
        lat2 = np.array(lane_markings[p.lat_2]) #+ p.Y_BIAS
        lon3 = np.array(lane_markings[p.lon_3]) #+ p.X_BIAS
        lat3 = np.array(lane_markings[p.lat_3]) #+ p.Y_BIAS
        (x1, y1) = getXYpos(p.ORIGIN_LAT, p.ORIGIN_LON, lat1, lon1)
        (x2, y2) = getXYpos(p.ORIGIN_LAT, p.ORIGIN_LON, lat2, lon2)
        (x3, y3) = getXYpos(p.ORIGIN_LAT, p.ORIGIN_LON, lat3, lon3)
            
        lane_markings_xy = np.zeros((4,len(lon1),2))
        lane_center_xy = np.zeros((3,len(lon1),2))
        
        lane_center_xy[0,:,0] = x1 
        lane_center_xy[0,:,1] = y1 
        lane_center_xy[1,:,0] = x2 
        lane_center_xy[1,:,1] = y2 
        lane_center_xy[2,:,0] = x3 
        lane_center_xy[2,:,1] = y3
        
        lane_markings_xy[1,:,0] = x1 + (x2-x1)/2
        lane_markings_xy[1,:,1] = y1 + (y2-y1)/2
        lane_markings_xy[2,:,0] = x2 + (x3-x2)/2
        lane_markings_xy[2,:,1] = y2 + (y3-y2)/2
        
        lane_markings_xy[0,:,0] = x1 - (x2-x1)/2
        lane_markings_xy[0,:,1] = y1 - (y2-y1)/2
        lane_markings_xy[3,:,0] = x3 + (x3-x2)/2
        lane_markings_xy[3,:,1] = y3 + (y3-y2)/2
        
        return lane_markings_xy, lane_center_xy

    def group_df(self, df, by):
        if p.D in df.columns and p.S in df.columns:
            frenet = True
        else:
            frenet = False
        
        grouped = df.groupby([by], sort = True)
        current_group = 0
        groups = [None] * grouped.ngroups

        for group_id, rows in grouped:
            groups[current_group] = {
                p.ID: rows[p.ID].values if by!=p.ID else np.int64(group_id),
                p.FRAME: rows[p.FRAME].values if by!=p.FRAME else np.int64(group_id),
                p.X: rows[p.X].values,
                p.Y: rows[p.Y].values,
                #p.YVELOCITY: rows[p.YVELOCITY].values,
                #p.XVELOCITY: rows[p.XVELOCITY].values,
            }
            if frenet == True:
                groups[current_group][p.D] = rows[p.D].values
                groups[current_group][p.S] = rows[p.S].values
            if p.LANE_ID in df.columns:
                groups[current_group][p.LANE_ID] = rows[p.LANE_ID].values    
            current_group+= 1
        return groups

    def visualise(self, frame_data_list, coordinate):
        if coordinate == 'cart':
            X = p.X
            Y = p.Y
            lane_markings = self.lane_markings_xy
            lane_centers = self.lane_centers_xy
            IMAGE_SAVE_DIR = p.XY_IMAGE_SAVE_DIR
            IMAGE_X = p.IMAGE_X
            IMAGE_Y = p.IMAGE_Y
            X_BIAS = p.X_BIAS
            Y_BIAS = p.Y_BIAS
        elif coordinate == 'frenet':
            X = p.D
            Y = p.S
            lane_markings = self.lane_markings_ds
            lane_centers = self.lane_centers_ds
            IMAGE_SAVE_DIR = p.DS_IMAGE_SAVE_DIR
            IMAGE_X = p.IMAGE_D
            IMAGE_Y = p.IMAGE_S
            X_BIAS = p.D_BIAS
            Y_BIAS = p.S_BIAS
        else:
            raise(ValueError('coordinate not found!'))

        for frame, frame_data in enumerate(frame_data_list):
            if frame>p.MAX_PLOTTED_FRAME:
                break
            image = np.zeros((IMAGE_Y, IMAGE_X,  3), dtype = np.uint8)
            # Vehicle
            for itr in range(len(frame_data[p.ID])):
                x = int(frame_data[X][itr]*p.X_SCALE)+ X_BIAS
                y = int(frame_data[Y][itr]*p.Y_SCALE)+ Y_BIAS
                cv2.circle(image, (x, y), radius= 4, color= (255,255,255), thickness = -1 )
                #image[y:y + p.VEHICLE_WIDTH, x:x + p.VEHICLE_LENGTH] = 255     
            # Lane markings            
            x1 = lane_markings[0,:,0]*p.X_SCALE + X_BIAS
            y1 = lane_markings[0,:,1]*p.Y_SCALE + Y_BIAS
            
            x2 = lane_markings[1,:,0]*p.X_SCALE + X_BIAS
            y2 = lane_markings[1,:,1]*p.Y_SCALE + Y_BIAS
            
            x3 = lane_markings[2,:,0]*p.X_SCALE + X_BIAS
            y3 = lane_markings[2,:,1]*p.Y_SCALE + Y_BIAS
            
            x4 = lane_markings[3,:,0]*p.X_SCALE + X_BIAS
            y4 = lane_markings[3,:,1]*p.Y_SCALE + Y_BIAS
            

            cx1 = lane_centers[0,:,0]*p.X_SCALE + X_BIAS
            cy1 = lane_centers[0,:,1]*p.Y_SCALE + Y_BIAS

            cx2 = lane_centers[1,:,0]*p.X_SCALE + X_BIAS
            cy2 = lane_centers[1,:,1]*p.Y_SCALE + Y_BIAS

            cx3 = lane_centers[2,:,0]*p.X_SCALE + X_BIAS
            cy3 = lane_centers[2,:,1]*p.Y_SCALE + Y_BIAS

            for itr in range(len(x1)):
                if itr == 0:
                    continue
                cv2.line(image, (int(x1[itr-1]), int(y1[itr-1])),(int(x1[itr]), int(y1[itr])), (0,255,0), thickness= p.LINE_THICKNESS )
                cv2.line(image, (int(x2[itr-1]), int(y2[itr-1])),(int(x2[itr]), int(y2[itr])), (0,255,0), thickness= p.LINE_THICKNESS )
                cv2.line(image, (int(x3[itr-1]), int(y3[itr-1])),(int(x3[itr]), int(y3[itr])), (0,255,0), thickness= p.LINE_THICKNESS )
                cv2.line(image, (int(x4[itr-1]), int(y4[itr-1])),(int(x4[itr]), int(y4[itr])), (0,255,0), thickness= p.LINE_THICKNESS )
                
                cv2.line(image, (int(cx1[itr-1]), int(cy1[itr-1])),(int(cx1[itr]), int(cy1[itr])), (0,0,255), thickness= p.LINE_THICKNESS )
                cv2.line(image, (int(cx2[itr-1]), int(cy2[itr-1])),(int(cx2[itr]), int(cy2[itr])), (0,0,255), thickness= p.LINE_THICKNESS )
                cv2.line(image, (int(cx3[itr-1]), int(cy3[itr-1])),(int(cx3[itr]), int(cy3[itr])), (0,0,255), thickness= p.LINE_THICKNESS )
                
            # Merging Point
            #(merge_x, merge_y) = getXYpos(p.ORIGIN_LAT, p.ORIGIN_LON, p.MERGE_LAT, p.MERGE_LON)
            #merge_x = merge_x*p.X_SCALE + p.X_BIAS
            #merge_y = merge_y*p.Y_SCALE + p.Y_BIAS
            #cv2.circle(image, (int(merge_x), int(merge_y)), radius= 3, color= (0,0,255), thickness = -1 )
            cv2.imwrite(os.path.join(IMAGE_SAVE_DIR, '{}.png'.format(frame)), image)
        
        
    
    def checknFix(self):
        
        return 0

def asRadians(degrees):
    return degrees * math.pi / 180

def getXYpos(null_lat, null_lon, dp_lat, dp_lon):
    """ Calculates X and Y distances in meters.
    """
    deltaLatitude = dp_lat - null_lat
    deltaLongitude = dp_lon - null_lon
    latitudeCircumference = 40075160 * math.cos(asRadians(null_lat))
    resultX = deltaLongitude * latitudeCircumference / 360
    resultY = deltaLatitude * 40008000 / 360
    
    return (resultX, -1*resultY)

if __name__ =='__main__':
    converter = convert_format(p.DATA_FILES, p.LANE_MARKINGS_FILE, preprocess_df= False)
    print('Visualising...')
    #converter.visualise(converter.frame_data_list[0], coordinate='cart')
    #converter.visualise(converter.frame_data_list[0], coordinate='frenet')
    
    #converter.checknFix()