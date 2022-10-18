import numpy as np
import pandas as pd
import  os
import pickle
import matplotlib.pyplot as plt

import params as p
from data_frame_functions import group_df, group2df
import coordinate_functions as cf
from utils_functions import digital_filter

class PostProcessData():
    def __init__(self, data_files, prediction_files, lane_markings_file):
        if p.DEBUG_FLAG:
            print('Debug mode: processing only 1 file')
            self.data_files = [data_files[0]]
            self.prediction_files = [prediction_files[0]]
        else:
            self.data_files = data_files
            self.prediction_files = prediction_files
        self.FPS = 10
        self.lane_markings = pd.read_csv(lane_markings_file)
        self.lane_markings = self.lane_markings.to_dict(orient='list')
        self.lane_markings_xy, self.lane_markings_ds = self.get_lane_markings()
        self.lane_markings_s = np.mean(self.lane_markings_ds[:,:,1], axis=0)
        
        self.prediction_load_dir = p.PREDICTION_LOAD_DIR
        self.prediction_data = []
        self.data_df_list = []
        self.track_data_list = []
        self.frame_data_list = []
        self.df_save_dir = p.DF_PREDICTION_SAVE_DIR
        self.track_save_dir = p.TRACK_SAVE_DIR
        self.frame_save_dir = p.FRAME_SAVE_DIR
        self.df_load_dir = p.DF_LOAD_DIR
        self.track_load_dir = p.TRACK_LOAD_DIR
        self.frame_load_dir = p.FRAME_LOAD_DIR

    def import_data(self, data_type, args = (None,)):
        if data_type == 'df':
            for file_itr, data_file in enumerate(self.data_files):
                print('Importing df file: {}.'.format(self.data_files[file_itr]))
                load_cdir = os.path.join(self.df_load_dir, data_file)
                df = pd.read_csv(load_cdir)
                self.data_df_list.append(df)
        
        elif data_type == 'tracks':
            self.track_data_list = []
            for file_itr, data_file in enumerate(self.data_files):
                track_data_file = data_file.split('.')[0]+ '_tracks.pickle'
                print('Importing track file: {}.'.format(track_data_file))
                load_cdir = os.path.join(self.track_load_dir, track_data_file)
                with open(load_cdir, 'rb') as handle:
                    self.track_data_list.append(pickle.load(handle))
            
        elif data_type == 'frames':
            self.frame_data_list = []
            for file_itr, data_file in enumerate(self.data_files):
                frame_data_file = data_file.split('.')[0]+ '_frames.pickle'
                print('Importing frame file: {}.'.format(frame_data_file))
                load_cdir = os.path.join(self.frame_load_dir, frame_data_file )
                with open(load_cdir, 'rb') as handle:
                    self.frame_data_list.append(pickle.load(handle))
        elif data_type == 'prediction':
            self.prediction_data = []
            for file_itr, prediction_file in enumerate(self.prediction_files):
                print('Importing prediction data : {}.'.format(prediction_file))
                load_cdir = os.path.join(self.prediction_load_dir, prediction_file)
                df = pd.read_csv(load_cdir)
                #print(df.head())
                self.prediction_data.append(df)

        else:
            raise(ValueError('Undefined data type'))    
    
    def update_df(self, source = 'track_data'):
        if source == 'track_data':
            self.data_df_list = []  
            self.frame_data_list = []
            for file_itr, track_data in enumerate(self.track_data_list):
                print('Update DF from source {} of file: {}'.format(source ,self.data_files[file_itr]))
                df = group2df(track_data)
                self.data_df_list.append(df)      
                self.frame_data_list.append(group_df(df, by = p.FRAME))
        elif source == 'frame_data':
            self.data_df_list = []  
            self.track_data_list = []
            for file_itr, frame_data in enumerate(self.frame_data_list):
                print('Update DF from source {} of file: {}'.format(source ,self.data_files[file_itr]))
                df = group2df(frame_data)
                self.data_df_list.append(df)      
                self.track_data_list.append(group_df(df, by = p.TRACK_ID))
        else:
            raise(ValueError('Unknown Source'))
    
    def update_track_frame_data_list(self, data_type='both'):
        
        if data_type == 'both':
            self.track_data_list = []
            self.frame_data_list = []
        elif data_type == 'tracks':
            self.track_data_list = []
        elif data_type == 'frames':
            self.frame_data_list = []
        else:
            raise(ValueError('undefined data type'))
        
        for df_itr in range(len(self.data_df_list)):
            print('Update Track and Frame data lists of file: {}'.format(self.data_files[df_itr]))
            if data_type == 'both':
                self.track_data_list.append(group_df(self.data_df_list[df_itr],by = p.TRACK_ID))
                self.frame_data_list.append(group_df(self.data_df_list[df_itr], by = p.FRAME))
            elif data_type == 'tracks':
                self.track_data_list.append(group_df(self.data_df_list[df_itr],by = p.TRACK_ID))
            elif data_type == 'frames':
                self.frame_data_list.append(group_df(self.data_df_list[df_itr], by = p.FRAME))
            else:
                raise(ValueError('undefined data type'))
    def convert2cart(self, ref):
        # extending ref for potential predictions that goes behind lane markings
        d_ref = ref[-1]-ref[-2]
        e_ref = np.zeros((500,2))
        e_ref[0] = ref[-1]
        for i in range(499):
            e_ref[i+1] = e_ref[i]+d_ref
        ref = np.concatenate((ref, e_ref), axis = 0)
        for df_itr, prediction_df in enumerate(self.prediction_data):
            print('Converting to cart coordinates : {}'.format(self.prediction_files[df_itr]))
            prediction_df.sort_values(by =[p.ID, p.FRAME], inplace = True)
            prediction_ids = sorted(prediction_df[p.ID].unique())
            self.data_df_list[df_itr].sort_values(by =[p.ID, p.FRAME], inplace = True)
            
            numpy_df = self.data_df_list[df_itr][[p.ID, p.FRAME]].to_numpy()
            prx_list = ['']* numpy_df.shape[0]
            pry_list = ['']* numpy_df.shape[0]
            itr = 0
            total_frame = 0
            total_frame_gt = 0
            cart_rmse = 0
            frenet_rmse = 0
            cart_gt_rmse = 0
            for prediction_id in prediction_ids:
                print('converting coordinates of vehicle id: {}/{}'.format(itr, len(prediction_ids)))
                itr+=1
                #if itr>10:
                #    break
                pr_df_slice = prediction_df[prediction_df[p.ID] == prediction_id]
                
                for fr_itr, frame in enumerate(pr_df_slice[p.FRAME]):
                    
                    pr_s = pr_df_slice[p.prS].values[fr_itr].split(';')
                    pr_d = pr_df_slice[p.prD].values[fr_itr].split(';')
                    frenet_traj = np.zeros((len(pr_s),2))
                    frenet_traj[:,0] = np.array(pr_d)
                    frenet_traj[:,1] = np.array(pr_s)
                    

                    cart_traj = cf.frenet2cart(frenet_traj, ref)
                    cart_traj[:,1] *= -1 
                    #print(cart_traj)
                    df_index = np.logical_and(numpy_df[:,0] == prediction_id,numpy_df[:,1] == frame)
                    
                    df_list_index = np.argmax(df_index)
                    #print(self.data_df_list[df_itr].iloc[list(np.arange(df_list_index,df_list_index+50))][['x','y']])
                    #print(frenet_traj)
                    #print(self.data_df_list[df_itr].iloc[list(np.arange(df_list_index,df_list_index+50))][[p.D_S,p.S_S]])
                    prx_list[df_list_index] = ';'.join('{:.4f}'.format(x) for x in cart_traj[:,0])
                    pry_list[df_list_index] = ';'.join('{:.4f}'.format(x) for x in cart_traj[:,1])
                    #prediction_array[df_index, 0] = cart_traj[:,0] 
                    #prediction_array[df_index, 1] = cart_traj[:,1]
                    #prediction_flag[df_index] = True
                    '''
                    cart_gt = self.data_df_list[df_itr].iloc[list(np.arange(df_list_index+1,df_list_index+51))][['x','y']].values
                    frenet_gt = self.data_df_list[df_itr].iloc[list(np.arange(df_list_index+1,df_list_index+51))][[p.D_S,p.S_S]].values
                    frenet_gt_obs = self.data_df_list[df_itr].iloc[list(np.arange(df_list_index+1,df_list_index+51))][[p.D,p.S]].values
                    cart_gt_obs_transformed = cf.frenet2cart(frenet_gt_obs, ref)
                    #cart_gt_obs = self.data_df_list[df_itr].iloc[list(np.arange(df_list_index-20,df_list_index))][['x','y']].values
                    if np.isnan(np.min(cart_gt_obs_transformed)) == False:
                        cart_gt_rmse += np.sum((cart_gt-cart_gt_obs_transformed)**2)
                        total_frame_gt += 50
                    
                    total_frame += 50
                    cart_traj[:,1] *= -1 
                    if np.isnan(np.min(cart_traj)) == False:
                        cart_rmse += np.sum((cart_traj-cart_gt)**2)
                    frenet_rmse += np.sum((frenet_traj-frenet_gt)**2)

            cart_gt_rmse = np.sqrt(cart_gt_rmse/total_frame_gt)
            cart_rmse = np.sqrt(cart_rmse/total_frame)
            frenet_rmse = np.sqrt(frenet_rmse/total_frame)
            print('RMSE cart:{}, frenet:{}, cart gt obs: {}'.format(cart_rmse, frenet_rmse, cart_gt_rmse))
'''
            self.data_df_list[df_itr][p.prX] = prx_list
            self.data_df_list[df_itr][p.prY] = pry_list
            
              
            
    def export_data(self, data_type, args= (None,)):
        if data_type == 'df':
            column_list = args[0]
            for file_itr, data_file in enumerate(self.data_files):
                print('Exporting DF file: {} with {} Tracks'.format(self.data_files[file_itr], len(self.track_data_list[file_itr])))
                save_cdir = os.path.join(self.df_save_dir, data_file)
                df = self.data_df_list[file_itr][column_list]
                df.sort_values(by=[p.ID, p.FRAME], inplace = True)
                #df = df.applymap(lambda x: ';'.join([str(i) for i in x]))
                df.to_csv(save_cdir, index = False)
        
        elif data_type == 'tracks':
            for file_itr, data_file in enumerate(self.data_files):
                tracks_data_file = data_file.split('.')[0]+ '_tracks.pickle'
                print('Exporting Tracks file: {} with {} Tracks'.format(tracks_data_file, len(self.track_data_list[file_itr])))
                save_cdir = os.path.join(self.track_save_dir, tracks_data_file)
                with open(save_cdir, 'wb') as handle:
                    pickle.dump(self.track_data_list[file_itr], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        elif data_type == 'frames':
            column_list = args[0]
            for file_itr, data_file in enumerate(self.data_files):
                frame_data_file = data_file.split('.')[0]+ '_frames.pickle'
                print('Exporting Frames file: {} with {} Frames'.format(frame_data_file, len(self.frame_data_list[file_itr])))
                save_cdir = os.path.join(self.frame_save_dir,  frame_data_file)
                with open(save_cdir, 'wb') as handle:
                    pickle.dump(self.frame_data_list[file_itr], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        else:
            raise(ValueError('Undefined data type'))
    def get_lane_markings(self):
        lane_marking_length = len(np.array(self.lane_markings[p.lon_1]))
        lane_markings_xy = np.zeros((lane_marking_length,4,2))

        lon1 = np.array(self.lane_markings[p.lon_1]) #+ p.X_BIAS
        lat1 = np.array(self.lane_markings[p.lat_1]) #+ p.Y_BIAS
        lon2 = np.array(self.lane_markings[p.lon_2]) #+ p.X_BIAS
        lat2 = np.array(self.lane_markings[p.lat_2]) #+ p.Y_BIAS
        lon3 = np.array(self.lane_markings[p.lon_3]) #+ p.X_BIAS
        lat3 = np.array(self.lane_markings[p.lat_3]) #+ p.Y_BIAS

        (x1, y1) = cf.longlat2xy((lon1, lat1), (p.ORIGIN_LON, p.ORIGIN_LAT))
        (x2, y2) = cf.longlat2xy((lon2, lat2), (p.ORIGIN_LON, p.ORIGIN_LAT))
        (x3, y3) = cf.longlat2xy((lon3, lat3), (p.ORIGIN_LON, p.ORIGIN_LAT))
        
        y1 *= -1
        y2 *= -1
        y3 *= -1

        lane_markings_xy[:,1,0] = x1 + (x2-x1)/2
        lane_markings_xy[:,1,1] = y1 + (y2-y1)/2
        lane_markings_xy[:,2,0] = x2 + (x3-x2)/2
        lane_markings_xy[:,2,1] = y2 + (y3-y2)/2
        
        lane_markings_xy[:,0,0] = x1 - (x2-x1)/2
        lane_markings_xy[:,0,1] = y1 - (y2-y1)/2
        lane_markings_xy[:,3,0] = x3 + (x3-x2)/2
        lane_markings_xy[:,3,1] = y3 + (y3-y2)/2

        lane_markings_ds = np.zeros_like(lane_markings_xy)

        for i in range(lane_markings_xy.shape[1]):
            if i == 0:
                continue
            lane_markings_ds[:,i], ref_frenet = cf.cart2frenet(lane_markings_xy[:,i],lane_markings_xy[:,0])
        lane_markings_ds[:,0] = ref_frenet

        return lane_markings_xy, lane_markings_ds
 
if __name__ == '__main__':
    column_list = [p.FRAME, p.TRACK_ID, p.X, p.Y, p.prX, p.prY]
    postprocessor = PostProcessData(['M40_h10.csv'],['M40_h10_Prediction.csv'], p.LANE_MARKINGS_FILE)
    postprocessor.import_data('df')
    postprocessor.import_data('tracks')
    postprocessor.import_data('prediction')
    postprocessor.convert2cart(postprocessor.lane_markings_xy[:,0])
    #postprocessor.update_df(source='track_data')
    postprocessor.export_data('df', args = (column_list,))