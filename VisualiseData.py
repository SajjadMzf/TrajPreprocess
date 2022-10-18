import numpy as np
import pandas as pd
import  os
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cv2

import params as p
from utils_functions import digital_filter
from PreprocessData import PreprocessData
class VisualiseData:
    def __init__(self, track_data_list, frame_data_list, lane_markings_xy, lane_markings, coordinate):
        self.export_dir = './visualisation/'
        self.plot_dir = './plots/'
        self.track_data_list = track_data_list
        self.frame_data_list = frame_data_list
        self.lane_markings_xy = lane_markings_xy
        self.lane_markings = lane_markings
        
        if coordinate != 'frenet':
            raise(ValueError("coordinate system not supported"))
        self.image_x = p.IMAGE_D
        self.image_y = p.IMAGE_S
        self.x_bias = p.D_BIAS
        self.y_bias = p.S_BIAS
        
        
        self.background_image = np.zeros((self.image_y, self.image_x,  3), dtype = np.uint8)
        for lane_marking in self.lane_markings:
            cv2.line(self.background_image, (0, int(lane_marking*p.Y_SCALE)),(self.image_x, int(lane_marking*p.Y_SCALE)), (0,255,0), thickness= p.LINE_THICKNESS)
        self.track_itr_list = [track[p.TRACK_ID][0] for track in self.track_data_list]
        self.frame_itr_list = [frame[p.FRAME][0] for frame in self.frame_data_list]

    
    def visualise_tracks(self, track_ids):
        for tv_id in track_ids:
            tv_itr = self.track_itr_list.index(tv_id)
            frames = self.track_data_list[tv_itr][p.FRAME]
            sv_ids = self.get_svs_ids(self.track_data_list[tv_itr])
            #images = []
            for fr_itr, frame in enumerate(frames):
                frame_itr = self.frame_itr_list.index(frame)
                frame_data = self.frame_data_list[frame_itr]
                image = np.copy(self.background_image)
                for track_itr, track_id in enumerate(frame_data[p.TRACK_ID]):
                    lane_id = int(frame_data[p.LANE_ID][track_itr])
                    if track_id == tv_id:
                        text = 'TV:{}'.format(lane_id)
                    elif track_id in sv_ids[fr_itr]:
                        text = '{}:{}:{}'.format(p.SV_IDS_ABBR[np.argwhere(sv_ids[fr_itr]==track_id)[0][0]],lane_id, int(track_id))
                    else:
                        text = 'NV:{}:{}'.format(lane_id,int(track_id))
                    image = self.plot_vehicle(image, 
                                        (frame_data[p.D_S][track_itr], frame_data[p.S_S][track_itr]), 
                                        (frame_data[p.WIDTH][track_itr],frame_data[p.HEIGHT][track_itr]),
                                        (0,255,0),
                                        text,
                                        (0,0,255)
                                        )
                cv2.imwrite(os.path.join(self.export_dir, 'TV{}_FR{}.png'.format(tv_id, frame)), image)

    def get_svs_ids(self, track_data):
        num_frames = len(track_data[p.FRAME])
        svs_ids = np.zeros((num_frames, 8))
        for itr in range(num_frames):
            for i in range(8):
                svs_ids[itr,i] = track_data[p.SV_IDs[i]][itr]
        return svs_ids


    def plot_vehicle(self, image, centre, dimension, v_color, text, t_color):
        top_left = (int((centre[0]-dimension[0]/2)*p.X_SCALE+ p.D_BIAS), int((centre[1]-dimension[1]/2)*p.Y_SCALE+ p.S_BIAS))
        bot_left = (int((centre[0]-dimension[0]/2)*p.X_SCALE+ p.D_BIAS), int((centre[1]+dimension[1]/2)*p.Y_SCALE+ p.S_BIAS))
        bot_right = (int((centre[0]+dimension[0]/2)*p.X_SCALE+ p.D_BIAS), int((centre[1]+dimension[1]/2)*p.Y_SCALE+ p.S_BIAS))
        image = cv2.rectangle(image,top_left, bot_right,color=v_color, thickness = -1)
        image = cv2.putText(image, text, bot_left, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color = t_color, thickness=1 )
        return image
    def plot_measurements(self, track_ids):
        for track_id in track_ids:
            track_itr = self.track_itr_list.index(track_id)
            track_data = self.track_data_list[track_itr]
            
            x = track_data[p.D]
            y = track_data[p.S]
            x_smooth = track_data[p.D_S]
            y_smooth = track_data[p.S_S]
            #x_smoother = digital_filter(x, [-21,14,39,54,59,54,39,14,-21],231, smoothing = True)
            #y_smoother = digital_filter(y, [-21,14,39,54,59,54,39,14,-21],231, smoothing = True)
            x_velocity = track_data[p.X_VELOCITY]
            y_velocity = track_data[p.Y_VELOCITY]
            
            x_acceleration = track_data[p.X_ACCELERATION]
            y_acceleration = track_data[p.Y_ACCELERATION]

            fr = np.arange(len(x))/10
            fig = plt.figure(figsize=(16,12))
            gs = gridspec.GridSpec(6,1)
            axes = []
            for i in range(6):
                axes.append(fig.add_subplot(gs[i]))
            axes[0].plot(fr,x, label = 'unfiltered')
            axes[0].plot(fr,x_smooth, label ='smoothed')
            #axes[0].plot(fr,x_smoother, label ='smoother')
            
            axes[0].set_ylabel('x(m)')
            axes[0].legend()
            axes[1].plot(fr,y, label = 'unfiltered')
            axes[1].plot(fr,y_smooth, label ='smoothed')
            #axes[1].plot(fr,y_smoother, label ='smoother')
            
            axes[1].set_ylabel('y(m)')
            axes[1].legend()
            axes[2].plot(fr,x_velocity)
            axes[2].set_ylabel('v_x(m)')
            axes[3].plot(fr,y_velocity)
            axes[3].set_ylabel('v_y(m)')
            axes[4].plot(fr,x_acceleration)
            axes[4].set_ylabel('a_x(m)')
            axes[5].plot(fr,y_acceleration)
            axes[5].set_ylabel('a_y(m)')

            
            for i in range(6):
                axes[i].grid(True)
                axes[i].set_xlabel('Time(s)')
            plt.savefig(os.path.join(self.plot_dir, '{}.png'.format(int(track_id))))    


if __name__ == '__main__':
    processed_data = [p.DATA_FILES[0]]
    preprocess = PreprocessData(processed_data, p.LANE_MARKINGS_FILE)
    preprocess.import_data('df')
    preprocess.import_data('frames')
    preprocess.import_data('tracks')
    #print(preprocess.lane_markings_s)
    #exit()
    
    visualiser = VisualiseData(preprocess.track_data_list[0], 
                            preprocess.frame_data_list[0],
                            preprocess.lane_markings_xy, 
                            preprocess.lane_markings_s,
                            coordinate = 'frenet')
    #visualiser.plot_measurements([2,3,4,5,6,7,8,9,10,11,12,13])
    visualiser.visualise_tracks([2,3,4,5,6,7,8,9,10,11,12,13])
