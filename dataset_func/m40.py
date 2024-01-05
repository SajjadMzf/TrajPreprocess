import numpy as np
import os, shutil
import xml.etree.ElementTree as ET
import params as p 
import pdb
import yaml
import utils.coordinate_functions as cf
import utils.visualise_functions as vf
import pickle
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import cv2
import random
import time
from pyproj import Proj
import pandas as pd
from matplotlib import gridspec

from dataset_func.ngsim import get_svs_ids
from utils.utils_functions import interpolate_polyline, parametrise_polyline
from utils.utils_functions import digital_filter




def visualise_measurements(configs,itr,  df_data, tracks_data, frames_data, vis_count =10):
   
    vis_cdir = os.path.join(os.path.join(p.VIS_DIR, configs['dataset']['name']), 'measurements')
    vis_cdir = os.path.join(vis_cdir, time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(vis_cdir):
        os.makedirs(vis_cdir)
    vis_count = min(vis_count, len(tracks_data))
    track_itrs = random.sample(range(len(tracks_data)),len(tracks_data))[:vis_count]
    fps = configs['dataset']['dataset_fps']
    for itr in track_itrs:
        x = tracks_data[itr][p.X]
        y = tracks_data[itr][p.Y]
        x_smooth = digital_filter(x, [-21,14,39,54,59,54,39,14,-21],231, smoothing = True)
        y_smooth = digital_filter(y, [-21,14,39,54,59,54,39,14,-21],231, smoothing = True)
        x_velocity = tracks_data[itr][p.X_VELOCITY]
        y_velocity = tracks_data[itr][p.Y_VELOCITY]
        x_velocity_df = digital_filter(x, np.array([-2,-1,0,1,2]), 10)*5
        y_velocity_df = digital_filter(y, np.array([-2,-1,0,1,2]), 10)*5
        x_acceleration = tracks_data[itr][p.X_ACCELERATION]
        y_acceleration = tracks_data[itr][p.Y_ACCELERATION]
        x_acceleration_df = digital_filter(x_velocity_df, np.array([-2,-1,0,1,2]), 10)*5
        y_acceleration_df = digital_filter(y_velocity_df, np.array([-2,-1,0,1,2]), 10)*5
        fr = np.arange(len(x))/fps
        fig = plt.figure(figsize=(18,12))
        gs = gridspec.GridSpec(2,1)
        axes = []
        for i in range(2):
            axes.append(fig.add_subplot(gs[i]))
        # axes[0].plot(fr,x, label = 'unfiltered')
        # axes[0].plot(fr,x_smooth, label ='smoothed')
        # #axes[0].plot(fr,x_smoother, label ='smoother')
        
        # axes[0].set_ylabel('x(m)')
        # axes[0].legend()
        # axes[1].plot(fr,y, label = 'unfiltered')
        # axes[1].plot(fr,y_smooth, label ='smoothed')
        # #axes[1].plot(fr,y_smoother, label ='smoother')
        #axes[1].set_ylabel('y(m)')
        #axes[1].legend()

        axes[0].plot(fr,x_velocity, label = 'Kalman')
        #axes[0].plot(fr,x_velocity_df, label = 'Digital Filter')
        axes[0].set_ylabel('v_x(m)')
        axes[0].legend()
        axes[1].plot(fr,y_velocity, label = 'Kalman')
        #axes[1].plot(fr,y_velocity_df, label = 'Digital Filter')
        axes[1].set_ylabel('v_y(m)')
        axes[1].legend()    
        # axes[2].plot(fr,x_acceleration, label = 'Kalman')
        # axes[2].plot(fr,x_acceleration_df, label = 'Digital Filter')
        # axes[2].set_ylabel('a_x(m)')
        # axes[2].legend()
        # axes[5].plot(fr,y_acceleration, label = 'Kalman')
        # axes[5].plot(fr,y_acceleration_df, label = 'Digital Filter')
        # axes[5].set_ylabel('a_y(m)')
        # axes[5].legend()

        
        for i in range(2):
            axes[i].grid(True)
            axes[i].set_xlabel('Time(s)')
        plt.savefig(os.path.join(vis_cdir, '{}_{}.png'.format(itr,int(tracks_data[itr][p.TRACK_ID][0]))))    
        plt.close(fig)

    return {'configs': None, 'df': None, 'tracks_data': None,'frames_data': None}



def filter_bboxes(configs, df_itr, df_data, tracks_data, frames_data):
    # fix ev bbox
    assert(tracks_data[0][p.TRACK_ID][0]==0)
    for itr, track_data in enumerate(tracks_data):
        if itr == 0:
            tracks_data[0][p.HEIGHT] = 2.5*np.ones_like(tracks_data[0][p.HEIGHT])
            tracks_data[0][p.WIDTH] = 5*np.ones_like(tracks_data[0][p.WIDTH])
        else:
            # take the median of the height and width
            tracks_data[itr][p.HEIGHT] = np.median(tracks_data[itr][p.HEIGHT])*np.ones_like(tracks_data[itr][p.HEIGHT])
            tracks_data[itr][p.WIDTH] = np.median(tracks_data[itr][p.WIDTH])*np.ones_like(tracks_data[itr][p.WIDTH])
    

    return {'configs': None,
            'df': None,
            'tracks_data': tracks_data,
            'frames_data': None}
def m40_preprocess(configs, df_itr, df_data, tracks_data, frames_data):
    
    # convert time to frame
    time = df_data[p.FRAME].values
    time = (time-time[0])*configs['dataset']['dataset_fps']
    # round to closest integer
    time = np.round(time)
    df_data[p.FRAME] = time
    # convert left hand driving to right hand driving
    df_data[p.Y] = -1*df_data[p.Y]
    df_data[p.Y_VELOCITY] = -df_data[p.Y_VELOCITY]
    df_data[p.Y_ACCELERATION] = -df_data[p.Y_ACCELERATION]
    return {'configs': None,
            'df': df_data,
            'tracks_data': None,
            'frames_data': None}

def extract_lms(configs, df_itr, df_data, tracks_data, frames_data):
    PLOT = False
    map_file = configs['dataset']['map_import_dir']
    # read npy file
    map_data = np.load(map_file, allow_pickle=True)
    [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5, x6, y6, merge_points] = map_data
    # left hand driving to right hand driving


    y1 = -y1
    y2 = -y2
    y3 = -y3
    y4 = -y4
    y5 = -y5
    y6 = -y6

    merge_b = merge_points[0]
    merge_e = merge_points[1]
    merge_into_b = merge_points[2]
    merge_into_e = merge_points[3]
    merge_lane_type = np.zeros_like(x2)
    merge_lane_type[merge_b:merge_e] = 2
    merge_into_lane_type = np.zeros_like(x3)
    merge_into_lane_type[merge_into_b:merge_into_e] = 3   
    lane1 = {'r': np.array([x1,y1]).T,
            'l': np.array([x2,y2]).T,
            'rt': np.zeros_like(x1),
            'lt': merge_lane_type}
    lane2 = {'r': np.array([x3,y3]).T,
            'l': np.array([x4,y4]).T,
            'rt': merge_into_lane_type,
            'lt': np.zeros_like(x4)}
    lane3 = {'r': np.array([x4,y4]).T,
            'l': np.array([x5,y5]).T,
            'rt': np.zeros_like(x4),
            'lt': np.zeros_like(x5)}
    lane4 = {'r': np.array([x5,y5]).T,
            'l': np.array([x6,y6]).T,
            'rt': np.zeros_like(x5),
            'lt': np.zeros_like(x6)}
    lms_cart = [lane4, lane3, lane2, lane1]
    merge_frenet_lm = lane1['l']
    main_frenet_lm = lane2['r']
    merge_point_on_merge_lane = merge_frenet_lm[merge_b]
    merge_point_on_main_lane = main_frenet_lm[merge_into_b]
    assert(np.all(merge_point_on_merge_lane == merge_point_on_main_lane) == True)    
    # assert merge point exist in main_frenet_lm
    main_matched_point_s = cf.cart2frenet(main_frenet_lm, merge_frenet_lm)[merge_into_b,0]

    lms_frenet = []
    for itr, lm in enumerate(lms_cart):
        # if merge lane
        if itr == (len(lms_cart)-1):
            l_points = cf.cart2frenet(lm['l'], merge_frenet_lm)
            r_points = cf.cart2frenet(lm['r'], merge_frenet_lm)
            merge_matched_point_s = l_points[merge_b,0]
            l_points[:,0] += main_matched_point_s - merge_matched_point_s
            r_points[:,0] += main_matched_point_s - merge_matched_point_s
        else:
            l_points = cf.cart2frenet(lm['l'], main_frenet_lm)
            r_points = cf.cart2frenet(lm['r'], main_frenet_lm)
        lms_frenet.append({'r': r_points,
                'l': l_points})
        
    # Extrpolate lane markings
    for itr, lane_nodes_f in enumerate(lms_frenet):
        lane_type = lms_cart[itr]['rt'][-1]
        s = lane_nodes_f['r'][:,0]
        d = lane_nodes_f['r'][:,1]
        s_ext = np.arange(s[-1]+10, s[-1]+300, 0.1)
        d_ext = np.ones_like(s_ext)*d[-1]
        sd_ext = np.stack((s_ext, d_ext), axis=  1)
        
        t_ext = np.ones_like(s_ext)*lane_type
        lms_frenet[itr]['r'] = np.append(lms_frenet[itr]['r'], sd_ext, axis = 0)
        lms_frenet[itr]['rt'] = np.append(lms_cart[itr]['rt'], t_ext, axis = 0)
        
        lane_type = lms_cart[itr]['lt'][-1]
        s = lane_nodes_f['l'][:,0]
        d = lane_nodes_f['l'][:,1]
        s_ext = np.arange(s[-1]+10, s[-1]+300, 0.1)
        d_ext = np.ones_like(s_ext)*d[-1]
        sd_ext = np.stack((s_ext, d_ext), axis=  1)
        
        t_ext = np.ones_like(s_ext)*lane_type
        lms_frenet[itr]['l'] = np.append(lms_frenet[itr]['l'], sd_ext, axis = 0)
        lms_frenet[itr]['lt'] = np.append(lms_cart[itr]['lt'], t_ext, axis = 0)
        

    lane_y_max = max([max(lane['l'][:,1]) for lane in lms_frenet])
    lane_y_min = min([min(lane['r'][:,1]) for lane in lms_frenet])
    lane_x_max = max([max(lane['l'][:,0]) for lane in lms_frenet])
    lane_x_min = min([min(lane['l'][:,0]) for lane in lms_frenet])
    image_width = lane_x_max - lane_x_min
    image_height = lane_y_max- lane_y_min
    # plot lines in lms_cart and lms_frenet in seperate plots
    if PLOT:
        plt.figure()
        for lane in lms_cart:
            plt.plot(lane['r'][:,0], lane['r'][:,1], '-ro')
            plt.plot(lane['l'][:,0], lane['l'][:,1], '-bo')
        plt.figure()
        for lane in lms_frenet:
            plt.plot(lane['r'][:,0], lane['r'][:,1], '-ro')
            plt.plot(lane['l'][:,0], lane['l'][:,1], '-bo')
        #plot merge origin lane and main origin lane
        plt.figure()
        plt.plot(merge_frenet_lm[:,0], merge_frenet_lm[:,1], '-ro', markersize=2)
        # plot with dot and lines

        plt.figure()
        plt.plot(main_frenet_lm[:,0], main_frenet_lm[:,1], '-bo', markersize=2)
        plt.show()
    #print(lane_markings_frenets)
    lane_marking_dict = {}
    lane_marking_dict['image_width'] = image_width
    lane_marking_dict['image_height'] = image_height
    lane_marking_dict['lane_nodes'] = lms_cart
    lane_marking_dict['lane_nodes_frenet'] = lms_frenet
    lane_marking_dict['merge_origin_lane'] = merge_frenet_lm
    lane_marking_dict['main_origin_lane'] = main_frenet_lm
    lane_marking_dict['merge2main_s_bias'] = main_matched_point_s - merge_matched_point_s
    lane_marking_dict['driving_dir'] = 2 #1: right to left in image plane, 2: left to right in image plane    
    with open(configs['dataset']['map_export_dir'], 'wb') as handle:
        pickle.dump(lane_marking_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    return {'configs': None,
            'df': None,
            'tracks_data': None,
            'frames_data': None}