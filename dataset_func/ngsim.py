import params as p 
import random
from utils.utils_functions import digital_filter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os, shutil
import cv2
import pdb
import time
import pickle
def relocate_tracking_point(configs,itr, df_data, tracks_data = None, frames_data = None):
    df_data[p.X] = df_data[p.X]- df_data[p.WIDTH]/2
    return {'configs': None, 'df': df_data, 'tracks_data': None,'frames_data': None}

def traj_smoothing(configs,itr,  df_data, tracks_data, frames_data):
    for itr, track_data in enumerate(tracks_data):
        tracks_data[itr][p.X_RAW] = track_data[p.X]
        tracks_data[itr][p.Y_RAW] = track_data[p.Y]
        tracks_data[itr][p.X] = track_data[p.X]#digital_filter(track_data[p.X], [-21,14,39,54,59,54,39,14,-21],231, smoothing = True)
        tracks_data[itr][p.Y] = track_data[p.Y]#digital_filter(track_data[p.Y], [-21,14,39,54,59,54,39,14,-21],231, smoothing = True)
    
    return {'configs': None, 'df': None, 'tracks_data': tracks_data,'frames_data': None}

def calc_vel_acc(configs,df_itr,  df_data, tracks_data, frames_data, frenet = True):
    if frenet:
        for itr, track_data in enumerate(tracks_data):
            x_velo = digital_filter(track_data[p.S], np.array([-2,-1,0,1,2]), 10)*configs['dataset']['dataset_fps']
            y_velo = digital_filter(track_data[p.D], np.array([-2,-1,0,1,2]), 10)*configs['dataset']['dataset_fps']
            x_acc = digital_filter(x_velo, np.array([-2,-1,0,1,2]), 10)*configs['dataset']['dataset_fps']
            y_acc = digital_filter(y_velo, np.array([-2,-1,0,1,2]), 10)*configs['dataset']['dataset_fps']
                    
            tracks_data[itr][p.S_VELOCITY] = x_velo
            tracks_data[itr][p.D_VELOCITY] = y_velo
            tracks_data[itr][p.S_ACCELERATION] = x_acc
            tracks_data[itr][p.D_ACCELERATION] = y_acc

    else:
        for itr, track_data in enumerate(tracks_data):
            x_velo = digital_filter(track_data[p.X], np.array([-2,-1,0,1,2]), 10)*configs['dataset']['dataset_fps']
            y_velo = digital_filter(track_data[p.Y], np.array([-2,-1,0,1,2]), 10)*configs['dataset']['dataset_fps']
            x_acc = digital_filter(x_velo, np.array([-2,-1,0,1,2]), 10)*configs['dataset']['dataset_fps']
            y_acc = digital_filter(y_velo, np.array([-2,-1,0,1,2]), 10)*configs['dataset']['dataset_fps']
                    
            tracks_data[itr][p.X_VELOCITY] = x_velo
            tracks_data[itr][p.Y_VELOCITY] = y_velo
            tracks_data[itr][p.X_ACCELERATION] = x_acc
            tracks_data[itr][p.Y_ACCELERATION] = y_acc

    return {'configs': None, 'df': None, 'tracks_data': tracks_data,'frames_data': None}

def visualise_measurements(configs,itr,  df_data, tracks_data, frames_data, vis_count =10):
   
    vis_cdir = os.path.join(os.path.join(p.VIS_DIR, configs['dataset']['name']), 'measurements')
    vis_cdir = os.path.join(vis_cdir, time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(vis_cdir):
        os.makedirs(vis_cdir)
    vis_count = min(vis_count, len(tracks_data))
    track_itrs = random.sample(range(len(tracks_data)),len(tracks_data))[:vis_count]
    fps = configs['dataset']['dataset_fps']
    for itr in track_itrs:
        x = tracks_data[itr][p.X_RAW]
        y = tracks_data[itr][p.Y_RAW]
        x_smooth = tracks_data[itr][p.X]
        y_smooth = tracks_data[itr][p.Y]
        x_velocity = tracks_data[itr][p.X_VELOCITY]
        y_velocity = tracks_data[itr][p.Y_VELOCITY]
        x_acceleration = tracks_data[itr][p.X_ACCELERATION]
        y_acceleration = tracks_data[itr][p.Y_ACCELERATION]
        fr = np.arange(len(x))/fps
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
        plt.savefig(os.path.join(vis_cdir, '{}_{}.png'.format(itr,int(tracks_data[itr][p.TRACK_ID][0]))))    
        plt.close(fig)

    return {'configs': None, 'df': None, 'tracks_data': None,'frames_data': None}

def unit_convertion(df):
    return 0



def estimate_lane_markings(configs,df_itr,  df_data, tracks_data, frames_data):
    # estimate lane marking based on given lane ids and y values
    #lane_markings
    max_lane = max([int(df_data_[p.LANE_ID].max()) for df_data_ in df_data])
    lane_markings = np.zeros((max_lane+2))
    # list of average y per lane for each df
    average_y = []
    # list of count of data per lane for each df
    count_y = []
    average_y_i = np.zeros((max_lane))
    count_y_i = np.zeros((max_lane))
    for df_data_ in df_data:
        for lane_id in range(1, max_lane+1):
            average_y_i[lane_id-1] = np.mean(df_data_[df_data_[p.LANE_ID]==lane_id][p.Y].to_numpy())
            count_y_i[lane_id-1] = df_data_[df_data_[p.LANE_ID]==lane_id][p.Y].to_numpy().shape[0]
        average_y.append(average_y_i)
        count_y.append(count_y_i)
    
    # average y per lane for all dfs    
    average_y_ws = np.zeros((max_lane))
    # count of data per lane for all dfs
    count_y_s = np.zeros((max_lane))
    for lane_itr in range(0, max_lane):
        count_y_s[lane_itr] = sum([count_y_[lane_itr] for count_y_ in count_y])
    
    for i in range(len(count_y)):
        for lane_itr in range(0, max_lane):
            average_y_ws[lane_itr] += (average_y[i][lane_itr]*count_y[i][lane_itr])/count_y_s[lane_itr]
    
    # estimate lane markings based on average y per lane for all dfs    
    for lane_itr in range(1, max_lane):
        lane_markings[lane_itr] = average_y_ws[lane_itr-1] + (average_y_ws[lane_itr] - average_y_ws[lane_itr-1])/2
    
    lane_markings[0] = 2*lane_markings[1] - lane_markings[2] #min(2*lane_markings[1] - lane_markings[2], min_y)
    lane_markings[-2] = 2*lane_markings[-3] - lane_markings [-4] #max(2*lane_markings[-3] - lane_markings [-4], max_y)
    lane_markings[-1] = lane_markings[-2] + (lane_markings[-3]- lane_markings[-4])
    
    with open(configs['dataset']['map_export_dir'], 'wb') as handle:
        pickle.dump(lane_markings, handle, protocol = pickle.HIGHEST_PROTOCOL)

    configs['dataset']['lane_markings'] = lane_markings
    print(lane_markings)
    return {'configs': configs, 'df': None, 'tracks_data': None,'frames_data': None}
 
def update_lane_ids(configs,df_itr,  df_data, tracks_data, frames_data):
    # update lane ids based on estimated lane markings
    #inner lane violation count
    olv_c = 0
    #outer lane violation count
    ilv_c = 0
    lane_markings = configs['dataset']['lane_markings']
    lane_update_ratio = 0
    for track_itr, track_data in enumerate(tracks_data):
        total_frames = len(track_data[p.X])
        lane_id = np.zeros((total_frames))
        for fr in range(total_frames):
            for i in range(len(lane_markings)-2):
                if track_data[p.Y][fr]<=lane_markings[i+1] and track_data[p.Y][fr]>=lane_markings[i]:
                    lane_id[fr] = i+2 #lane ids start at 2 in highD
            
            #exceptions
            if track_data[p.Y][fr]<lane_markings[0]:
                ilv_c +=1
                lane_id[fr] = 2
            if track_data[p.Y][fr]>lane_markings[-2]:
                olv_c +=1
                lane_id[fr] = len(lane_markings)-1
        lane_update_ratio += np.sum(tracks_data[track_itr][p.LANE_ID] != lane_id-1)/(total_frames*len(tracks_data))  
        tracks_data[track_itr][p.LANE_ID] = lane_id
    
    print('DF Itr: {}. Lane Update Ratio:{}, Outer lane violation counts (classified as lane max): {}, Inner lane violation counts (classified as lane min): {}'.format(df_itr, lane_update_ratio, olv_c, ilv_c))
    return {'configs': None, 'df': None, 'tracks_data': tracks_data,'frames_data': None}

def calc_svs(configs, df_itr,  df_data, tracks_data, frames_data):  
    for frame_itr, frame_data in enumerate(frames_data):
        for track_itr, track_id in enumerate(frame_data[p.TRACK_ID]):
            lane_id = frame_data[p.LANE_ID][frame_data[p.TRACK_ID] == track_id]
            vehicle_front_xloc = frame_data[p.X][frame_data[p.TRACK_ID] == track_id] + frame_data[p.WIDTH][frame_data[p.TRACK_ID] == track_id]/2
            vehicle_back_xloc = frame_data[p.X][frame_data[p.TRACK_ID] == track_id] - frame_data[p.WIDTH][frame_data[p.TRACK_ID] == track_id]/2
            same_lane_vehicles_itrs = frame_data[p.LANE_ID] == lane_id
            right_lane_vehicles_itrs = frame_data[p.LANE_ID] == lane_id+1 #for right to left driving in image plane
            left_lane_vehicles_itrs = frame_data[p.LANE_ID] == lane_id-1 #for right to left driving in image plane

            preceding_vehicles_itrs = (frame_data[p.X] - frame_data[p.WIDTH]/2)> vehicle_front_xloc
            following_vehicles_itrs = (frame_data[p.X] + frame_data[p.WIDTH]/2)< vehicle_back_xloc
            alongside_vehicles_itrs = np.logical_not(np.logical_or(preceding_vehicles_itrs, following_vehicles_itrs))

            rpv_itrs = np.nonzero(np.logical_and(preceding_vehicles_itrs, right_lane_vehicles_itrs))[0]
            lpv_itrs = np.nonzero(np.logical_and(preceding_vehicles_itrs, left_lane_vehicles_itrs))[0]
            pv_itrs = np.nonzero(np.logical_and(preceding_vehicles_itrs, same_lane_vehicles_itrs))[0]

            rfv_itrs = np.nonzero(np.logical_and(following_vehicles_itrs, right_lane_vehicles_itrs))[0]
            lfv_itrs = np.nonzero(np.logical_and(following_vehicles_itrs, left_lane_vehicles_itrs))[0]
            fv_itrs = np.nonzero(np.logical_and(following_vehicles_itrs, same_lane_vehicles_itrs))[0]

            rav_itrs = np.nonzero(np.logical_and(alongside_vehicles_itrs, right_lane_vehicles_itrs))[0]
            lav_itrs = np.nonzero(np.logical_and(alongside_vehicles_itrs, left_lane_vehicles_itrs))[0]

            if len(rav_itrs)>0:
                rav_itr = np.argmin(abs(frame_data[p.X][rav_itrs] + frame_data[p.WIDTH][rav_itrs]/2 - vehicle_back_xloc))
                frames_data[frame_itr][p.RIGHT_ALONGSIDE_ID][track_itr] = frame_data[p.TRACK_ID][rav_itrs[rav_itr]]                                            

            if len(lav_itrs)>0:
                lav_itr = np.argmin(abs(frame_data[p.X][lav_itrs] + frame_data[p.WIDTH][lav_itrs]/2 - vehicle_back_xloc))
                frames_data[frame_itr][p.LEFT_ALONGSIDE_ID][track_itr] = frame_data[p.TRACK_ID][lav_itrs[lav_itr]]      
            
            if len(rpv_itrs)>0:
                rpv_itr = np.argmin(abs(frame_data[p.X][rpv_itrs]- vehicle_front_xloc))
                frames_data[frame_itr][p.RIGHT_PRECEDING_ID][track_itr] = frame_data[p.TRACK_ID][rpv_itrs[rpv_itr]]
            if len(lpv_itrs)>0:
                lpv_itr = np.argmin(abs(frame_data[p.X][lpv_itrs]- vehicle_front_xloc))
                frames_data[frame_itr][p.LEFT_PRECEDING_ID][track_itr] = frame_data[p.TRACK_ID][lpv_itrs[lpv_itr]]
            if len(pv_itrs)>0:
                pv_itr = np.argmin(abs(frame_data[p.X][pv_itrs]- vehicle_front_xloc))
                frames_data[frame_itr][p.PRECEDING_ID][track_itr] = frame_data[p.TRACK_ID][pv_itrs[pv_itr]]

            if len(fv_itrs)>0:
                fv_itr = np.argmin(abs(frame_data[p.X][fv_itrs]- vehicle_back_xloc))
                frames_data[frame_itr][p.FOLLOWING_ID][track_itr] = frame_data[p.TRACK_ID][fv_itrs[fv_itr]]
            
            if len(rfv_itrs)>0:
                rfv_itr = np.argmin(abs(frame_data[p.X][rfv_itrs]- vehicle_back_xloc))
                frames_data[frame_itr][p.RIGHT_FOLLOWING_ID][track_itr] = frame_data[p.TRACK_ID][rfv_itrs[rfv_itr]]
            
            if len(lfv_itrs)>0:
                lfv_itr = np.argmin(abs(frame_data[p.X][lfv_itrs]- vehicle_back_xloc))
                frames_data[frame_itr][p.LEFT_FOLLOWING_ID][track_itr] = frame_data[p.TRACK_ID][lfv_itrs[lfv_itr]]             

    return {'configs': None, 'df': None, 'tracks_data': None,'frames_data': frames_data}


def convert_units(configs,df_itr,  df_data, tracks_data, frame_data):
    FOOT2METER = 0.3048
    lane_markings = configs['dataset']['lane_markings']
    #print(lane_markings[0])
    #print(lane_markings)
    lane_markings[df_itr] = lane_markings[df_itr]*FOOT2METER 
    configs['dataset']['lane_markings'] = lane_markings
    df_data[p.X] = df_data[p.X].apply(lambda x: x*FOOT2METER)
    df_data[p.Y] = df_data[p.Y].apply(lambda x: x*FOOT2METER)
    df_data[p.WIDTH] = df_data[p.WIDTH].apply(lambda x: x*FOOT2METER)
    df_data[p.HEIGHT] = df_data[p.HEIGHT].apply(lambda x: x*FOOT2METER)
    
    df_data[p.X_VELOCITY] = df_data[p.X_VELOCITY].apply(lambda x: x*FOOT2METER)
    df_data[p.Y_VELOCITY] = df_data[p.Y_VELOCITY].apply(lambda x: x*FOOT2METER)
    df_data[p.X_ACCELERATION] = df_data[p.X_ACCELERATION].apply(lambda x: x*FOOT2METER)
    df_data[p.Y_ACCELERATION] = df_data[p.Y_ACCELERATION].apply(lambda x: x*FOOT2METER)
    return {'configs': configs, 'df': df_data, 'tracks_data': None,'frames_data': None}


def visualise_tracks(configs,df_itr,  df_data, tracks_data, frames_data):
    
    
    x_bias = configs['dataset']['x_bias']
    y_bias = configs['dataset']['y_bias']
    with open(configs['dataset']['map_export_dir'], 'rb') as handle:
        lane_marking_dict = pickle.load(handle)
    #lane_markings = [-0.3991832068562266, 3.300408440899772, 7.000000088655771, 10.707917576951486,14.479769440085882, 18.369294269581566, 22.682077437497007, 26.994860605412452, 31.307643773327893]#configs['meta_data']['lane_markings'][df_itr]
    vis_cdir = os.path.join(os.path.join(p.VIS_DIR, configs['dataset']['name']), 'tracks')
    vis_cdir = os.path.join(vis_cdir, time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(vis_cdir):
        os.makedirs(vis_cdir)
    
    frame_itr_list = [frame[p.FRAME][0] for frame in frames_data]
    image_height = configs['dataset']['image_height']
    image_width = configs['dataset']['image_width']
    background_image = np.zeros(( image_height*p.Y_SCALE, image_width*p.X_SCALE, 3), dtype = np.uint8)
    #print(background_image.shape)
    x_pos = lambda x:x
    y_pos = lambda x:x
    if isinstance(lane_marking_dict, dict):
        for lane in lane_marking_dict:
                for itr in range(len(lane['r'])-1):
                    cv2.line(background_image, (x_pos(lane['r'][itr,0]), y_pos(lane['r'][itr,1])),(x_pos(lane['r'][itr+1,0]), y_pos(lane['r'][itr+1,1])), (0,255,0), thickness= 3)
                for itr in range(len(lane['l'])-1):
                    cv2.line(background_image, (x_pos(lane['l'][itr,0]), y_pos(lane['l'][itr,1])),(x_pos(lane['l'][itr+1,0]), y_pos(lane['l'][itr+1,1])), (0,255,0), thickness= 3)
    elif isinstance(lane_marking_dict, np.ndarray):
        for itr in range(len(lane_marking_dict)-1):
            cv2.line(background_image, (0, int((lane_marking_dict[itr]+y_bias)*p.Y_SCALE)),(image_width, int((lane_marking_dict[itr]+y_bias)*p.Y_SCALE)), (0,255,0), thickness= 3)
    else:
        raise ValueError('lane_marking_dict should be either a dictionary or a numpy array')
    #print(int(image_width*p.X_SCALE))
    total_vis = min(p.VISUALISATION_COUNT, len(tracks_data))
    track_itrs = random.sample(range(len(tracks_data)),len(tracks_data))[:total_vis]
    for tv_itr in track_itrs:
            tv_id = tracks_data[tv_itr][p.TRACK_ID][0]
            frames = tracks_data[tv_itr][p.FRAME]
            sv_ids = get_svs_ids(tracks_data[tv_itr])
            #images = []
            for fr_itr, frame in enumerate(frames):
                frame_itr = frame_itr_list.index(frame)
                frame_data = frames_data[frame_itr]
                image = np.copy(background_image)
                for track_itr, track_id in enumerate(frame_data[p.TRACK_ID]):
                    lane_id = int(frame_data[p.LANE_ID][track_itr])
                    if track_id == tv_id:
                        text = 'TV:{}'.format(lane_id)
                        v_color = (255,51,51)
                    elif track_id in sv_ids[fr_itr]:
                        text = '{}:{}:{}'.format(p.SV_IDS_ABBR[np.argwhere(sv_ids[fr_itr]==track_id)[0][0]],lane_id, int(track_id))
                        v_color = (255,51,51)
                    else:
                        text = 'NV:{}:{}'.format(lane_id,int(track_id))
                        v_color = (0,255,0)
                    image = plot_vehicle(image, 
                                        (frame_data[p.X][track_itr], frame_data[p.Y][track_itr]), 
                                        (frame_data[p.WIDTH][track_itr],frame_data[p.HEIGHT][track_itr]),
                                        (x_bias, y_bias),
                                        v_color,
                                        text,
                                        (0,0,255)
                                        )
                cv2.imwrite(os.path.join(vis_cdir, 'File{}_TV{}_FR{}.png'.format(df_itr,tv_id, frame)), image)

    return {'configs': None, 'df': None, 'tracks_data': None,'frames_data': None}

def get_svs_ids(track_data):
        num_frames = len(track_data[p.FRAME])
        svs_ids = np.zeros((num_frames, 8))
        for itr in range(num_frames):
            for i in range(8):
                svs_ids[itr,i] = track_data[p.SV_IDs[i]][itr]
        return svs_ids


def plot_vehicle(image, centre, dimension, biases, v_color, text, t_color):
    (x_bias, y_bias) = biases
    top_left = (int((x_bias + centre[0]-dimension[0]/2)*p.X_SCALE), int((y_bias + centre[1]-dimension[1]/2)*p.Y_SCALE))
    bot_left = (int((x_bias + centre[0]-dimension[0]/2)*p.X_SCALE ), int((y_bias +centre[1]+dimension[1]/2)*p.Y_SCALE))
    bot_right = (int((x_bias + centre[0]+dimension[0]/2)*p.X_SCALE), int((y_bias +centre[1]+dimension[1]/2)*p.Y_SCALE))
    image = cv2.rectangle(image,top_left, bot_right,color=v_color, thickness = -1)
    image = cv2.putText(image, text, bot_left, cv2.FONT_HERSHEY_SIMPLEX, fontScale=p.FONT_SCALE, color = t_color, thickness=1 )
    return image

def template(configs,df_itr,  df_data, tracks_data, frames_data):
    return {'configs': None, 'df': None, 'tracks_data': None,'frames_data': None}

