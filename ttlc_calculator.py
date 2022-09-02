import numpy as np
import pandas as pd
import  os
import pickle
import params as p
from data_format_converter import convert_format
import matplotlib.pyplot as plt
def ttlc_calculator(tracks, export_file_dir, lc_tracks_only = False):
    dfs =[]
    for track_idx, track in enumerate(tracks):
        driving_dir = 2
        lane_data = track[p.LANE_ID]
        lane_set = list(set(lane_data))
        if len(lane_set) >1:     
            lane_mem = lane_data
            prev_last_idx = 0
            last_idxs = []
            labels = []
            while True:
                start_lane = lane_data[0]
                last_idxs.append(prev_last_idx + np.nonzero(lane_data!= start_lane)[0][0]) # the idx of the first element in lane_data which is not equal to start_lane
                prev_last_idx = last_idxs[-1]
                lane_data = lane_mem[prev_last_idx:]
                cur_lane = lane_data[0]
                if driving_dir == 1:
                    if  cur_lane < start_lane:
                        label = 1 # right lane change
                    elif cur_lane > start_lane:
                        label = 2 # left lane change                      
                elif driving_dir == 2:
                    if cur_lane > start_lane:
                        label = 1 # right lane change
                    elif cur_lane < start_lane:
                        label = 2 # left lane change
                labels.append(label)
                if np.all(lane_data == lane_data[0]):
                    break
            
            TTLC_array = np.array([np.ones_like(track[p.LANE_ID]), np.ones_like(track[p.LANE_ID])], dtype=np.float)*p.MAX_TTLC
            pred_TTLC_array = np.array([np.ones_like(track[p.LANE_ID]), np.ones_like(track[p.LANE_ID])], dtype=np.float)*p.MAX_TTLC
            
            start_idx = 0
            for num, label in enumerate(labels):
                TTLC_array[label-1, start_idx:last_idxs[num]] = np.arange(last_idxs[num]-start_idx,0,-1)/p.FPS
            TTLC_array = np.clip(TTLC_array, 0, p.MAX_TTLC)
            #tracks[track_idx][p.PTTRLC] = pred_TTLC_array[0]
            #tracks[track_idx][p.PTTLLC] = pred_TTLC_array[1]
            tracks[track_idx][p.TTRLC] = TTLC_array[0]
            tracks[track_idx][p.TTLLC] = TTLC_array[1]
        else:
            TTLC_array = np.array([np.ones_like(track[p.LANE_ID]), np.ones_like(track[p.LANE_ID])], dtype=np.float)*p.MAX_TTLC
            pred_TTLC_array = np.array([np.ones_like(track[p.LANE_ID]), np.ones_like(track[p.LANE_ID])], dtype=np.float)*p.MAX_TTLC
            #tracks[track_idx][p.PTTRLC] = pred_TTLC_array[0]
            #tracks[track_idx][p.PTTLLC] = pred_TTLC_array[1]
            tracks[track_idx][p.TTRLC] = TTLC_array[0]
            tracks[track_idx][p.TTLLC] = TTLC_array[1]

        tracks[track_idx][p.ID] = np.ones_like(track[p.LANE_ID])*(track_idx+1)
        if lc_tracks_only == False or len(lane_set)>1:
            df=pd.DataFrame.from_dict(tracks[track_idx])
            dfs.append(df)
    
    export_df = pd.concat(dfs, sort=True) 
    export_df = export_df[[p.FRAME, p.ID, p.TTLLC, p.TTRLC]]
    export_df.to_csv(export_file_dir, index = False)
        


def plot_lc_vs_x(tracks):
    last_idxs = []
    labels = []
    track_idxs = []        
    for track_idx, track in enumerate(tracks):
        driving_dir = 2
        lane_data = track[p.LANE_ID]
        lane_set = list(set(lane_data))
        if len(lane_set) >1:     
            lane_mem = lane_data
            prev_last_idx = 0
            while True:
                start_lane = lane_data[0]
                last_idxs.append(prev_last_idx + np.nonzero(lane_data!= start_lane)[0][0]) # the idx of the first element in lane_data which is not equal to start_lane
                prev_last_idx = last_idxs[-1]
                lane_data = lane_mem[prev_last_idx:]
                cur_lane = lane_data[0]
                if driving_dir == 1:
                    if  cur_lane < start_lane:
                        label = 1 # right lane change
                    elif cur_lane > start_lane:
                        label = 2 # left lane change                      
                elif driving_dir == 2:
                    if cur_lane > start_lane:
                        label = 1 # right lane change
                    elif cur_lane < start_lane:
                        label = 2 # left lane change
                labels.append(label)
                track_idxs.append(track_idx)
                if np.all(lane_data == lane_data[0]):
                    break
    
    d = []
    for i,id in enumerate(track_idxs):
        d.append(tracks[id][p.D][last_idxs[i]])
    labels = np.array(labels)
    d = np.array(d)
    rlc_indx = labels == 1
    llc_indx = labels == 2
    
    print('RLC: {}, LLC: {}, TRACK_NUM:{}'.format(sum(rlc_indx), sum(llc_indx), len(tracks)))
    plt.hist(d, bins=100)
    plt.ylabel('Number')
    plt.xlabel('Data')
    plt.show()
            
if __name__ == '__main__':
    
    converter = convert_format(p.DATA_FILES, p.LANE_MARKINGS_FILE, preprocess_df= False)
    ttlc_calculator(converter.vehicle_data_list[0], export_file_dir = './M40draft2_GT_TTLCs.csv')
    #plot_lc_vs_x(converter.vehicle_data_list[0])