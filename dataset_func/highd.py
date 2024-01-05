import numpy as np
import pandas as pd
import os
import pdb
import params as p

def convert_drivingdir(configs,itr, df_data = None, tracks_data = None, frames_data = None):
    dset_conf = configs['dataset']
    file_id = eval(dset_conf['fileranges'])[itr]
    
    static_file = os.path.join(dset_conf['statics_dir'],
                                dset_conf['static_file'].format(str(file_id).zfill(2)))
    
    static_df = pd.read_csv(static_file)
    static_df = static_df[['id', 'drivingDirection']]
    static_df = static_df.sort_values(by=['id'])
    static_df = static_df.to_numpy()
    d2ids = static_df[static_df[:,1]== 2]
    d2ids =d2ids[:,0]
    d1ids = static_df[static_df[:,1]== 1]
    d1ids =d1ids[:,0]
    tracks_data_dir2 = [track_data for track_data in tracks_data \
                        if track_data[p.TRACK_ID][0] in d2ids]
    #TODO: to be moved
     
    #TODO continue development

    return {'configs': None, 'df': None, 'tracks_data': tracks_data_dir2,'frames_data': None}

def extract_lanemarkings(configs,itr, df_data, tracks_data = None, frames_data = None):
    dset_conf = configs['dataset']
    file_id = eval(dset_conf['fileranges'])[itr]
    meta_file = os.path.join(dset_conf['meta_dir'],
                                dset_conf['meta_file'].format((str(file_id).zfill(2))))
    meta_df = pd.read_csv(meta_file)
    lwr_lm = meta_df['lowerLaneMarkings']
    upr_lm = meta_df['upperLaneMarkings']
    return {'configs': None, 'df': df_data, 'tracks_data': None,'frames_data': None}