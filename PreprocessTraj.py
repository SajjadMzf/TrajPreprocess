import numpy as np
import pandas as pd
import  os
import pickle
import matplotlib.pyplot as plt
import yaml

import params as p
from utils.data_frame_functions import group_df, group2df
import utils.coordinate_functions as cf
from time import time
import ngsim, exid
import argparse
import pdb 

# From ID,Frame,X,Y to xVelocity, yVelocity, xAcceleration, yAcceleration, SVs_ID
class PreprocessTraj():
    def __init__(self, configs_file, constants_file):
        
        with open(configs_file) as f:
            self.configs = yaml.load(f, Loader = yaml.SafeLoader)
        with open(constants_file) as f:
            self.constants = yaml.load(f, Loader = yaml.FullLoader)
             
        if 'filenames' in self.configs['dataset']:
            self.data_files = self.configs['dataset']['filenames']
        else:
            self.data_files = []
            string_format = self.configs['dataset']['filestring']
            file_ranges = []
            for i in eval(self.configs['dataset']['fileranges']):
                file_ranges.append(str(i).zfill(2))
                self.data_files.append(string_format.format(file_ranges[-1]))
        self.frame_dirs = []
        self.track_dirs = []
        self.df_dirs = []
        
        # handle dirs
        frame_dir = os.path.join(self.configs['dataset']['export_dir'], p.FRAME_SAVE_DIR)
        if os.path.exists(frame_dir) == False:
            os.makedirs(frame_dir)
        track_dir = os.path.join(self.configs['dataset']['export_dir'], p.TRACK_SAVE_DIR)
        if os.path.exists(track_dir) == False:
            os.makedirs(track_dir)
        df_dir = os.path.join(self.configs['dataset']['export_dir'], p.DF_SAVE_DIR)
        if os.path.exists(df_dir) == False:
            os.makedirs(df_dir)
        map_dir = '/'.join(self.configs['dataset']['map_export_dir'].split('/')[:-1])
        if os.path.exists(map_dir) == False:
            os.makedirs(map_dir)

        for i ,data_file in enumerate(self.data_files):
            if 'filenames' in self.configs['dataset']:
                frame_file = ''.join(data_file.split('.')[0:-1])+ '_frames.pickle'
                track_file = ''.join(data_file.split('.')[0:-1])+ '_tracks.pickle'
                df_file = ''.join(data_file.split('.')[0:-1]) + '_tracks.csv'
            else:
                frame_file = ''.join(file_ranges[i])+ '_frames.pickle'
                track_file = ''.join(file_ranges[i])+ '_tracks.pickle'
                df_file = ''.join(file_ranges[i]) + '_tracks.csv'
            self.frame_dirs.append(os.path.join(frame_dir, frame_file))
            self.track_dirs.append(os.path.join(track_dir, track_file))
            self.df_dirs.append(os.path.join(df_dir, df_file)) 
            
        self.track_data_list = []
        self.frame_data_list = []
        

    def dataset_specific_preprocess(self):
        function_list = [func_list for  func_list in self.configs['ordered_preprocess_functions']]
        for func_itr, func_list in enumerate(function_list):
            func_str = func_list[0]
            func_type = func_list[1]
            if func_type== 'all':
                if ('(' in func_str) and (')' in func_str):
                    eval(func_str)
                else:
                    print('{}. {} of all files'.format(func_itr+1, func_str.split('.')[1].split('(')[0]))
                    start = time()
                    res_df = eval(func_str)(self.configs,None, self.df_data_list, self.track_data_list, self.frame_data_list)
                    if res_df['df'] is not None:
                        self.df_data_list = res_df['df']
                    if res_df['tracks_data'] is not None:
                        self.track_data_list = res_df['tracks_data']
                    if res_df['frames_data'] is not None:
                        self.frame_data_list = res_df['frames_data']
                    if res_df['configs'] is not None:
                        self.configs = res_df['configs']
                    print('In {} sec'.format(time()-start))

            elif func_type== 'one':
                
                for itr, df_data in enumerate(self.df_data_list):
                    print('{}.{}. {} of file {}'.format(func_itr+1,itr+1, func_str.split('.')[1].split('(')[0],self.data_files[itr]))
                    start = time()
                    
                    res_df = eval(func_str)(self.configs,itr, df_data, self.track_data_list[itr], self.frame_data_list[itr])
                    if res_df['df'] is not None:
                        self.df_data_list[itr] = res_df['df']
                    if res_df['tracks_data'] is not None:
                        self.track_data_list[itr] = res_df['tracks_data']
                    if res_df['frames_data'] is not None:
                        self.frame_data_list[itr] = res_df['frames_data']
                    if res_df['configs'] is not None:
                        self.configs = res_df['configs']
                    print('In {} sec'.format(time()-start))
            else:
                raise(ValueError('Undefined func_type'))


    def match_columns(self):
            self.output_columns = []
            matched_input_columns = []
            matched_output_columns = []
            unmatched_input_columns = self.configs['unmatched_columns']
            self.unmatched_output_columns = []
            for key, value in self.configs['matched_columns'].items():
                self.output_columns.append(eval('p.{}'.format(key)))
                if value != 'None':
                    matched_input_columns.append(value)
                    matched_output_columns.append(eval('p.{}'.format(key)))
                else:
                    self.unmatched_output_columns.append(eval('p.{}'.format(key)))
            matched_input_columns.extend(unmatched_input_columns)
            matched_output_columns.extend(unmatched_input_columns)
            self.output_columns.extend(unmatched_input_columns)
            self.input2output = dict(zip(matched_input_columns, matched_output_columns))
            self.output2input = dict(zip(matched_output_columns, matched_input_columns))

    def initialise_df(self):
        self.df_data_list = []
        data_files_cdir = [os.path.join(self.configs['dataset']['import_dir'], file_name) for file_name in self.data_files]
        for file_itr, data_file_cdir in enumerate(data_files_cdir):
            print('Importing df file: {}.'.format(data_file_cdir))
            df = pd.read_csv(data_file_cdir)
            df = df[list(self.output2input.values())]
            df = df.rename(columns = self.input2output)
            for empty_column in self.unmatched_output_columns:
                df[empty_column] = (np.ones((df.shape[0]))*-1).tolist()
            df = df.sort_values([p.TRACK_ID, p.FRAME], ascending=  [1,1])
            self.df_data_list.append(df) 

    def reduce_fps(self):
        fps_div = int(self.configs['dataset']['dataset_fps']/self.configs['dataset']['desired_fps'])
        for file_itr, frame_data in enumerate(self.frame_data_list):
            self.frame_data_list[file_itr] = frame_data[0:-1:fps_div]



    def export_statics_metas(self):
        for file_itr, file_name in enumerate(self.data_files):
            meta_data = [-1]*len(p.metas_columns)
            meta_data[p.metas_columns.index('upperLaneMarkings')] = ';'.join(str(e) for e in self.configs['meta_data']['lane_markings'][file_itr])
            meta_data[p.metas_columns.index('lowerLaneMarkings')] = ';'.join(str(e) for e in self.configs['meta_data']['lane_markings'][file_itr])
            meta_data[p.metas_columns.index('frameRate')] = self.configs['dataset']['fps']
            #print(meta_data)
            meta_df = pd.DataFrame([meta_data], columns= p.metas_columns)
            
            print('Exporting statics/metas of file: {}'.format(self.data_files[file_itr]))
            static_df = pd.DataFrame()
            track_ids = self.df_data_list[file_itr][p.TRACK_ID].values
            track_ids = np.unique(track_ids)
            static_df[p.TRACK_ID] = track_ids
            remaining_columns = p.statics_columns
            static_df['drivingDirection'] = np.ones((len(static_df[p.TRACK_ID]))) 
            
            for column in remaining_columns:
                static_df[column] = np.ones((len(static_df[p.TRACK_ID])))*-1
            
            meta_data_file = ''.join(self.data_files[file_itr].split('.csv')[0:-1]) + '_recordingMeta.csv'
            static_data_file = ''.join(self.data_files[file_itr].split('.csv')[0:-1]) + '_tracksMeta.csv'
            meta_cdir = os.path.join(self.configs['dataset']['export_dir']+p.META_SAVE_DIR, meta_data_file)
            static_cdir = os.path.join(self.configs['dataset']['export_dir']+p.STATICS_SAVE_DIR, static_data_file)
            meta_df.to_csv(meta_cdir, index = False)
            static_df.to_csv(static_cdir, index = False)
            
    def import_processed_data(self, data_type = 'all', args = (None,)):
        valid_data_types = ['df', 'track', 'frame', 'all']
        if data_type not in valid_data_types:
            raise(ValueError('Invalid data type'))
        print('Import processed data of type: {}'.format(data_type))
        if data_type == 'df' or data_type == 'all':
            self.df_data_list = []
            for file_itr, data_file in enumerate(self.data_files):
                df = pd.read_csv(self.df_dirs[file_itr])
                self.df_data_list.append(df)
        
        if data_type == 'tracks' or data_type == 'all':
            self.track_data_list = []
            for file_itr, data_file in enumerate(self.data_files):
                with open(self.track_dirs[file_itr], 'rb') as handle:
                    self.track_data_list.append(pickle.load(handle))
            
        if data_type == 'frames' or data_type == 'all':
            self.frame_data_list = []
            for file_itr, data_file in enumerate(self.data_files):
                with open(self.frame_dirs[file_itr], 'rb') as handle:
                    self.frame_data_list.append(pickle.load(handle))
    
    def export_data(self, data_type = 'all', args= (None,)):
        valid_data_types = ['df', 'track', 'frame', 'all']
        if data_type not in valid_data_types:
            raise(ValueError('Invalid data type'))
        if data_type == 'df' or data_type == 'all':
            for file_itr, data_file in enumerate(self.data_files):
                print('Exporting DF file: {}'.format(self.data_files[file_itr], len(self.track_data_list[file_itr])))
                df = self.df_data_list[file_itr]
                df.sort_values(by=[p.TRACK_ID, p.FRAME], inplace = True)
                df = df[self.output_columns]
                df.to_csv(self.df_dirs[file_itr], index = False)
        
        if data_type == 'track' or data_type == 'all':
            for file_itr, data_file in enumerate(self.data_files):
                print('Exporting Tracks file: {} with {} Tracks'.format(self.data_files[file_itr], len(self.track_data_list[file_itr])))
                with open(self.track_dirs[file_itr], 'wb') as handle:
                    pickle.dump(self.track_data_list[file_itr], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        if data_type == 'frame' or data_type == 'all':
            column_list = args[0]
            for file_itr, data_file in enumerate(self.data_files):
                print('Exporting Frames file: {} with {} Frames'.format(self.data_files[file_itr], len(self.frame_data_list[file_itr])))
                with open(self.frame_dirs[file_itr], 'wb') as handle:
                    pickle.dump(self.frame_data_list[file_itr], handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    def overwrite_data(self, source, target = 'rest'):
        '''
        source options: 'df', 'track', 'frame'
        target_options: 'df', 'track', 'frame', 'rest
        '''
        valid_targets = ['track','frame', 'df', 'rest']
        valid_sources = ['track','frame', 'df']
        if source not in valid_sources or target not in valid_targets or source==target:
            raise(ValueError('Invalid source target'))
        print('Overwrite {} based on {}'.format(target, source))
        if source != 'df' and ( target == 'df' or target == 'rest'):
            self.df_data_list = []
        if source != 'track' and (target == 'track' or target == 'rest'):
            self.track_data_list = []
        if source != 'frame' and (target == 'frame' or target == 'rest'):
            self.frame_data_list = []
        
        if source == 'track':
            for file_itr, track_data in enumerate(self.track_data_list):
                #print('Update DF from source {} of file: {}'.format(source ,self.data_files[file_itr]))
                df = group2df(track_data)
                if target == 'df' or target == 'rest':
                    self.df_data_list.append(df)      
                if target == 'frame' or target == 'rest':
                    self.frame_data_list.append(group_df(df, by = p.FRAME))
        
        elif source == 'frame':
            for file_itr, frame_data in enumerate(self.frame_data_list):
                #print('Update DF from source {} of file: {}'.format(source ,self.data_files[file_itr]))
                df = group2df(frame_data)
                if target == 'df' or target == 'rest':
                    self.df_data_list.append(df)      
                if target == 'track' or target == 'rest':
                    self.track_data_list.append(group_df(df, by = p.TRACK_ID))
        elif source == 'df':
            for df_itr in range(len(self.df_data_list)):
                if target == 'tracks' or target == 'rest':
                    self.track_data_list.append(group_df(self.df_data_list[df_itr],by = p.TRACK_ID))
                if target == 'frames' or target == 'rest':
                    self.frame_data_list.append(group_df(self.df_data_list[df_itr], by = p.FRAME))    
        else:
            raise(ValueError('Unknown Source'))
      

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('config_file', type=str)
    #args = parser.parse_args()

    for i in [2, 3,4,6]:
        preprocess = PreprocessTraj(
            #args.config_file,
            'configs/exid_preprocess{}.yaml'.format(i),
            'configs/constants.yaml'
        )
        preprocess.dataset_specific_preprocess()
    
    # TODO: update traj filtering
    # TODO tracks_tracks!