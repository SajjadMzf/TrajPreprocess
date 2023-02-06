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
import cv2
import random
from pyproj import Proj
from ngsim import get_svs_ids

def calc_svs(configs, df_itr,  df_data, tracks_data, frames_data):  
    # pv: preceding vehicle, fv:following vehicle, rv1(prv),rv2(rav),rv3(frv): three closest vehicles in right lane, lv1(plv),lv2(lav),lv3(frv): three closest vehicles in left lane.
    # we used same names for side vehicles as highD, however, their definitions are different. 
    for frame_itr, frame_data in enumerate(frames_data):
        for track_itr, track_id in enumerate(frame_data[p.TRACK_ID]):
            X_ = frame_data[p.X][track_itr]
            lane_id = frame_data[p.LANE_ID][track_itr]
            slv_itrs = frame_data[p.LANE_ID] == lane_id
            # pv
            pv_itrs = np.nonzero(np.logical_and(frame_data[p.X]> X_, slv_itrs))[0]
            if len(pv_itrs)>0:
                pv_itr = np.argmin(abs(frame_data[p.X][pv_itrs]- X_))
                frames_data[frame_itr][p.PRECEDING_ID][track_itr] = frame_data[p.TRACK_ID][pv_itrs[pv_itr]]
            #fv
            fv_itrs = np.nonzero(np.logical_and(frame_data[p.X]< X_, slv_itrs))[0]
            if len(fv_itrs)>0:
                fv_itr = np.argmin(abs(frame_data[p.X][fv_itrs]- X_))
                frames_data[frame_itr][p.FOLLOWING_ID][track_itr] = frame_data[p.TRACK_ID][fv_itrs[fv_itr]]
            
            # rv1, rv2, rv3
            rvs = np.nonzero(frame_data[p.LANE_ID] == (lane_id+1))[0] #left to right driving in image plane, bot to top lane orders 
            if len(rvs)>0:
                rv_itrs = np.argsort(np.abs(frame_data[p.X][rvs]-X_))
                rv_n = min([len(rv_itrs),3])
                rv_itrs = rv_itrs[0:rv_n]
                rv_itrs = np.argsort(frame_data[p.X][rvs[rv_itrs]]-X_)
                for i, rv_itr in enumerate(rv_itrs):
                    frames_data[frame_itr][p.RV_IDs[i]][track_itr] = frame_data[p.TRACK_ID][rvs[rv_itr]]


            # lv1, lv2, lv3
            lvs = np.nonzero(frame_data[p.LANE_ID] == (lane_id-1))[0] #left to right driving in image plane, bot to top lane orders
            if len(lvs)>0:    
                lv_itrs = np.argsort(np.abs(frame_data[p.X][lvs]-X_))
                lv_n = min([len(lv_itrs),3])
                lv_itrs = lv_itrs[0:lv_n]
                lv_itrs = np.argsort(frame_data[p.X][lvs[lv_itrs]]-X_)
                #pdb.set_trace()
                for i, lv_itr in enumerate(lv_itrs):
                    frames_data[frame_itr][p.LV_IDs[i]][track_itr] = frame_data[p.TRACK_ID][lvs[lv_itr]]
            
            
    return {'configs': None, 'df': None, 'tracks_data': None,'frames_data': frames_data}



def visualise_tracks(configs,df_itr, df_data, tracks_data = None, frames_data = None):
    with open(configs['meta_data']['lane_markings_export_file'], 'rb') as handle:
        lane_marking_dict = pickle.load(handle)
    if p.VISUALISE_FRENET == False:
        lanes = lane_marking_dict['lane_nodes']
        X_ = 'xCenter'
        Y_ = 'yCenter'
        circle = True
    else:
        lanes = lane_marking_dict['lane_nodes_frenet']
        X_ = p.X
        Y_ = p.Y
        circle = False

    lane_y_max = max([max(lane['l'][:,1]) for lane in lanes])
    lane_y_min = min([min(lane['l'][:,1]) for lane in lanes])
    lane_x_max = max([max(lane['l'][:,0]) for lane in lanes])
    lane_x_min = min([min(lane['l'][:,0]) for lane in lanes])
    
    bias = 10
    tracks_dir = 'visualisations/exid'
    if os.path.exists(tracks_dir) == False:
        os.makedirs(tracks_dir)
    for filename in os.listdir(tracks_dir):
        if 'File{}'.format(df_itr) in filename:
            file_path = os.path.join(tracks_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    frame_itr_list = [frame[p.FRAME][0] for frame in frames_data]
    image_width = lane_x_max - lane_x_min
    image_height = lane_y_max+bias - lane_y_min
    print('image_width: {}, image_height: {}'.format(image_width*p.X_SCALE, image_height*p.Y_SCALE))
    
    background_image = np.zeros(( int(image_height*p.Y_SCALE), int(image_width*p.X_SCALE), 3), dtype = np.uint8)
    #print(background_image.shape)
    x_pos = lambda x: int((x -lane_x_min +bias/2)*p.X_SCALE)
    y_pos = lambda y: int((lane_y_max+bias/2- y)*p.Y_SCALE)
    for lane in lanes:
            for itr in range(len(lane['r'])-1):
                cv2.line(background_image, (x_pos(lane['r'][itr,0]), y_pos(lane['r'][itr,1])),(x_pos(lane['r'][itr+1,0]), y_pos(lane['r'][itr+1,1])), (0,255,0), thickness= 3)
            for itr in range(len(lane['l'])-1):
                cv2.line(background_image, (x_pos(lane['l'][itr,0]), y_pos(lane['l'][itr,1])),(x_pos(lane['l'][itr+1,0]), y_pos(lane['l'][itr+1,1])), (0,255,0), thickness= 3)
        #print(i
    #print(int(image_width*p.X_SCALE))
    cv2.imwrite(os.path.join(tracks_dir, 'lanes.png'), background_image)
    #pdb.set_trace()
    total_vis = min(p.VISUALISATION_COUNT, len(tracks_data))
    track_itrs = range(total_vis)#random.sample(range(len(tracks_data)),len(tracks_data))[:total_vis]
    for tv_itr in track_itrs:
            tv_id = tracks_data[tv_itr][p.TRACK_ID][0]
            frames = tracks_data[tv_itr][p.FRAME]
            print('Track X:{}-{}, Y:{}-{}'.format(x_pos(min(tracks_data[tv_itr][X_])),x_pos(max(tracks_data[tv_itr][X_])),y_pos(min(tracks_data[tv_itr][Y_])),y_pos(max(tracks_data[tv_itr][Y_]))))
    
            sv_ids = get_svs_ids(tracks_data[tv_itr])
            #images = []
            for fr_itr, frame in enumerate(frames):
                if fr_itr%5:
                    continue
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
                    image = vf.plot_vehicle(image, 
                                        (x_pos(frame_data[X_][track_itr]), y_pos(frame_data[Y_][track_itr])), 
                                        (frame_data[p.WIDTH][track_itr]*p.X_SCALE,frame_data[p.HEIGHT][track_itr]*p.Y_SCALE),
                                        v_color,
                                        text,
                                        (0,0,255),
                                        circle
                                        )
                cv2.imwrite(os.path.join(tracks_dir, 'File{}_TV{}_FR{}.png'.format(df_itr,tv_id, frame)), image)
    
    #cv2.imwrite(os.path.join(tracks_dir, 'lanes.png'), background_image)
    #pdb.set_trace()
    return {'configs': None, 'df': None, 'tracks_data': None,'frames_data': None}

def get_lane_ids(configs,df_itr, df_data, tracks_data = None, frames_data = None):
    # compute p.LANE_ID, p.LANE_WIDTH, p.DRIVING_DIR
    with open(configs['meta_data']['lane_markings_export_file'], 'rb') as handle:
        lane_marking_dict = pickle.load(handle)
    lanes = lane_marking_dict['lane_nodes_frenet']
    ids2remove = []
    n_violation = 0
    for id, track_data in enumerate(tracks_data):
        lane_id = np.ones((len(track_data[p.X])))*-1
        lane_width = np.ones((len(track_data[p.X])))*-1
        for fr in range(len(track_data[p.X])):
            x = track_data[p.X][fr]
            y = track_data[p.Y][fr]
            l_pos_y = []
            l_pos = []
            for lane_itr, lane in enumerate(lanes):
                l_pos.append(np.argmin(np.abs(lane['l'][:,0]-x)))
                pos_y = lane['l'][l_pos[-1],1]
                l_pos_y.append(pos_y)
            l_pos_y = np.array(l_pos_y)
            if np.any(l_pos_y>y) == False:
                #print('Warning: out of lane boundary data, y:{}, lane_y:{}'.format(y, max(l_pos_y)))
                cur_lane_id = len(l_pos_y)+2 # This represents vehicles driving in other direction
            else:
                cur_lane_id = np.nonzero(l_pos_y>y)[0][0]+1
                
                assert(len(lane['r'])>l_pos[cur_lane_id-1])
                lane_width[fr] = l_pos_y[cur_lane_id-1] - lane['r'][l_pos[cur_lane_id-1],1]
            lane_id[fr] = len(l_pos_y)+2 - cur_lane_id
        
        n_lane_violation = np.sum(lane_id==0)
        if n_lane_violation>0 and n_lane_violation<len(lane_id):
            n_violation +=1
            print('Warning: Out of lane boundary data, Track:{}, N:{}/{}'.format(id, n_lane_violation, len(lane_id)))
        
        if np.all(lane_id==0):
            ids2remove.append(id)
        else:
            tracks_data[id][p.LANE_ID] = lane_id
            tracks_data[id][p.LANE_WIDTH] = lane_width
    # Vehicles driving in the other direction
    print('N Violation : {}'.format(n_violation))
    print('Deleting {}/{} tracks (other driving direction)'.format(len(ids2remove), len(tracks_data)))
    for id in sorted(ids2remove, reverse = True):
        del tracks_data[id]
    #pdb.set_trace()  
    return {'configs': None, 'df': None, 'tracks_data': tracks_data,'frames_data': None}

def convert2frenet(configs,df_itr, df_data, tracks_data = None, frames_data = None):
    # compute p.X p.Y
    with open(configs['meta_data']['lane_markings_export_file'], 'rb') as handle:
        lane_marking_dict = pickle.load(handle)
    merge_frenet_origin = lane_marking_dict['merge_origin_lane']
    main_frenet_origin = lane_marking_dict['main_origin_lane']
    #pdb.set_trace()
    merge_s_bias = lane_marking_dict['merge2main_s_bias']
    for id, track_data in enumerate(tracks_data):
        traj = np.stack((track_data['xCenter'], track_data['yCenter']), axis = 1)
        merge_min_itr = np.argmin(np.linalg.norm(traj[0]-merge_frenet_origin, axis = 1)) 
        if merge_min_itr ==0:
            merge_min_itr += 1
        m_point1 = merge_frenet_origin[merge_min_itr-1] 
        m_point2 = merge_frenet_origin[merge_min_itr]
        if np.cross(m_point2-m_point1, traj[0]-m_point1)<0:
            frenet_ref = merge_frenet_origin
            x_bias = merge_s_bias
            
        else:
            frenet_ref = main_frenet_origin
            x_bias = 0
        traj_frenet = cf.cart2frenet(traj, frenet_ref)
        #if track_data[p.TRACK_ID][0] == 6:
        #    pdb.set_trace()
        tracks_data[id][p.X] = traj_frenet[:,0] + x_bias
        tracks_data[id][p.Y] = traj_frenet[:,1]
        #pdb.set_trace()
    return {'configs': None, 'df': None, 'tracks_data': tracks_data,'frames_data': None}



def hdmaps2lane_markings(configs,df_itr, df_data, tracks_data = None, frames_data = None):
    ll2p_yml_dir = configs['meta_data']['lane_markings_yml_dir']
    with open(ll2p_yml_dir) as f:
            lm_nodes = yaml.load(f, Loader = yaml.SafeLoader)
    lanes_ways = []
    for key in lm_nodes:
        lane = {}
        lane['r'] = lm_nodes[key]['right']
        lane['l'] = lm_nodes[key]['left']
        lane['rt'] = lm_nodes[key]['right_type']
        lane['lt'] = lm_nodes[key]['left_type']
        lanes_ways.append(lane)
        
    ll2p_file_dir = configs['meta_data']['lanelet2_file_dir']
    xml_tree = ET.parse(ll2p_file_dir)
    root = xml_tree.getroot()
    nodes = {}
    ways = {}
    for child in root:
        if child.tag == 'node':
            nodes[child.attrib['id']] = [child.attrib['lon'], child.attrib['lat']]
        elif child.tag == 'way':
            way_nodes = []
            for g_child in child:
                if g_child.tag == 'nd':
                    way_nodes.append(g_child.attrib['ref'])
            ways[child.attrib['id']] = way_nodes
    lanes_nodes = []
    lanes_nodes_types = []
    #longlat_origin = (configs['dataset']['lonOrigin'], configs['dataset']['latOrigin'])
    lonlat2utm = Proj("+proj=utm +zone=32U, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    utmXorig = configs['dataset']['xUtmOrigin']
    utmYorig = configs['dataset']['yUtmOrigin']  
    
    for lane_ways in lanes_ways:
        lane_nodes = {}
        lane_nodes_types = {}
        n_ways = len(lane_ways['r'])
        
        r_nodes = []
        l_nodes = []
        
        r_nodes_type = []
        l_nodes_type = []
        
        for way_itr in range(n_ways):
            r_way = lane_ways['r'][way_itr]
            r_node_ids = ways[str(r_way)]
            for r_node_id in r_node_ids:
                node = nodes[r_node_id]
                r_nodes.append([float(node[0]), float(node[1])])
            r_nodes_type.extend([lane_ways['rt'][way_itr]]*len(r_node_ids))
            l_way = lane_ways['l'][way_itr]
            l_node_ids = ways[str(l_way)]
            
            for l_node_id in l_node_ids:
                node = nodes[l_node_id]
                l_nodes.append([float(node[0]), float(node[1])])
            l_nodes_type.extend([lane_ways['lt'][way_itr]]*len(l_node_ids))
        

        r_nodes = np.array(r_nodes)
        utmX, utmY = lonlat2utm(r_nodes[:,0], r_nodes[:,1])
        r_nodes = np.stack((utmX - utmXorig, utmY - utmYorig), axis = 1)
        l_nodes = np.array(l_nodes)
        utmX, utmY = lonlat2utm(l_nodes[:,0], l_nodes[:,1])
        l_nodes = np.stack((utmX - utmXorig, utmY - utmYorig), axis = 1)
        lane_nodes['r'] = r_nodes#cf.longlat2xy(np.array(r_nodes), longlat_origin)
        lane_nodes['l'] = l_nodes#cf.longlat2xy(np.array(l_nodes), longlat_origin)
        lane_nodes_types['r'] = np.array(r_nodes_type)
        lane_nodes_types['l'] = np.array(l_nodes_type)
        #pdb.set_trace()
        r_keep = np.ones((len(lane_nodes['r'])), dtype = bool)
        for i in range(len(lane_nodes['r'])-1):
            if np.all(lane_nodes['r'][i] == lane_nodes['r'][i+1]):
                r_keep[i] = False
        lane_nodes['r'] = lane_nodes['r'][r_keep] 
        lane_nodes_types['r'] = lane_nodes_types['r'][r_keep]
        l_keep = np.ones((len(lane_nodes['l'])), dtype = bool)
        for i in range(len(lane_nodes['l'])-1):
            if np.all(lane_nodes['l'][i] == lane_nodes['l'][i+1]):
                l_keep[i] = False
        lane_nodes['l'] = lane_nodes['l'][l_keep] 
        lane_nodes_types['l'] = lane_nodes_types['l'][l_keep]
        
        lanes_nodes.append(lane_nodes)
        lanes_nodes_types.append(lane_nodes_types)
    
    n_lanes = len(lanes_nodes)
    
    for itr in range(n_lanes):
        if np.any(lanes_nodes_types[itr]['l'] == 2):
            merge_frenet_lm = lanes_nodes[itr]['l']
        elif np.any(lanes_nodes_types[itr]['r'] == 3):
            main_frenet_lm = lanes_nodes[itr]['r']
    merge2main_node = merge_frenet_lm[-1]
    merge2main_itr = np.nonzero(main_frenet_lm==merge2main_node)[0]
    assert(np.all(merge2main_itr == merge2main_itr[0]))
    merge2main_itr = merge2main_itr[0]
    lanes_nodes_frenet = []
    main_matched_point_s = cf.cart2frenet(main_frenet_lm, main_frenet_lm)[merge2main_itr,0]

    for itr, lane_nodes in enumerate(lanes_nodes):
        lane_nodes_frenet = {}
        if np.any(lanes_nodes_types[itr]['l'] == 2):
            l_nodes_frenet = cf.cart2frenet(lane_nodes['l'], merge_frenet_lm)
            r_nodes_frenet = cf.cart2frenet(lane_nodes['r'], merge_frenet_lm)
            merging_matched_point_s = l_nodes_frenet[-1,0]
            l_nodes_frenet[:,0] += main_matched_point_s - merging_matched_point_s
            r_nodes_frenet[:,0] += main_matched_point_s - merging_matched_point_s
        else:
            l_nodes_frenet = cf.cart2frenet(lane_nodes['l'], main_frenet_lm)
            r_nodes_frenet = cf.cart2frenet(lane_nodes['r'], main_frenet_lm)
        

        lane_nodes_frenet['r'] = r_nodes_frenet
        lane_nodes_frenet['l'] = l_nodes_frenet
        lanes_nodes_frenet.append(lane_nodes_frenet)

    '''
    lane_width_var = []
    for itr, lane_nodes_f in enumerate(lanes_nodes_frenet):
        r_var = np.max(lane_nodes_f['r'][:,1])- np.min(lane_nodes_f['r'][:,1])
        l_var = np.max(lane_nodes_f['l'][:,1])- np.min(lane_nodes_f['l'][:,1])
        lane_width_var.append(r_var)
        lane_width_var.append(l_var)
    
    lane_width_var = np.array(lane_width_var)
    print(lane_width_var)
    m = cf.cart2frenet(main_frenet_origin, main_frenet_origin)
    print(np.max(m[:,1])-np.min(m[:,1]))
    assert(max(lane_width_var<0.5))
    pdb.set_trace()
    '''
    # interpolate lane markings
    for itr, lane_nodes_f in enumerate(lanes_nodes_frenet):
        s = lane_nodes_f['r'][:,0]
        d = lane_nodes_f['r'][:,1]
        interpolate_fn = interp1d(s,d)
        new_s = np.arange(min(s), max(s), 0.1)
        new_d = interpolate_fn(new_s)
        new_traj = np.stack((new_s, new_d), axis = 1)
        lanes_nodes_frenet[itr]['r'] = new_traj

        s = lane_nodes_f['l'][:,0]
        d = lane_nodes_f['l'][:,1]
        interpolate_fn = interp1d(s,d)
        new_s = np.arange(min(s), max(s), 0.1)
        new_d = interpolate_fn(new_s)
        new_traj = np.stack((new_s, new_d), axis = 1)
        lanes_nodes_frenet[itr]['l'] = new_traj

    #print(lane_markings_frenets)
    lane_marking_dict = {}
    lane_marking_dict['lane_types'] = lanes_nodes_types
    lane_marking_dict['lane_nodes'] = lanes_nodes
    lane_marking_dict['lane_nodes_frenet'] = lanes_nodes_frenet
    lane_marking_dict['merge_origin_lane'] = merge_frenet_lm
    lane_marking_dict['main_origin_lane'] = main_frenet_lm
    lane_marking_dict['merge2main_s_bias'] = main_matched_point_s - merging_matched_point_s
    with open(configs['meta_data']['lane_markings_export_file'], 'wb') as handle:
        pickle.dump(lane_marking_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    return {'configs': configs, 'df': None, 'tracks_data': None,'frames_data': None}



