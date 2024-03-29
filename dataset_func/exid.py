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
from dataset_func.ngsim import get_svs_ids
from utils.utils_functions import interpolate_polyline, parametrise_polyline

def calc_svs_mmntp(configs, df_itr,  df_data, tracks_data, frames_data):  
    # pv: preceding vehicle, fv:following vehicle, 
    # rv1(prv),rv2(rav),rv3(frv): three closest vehicles in right lane,
    # lv1(plv),lv2(lav),lv3(frv): three closest vehicles in left lane.
    for frame_itr, frame_data in enumerate(frames_data):
        for track_itr, track_id in enumerate(frame_data[p.TRACK_ID]):
            X_ = frame_data[p.S][track_itr]
            lane_id = frame_data[p.LANE_ID][track_itr]
            slv_itrs = frame_data[p.LANE_ID] == lane_id
            # pv
            pv_itrs = np.nonzero(\
                np.logical_and(frame_data[p.S]> X_, slv_itrs))[0]
            if len(pv_itrs)>0:
                pv_itr = np.argmin(abs(frame_data[p.S][pv_itrs]- X_))
                frames_data[frame_itr][p.PRECEDING_ID][track_itr] =\
                      frame_data[p.TRACK_ID][pv_itrs[pv_itr]]
            #fv
            fv_itrs = np.nonzero(\
                np.logical_and(frame_data[p.S]< X_, slv_itrs))[0]
            if len(fv_itrs)>0:
                fv_itr = np.argmin(abs(frame_data[p.S][fv_itrs]- X_))
                frames_data[frame_itr][p.FOLLOWING_ID][track_itr] =\
                      frame_data[p.TRACK_ID][fv_itrs[fv_itr]]
            
            # rv1, rv2, rv3
            #left to right driving in image plane, bot to top lane orders 
            rvs = np.nonzero(frame_data[p.LANE_ID] == (lane_id+1))[0] 
            if len(rvs)>0:
                rv_itrs = np.argsort(np.abs(frame_data[p.S][rvs]-X_))
                rv_n = min([len(rv_itrs),3])
                rv_itrs = rv_itrs[0:rv_n]
                rv_itrs = np.argsort(frame_data[p.S][rvs[rv_itrs]]-X_)
                for i, rv_itr in enumerate(rv_itrs):
                    frames_data[frame_itr][p.RV_IDs[i]][track_itr] = \
                        frame_data[p.TRACK_ID][rvs[rv_itr]]


            # lv1, lv2, lv3
            #left to right driving in image plane, bot to top lane orders
            lvs = np.nonzero(frame_data[p.LANE_ID] == (lane_id-1))[0] 
            if len(lvs)>0:    
                lv_itrs = np.argsort(np.abs(frame_data[p.S][lvs]-X_))
                lv_n = min([len(lv_itrs),3])
                lv_itrs = lv_itrs[0:lv_n]
                lv_itrs = np.argsort(frame_data[p.S][lvs[lv_itrs]]-X_)
                #pdb.set_trace()
                for i, lv_itr in enumerate(lv_itrs):
                    frames_data[frame_itr][p.LV_IDs[i]][track_itr] = \
                        frame_data[p.TRACK_ID][lvs[lv_itr]]
            
            
    return {'configs': None, 
            'df': None, 
            'tracks_data': None,
            'frames_data': frames_data}



def calc_svs_povl(configs, df_itr,  df_data, tracks_data, frames_data):  
    # pv: preceding vehicle, fv:following vehicle, 
    # rcp/lcp,rfp/lfp: right/left close preceding, right/left far preceding
    # rcf/lcf,rff/lff: right/left close following, right/left far following
    
    for frame_itr, frame_data in enumerate(frames_data):
        for track_itr, track_id in enumerate(frame_data[p.TRACK_ID]):
            X_ = frame_data[p.S][track_itr]
            lane_id = frame_data[p.LANE_ID][track_itr]
            slv_itrs = frame_data[p.LANE_ID] == lane_id
            # pv
            pv_itrs = np.nonzero(\
                np.logical_and(frame_data[p.S]> X_, slv_itrs))[0]
            if len(pv_itrs)>0:
                pv_itr = np.argmin(abs(frame_data[p.S][pv_itrs]- X_))
                frames_data[frame_itr][p.PRECEDING_ID][track_itr] =\
                      frame_data[p.TRACK_ID][pv_itrs[pv_itr]]
            #fv
            fv_itrs = np.nonzero(\
                np.logical_and(frame_data[p.S]< X_, slv_itrs))[0]
            if len(fv_itrs)>0:
                fv_itr = np.argmin(abs(frame_data[p.S][fv_itrs]- X_))
                frames_data[frame_itr][p.FOLLOWING_ID][track_itr] =\
                      frame_data[p.TRACK_ID][fv_itrs[fv_itr]]
            
            # rcp, rfp, rcf, rff
            #left to right driving in image plane, bot to top lane orders 
            
            rv_itrs = frame_data[p.LANE_ID] == (lane_id+1) 
            
            rps = np.nonzero(\
                np.logical_and(frame_data[p.S]>= X_, rv_itrs))[0]
            
            if len(rps)>0:
                rp_itrs = np.argsort(np.abs(frame_data[p.S][rps]-X_))
                rp_n = min([len(rp_itrs),2])
                rp_itrs = rp_itrs[0:rp_n]
                for i, rp_itr in enumerate(rp_itrs):
                    frames_data[frame_itr][p.RP_IDs[i]][track_itr] = \
                        frame_data[p.TRACK_ID][rps[rp_itr]]
                    
            rfs = np.nonzero(\
                np.logical_and(frame_data[p.S]< X_, rv_itrs))[0]
            if len(rfs)>0:
                rf_itrs = np.argsort(np.abs(frame_data[p.S][rfs]-X_))
                rf_n = min([len(rf_itrs),2])
                rf_itrs = rf_itrs[0:rf_n]
                for i, rf_itr in enumerate(rf_itrs):
                    frames_data[frame_itr][p.RF_IDs[i]][track_itr] = \
                        frame_data[p.TRACK_ID][rfs[rf_itr]]
                    
            lv_itrs = frame_data[p.LANE_ID] == (lane_id-1) 
            
            lps = np.nonzero(\
                np.logical_and(frame_data[p.S]>= X_, lv_itrs))[0]
            
            if len(lps)>0:
                lp_itrs = np.argsort(np.abs(frame_data[p.S][lps]-X_))
                lp_n = min([len(lp_itrs),2])
                lp_itrs = lp_itrs[0:lp_n]
                for i, lp_itr in enumerate(lp_itrs):
                    frames_data[frame_itr][p.LP_IDs[i]][track_itr] = \
                        frame_data[p.TRACK_ID][lps[lp_itr]]
                    
            lfs = np.nonzero(\
                np.logical_and(frame_data[p.S]< X_, lv_itrs))[0]
            if len(lfs)>0:
                lf_itrs = np.argsort(np.abs(frame_data[p.S][lfs]-X_))
                lf_n = min([len(lf_itrs),2])
                lf_itrs = lf_itrs[0:lf_n]
                for i, lf_itr in enumerate(lf_itrs):
                    frames_data[frame_itr][p.LF_IDs[i]][track_itr] = \
                        frame_data[p.TRACK_ID][lfs[lf_itr]]
            
    return {'configs': None, 
            'df': None, 
            'tracks_data': None,
            'frames_data': frames_data}





def visualise_tracks(configs,df_itr, df_data, tracks_data = None, frames_data = None):
    with open(configs['dataset']['map_export_dir'], 'rb') as handle:
        lane_marking_dict = pickle.load(handle)
    if p.VISUALISE_FRENET == False:
        lanes = lane_marking_dict['lane_nodes']
        X_ = p.X
        Y_ = p.Y
        circle = True
    else:
        lanes = lane_marking_dict['lane_nodes_frenet']
        X_ = p.S
        Y_ = p.D
        circle = False

    lane_y_max = max([max(lane['l'][:,1]) for lane in lanes])
    lane_y_min = min([min(lane['l'][:,1]) for lane in lanes])
    lane_x_max = max([max(lane['l'][:,0]) for lane in lanes])
    lane_x_min = min([min(lane['l'][:,0]) for lane in lanes])
    
    bias = 10
    
    
    vis_cdir = os.path.join(os.path.join(p.VIS_DIR, configs['dataset']['name']), 'tracks')
    tracks_dir = os.path.join(vis_cdir, time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(tracks_dir):
        os.makedirs(tracks_dir)
        
    frame_itr_list = [frame[p.FRAME][0] for frame in frames_data]
    image_width = lane_x_max - lane_x_min
    image_height = lane_y_max+bias - lane_y_min
    print('image_width: {}, image_height: {}'.format(image_width*p.X_SCALE, image_height*p.Y_SCALE))
    
    background_image = np.ones(( int(image_height*p.Y_SCALE), int(image_width*p.X_SCALE), 3), dtype = np.uint8)
    # change color of background imag to warm white
    background_image[:,:,0] = background_image[:,:,0]*250
    background_image[:,:,1] = background_image[:,:,1]*250
    background_image[:,:,2] = background_image[:,:,2]*255
    #print(background_image.shape)
    x_pos = lambda x: int((x -lane_x_min +bias/2)*p.X_SCALE)
    y_pos = lambda y: int((lane_y_max+bias/2- y)*p.Y_SCALE)
    
    for lane in lanes:
            for itr in range(len(lane['r'])-1):
                cv2.line(background_image, (x_pos(lane['r'][itr,0]), y_pos(lane['r'][itr,1])),(x_pos(lane['r'][itr+1,0]), y_pos(lane['r'][itr+1,1])), (0,0,0), thickness= 3)
            for itr in range(len(lane['l'])-1):
                cv2.line(background_image, (x_pos(lane['l'][itr,0]), y_pos(lane['l'][itr,1])),(x_pos(lane['l'][itr+1,0]), y_pos(lane['l'][itr+1,1])), (0,0,0), thickness= 3)
        #print(i
    #print(int(image_width*p.X_SCALE))
    if not cv2.imwrite(os.path.join(tracks_dir, 'lanes.png'), background_image):
        raise Exception("Could not write image")
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
                frame_itr = frame_itr_list.index(frame)
                frame_data = frames_data[frame_itr]
                image = np.copy(background_image)
                for track_itr, track_id in enumerate(frame_data[p.TRACK_ID]):
                    lane_id = int(frame_data[p.LANE_ID][track_itr])
                    if track_id == tv_id:
                        text = 'TV:{}'.format(lane_id)
                        v_color = (235,206,135)
                    elif track_id in sv_ids[fr_itr]:
                        text = '{}:{}:{}'.format(p.SV_IDS_ABBR[np.argwhere(sv_ids[fr_itr]==track_id)[0][0]],lane_id, int(track_id))
                        v_color = (235,206,135)
                    else:
                        text = 'NV:{}:{}'.format(lane_id,int(track_id))
                        v_color = (210,210,210)
                    image = vf.plot_vehicle(image, 
                                        (x_pos(frame_data[X_][track_itr]), y_pos(frame_data[Y_][track_itr])), 
                                        (frame_data[p.WIDTH][track_itr]*p.X_SCALE,frame_data[p.HEIGHT][track_itr]*p.Y_SCALE),
                                        v_color,
                                        text,
                                        (0,0,0),
                                        circle,
                                        font_scale=p.FONT_SCALE
                                        )
                cv2.imwrite(os.path.join(tracks_dir, 'File{}_TV{}_FR{}.png'.format(df_itr,tv_id, frame)), image)
    
    #cv2.imwrite(os.path.join(tracks_dir, 'lanes.png'), background_image)
    #pdb.set_trace()
    #with open(configs['dataset']['map_export_dir'], 'wb') as handle:
    #     pickle.dump(lane_marking_dict, handle)
    return {'configs': None, 'df': None, 'tracks_data': None,'frames_data': None}


def get_lane_ids(configs, df_itr, df_data, tracks_data = None, frames_data = None):
    # compute p.LANE_ID, p.LANE_WIDTH, p.Y2LANE
    with open(configs['dataset']['map_export_dir'], 'rb') as handle:
        lane_marking_dict = pickle.load(handle)
    lanes = lane_marking_dict['lane_nodes_frenet']
    ids2remove = []
    n_violation = 0
    for id, track_data in enumerate(tracks_data):
        
        lane_id = np.ones((len(track_data[p.S])))*-1
        lane_width = np.ones((len(track_data[p.S])))*-1
        y2lane = np.zeros((len(track_data[p.S])))
        
        for fr in range(len(track_data[p.S])):
            x = track_data[p.S][fr]
            y = track_data[p.D][fr]
            
            # find the itr of closest longitudinal position to the vehicle for each lane.
            l_closest_y = []
            l_closest_itr = []
            for lane_itr, lane in enumerate(lanes):
                l_closest_itr.append(np.argmin(np.abs(lane['l'][:,0]-x)))
                pos_y = lane['l'][l_closest_itr[-1],1]
                l_closest_y.append(pos_y)
            l_closest_y = np.array(l_closest_y)
            
            if np.any(y>l_closest_y) == False: 
                cur_lane_id =  len(lanes) #merge lane
            else:
                cur_lane_id = np.nonzero(y>l_closest_y)[0][0]
                
            if cur_lane_id>0:
                lane_width[fr] = abs(l_closest_y[cur_lane_id-1] - lanes[cur_lane_id-1]['r'][l_closest_itr[cur_lane_id-1],1])
                y2lane[fr] = abs(y-l_closest_y[cur_lane_id-1])
            lane_id[fr] = cur_lane_id
        
        n_lane_violation = np.sum(lane_id==0)
        if n_lane_violation>0 and n_lane_violation<len(lane_id):
            n_violation +=1
            print('Warning: Track goes out of road boundries for some frames, Track:{}, N:{}/{}'.format(id, n_lane_violation, len(lane_id)))
        
        if np.all(lane_id==0): # driving in opposite direction
            ids2remove.append(id)
        else:
            tracks_data[id][p.LANE_ID] = lane_id
            tracks_data[id][p.LANE_WIDTH] = lane_width
            tracks_data[id][p.Y2LANE] = y2lane 
    # Vehicles driving in the other direction
    print('N Violation : {}'.format(n_violation))
    print('Deleting {}/{} tracks (other driving direction)'.format(len(ids2remove), len(tracks_data)))
    for id in sorted(ids2remove, reverse = True):
        del tracks_data[id]
    #pdb.set_trace()  
    return {'configs': None, 'df': None, 'tracks_data': tracks_data,'frames_data': None}

def convert2frenet(configs,df_itr, df_data, tracks_data = None, frames_data = None):
    # compute p.S p.Y
    with open(configs['dataset']['map_export_dir'], 'rb') as handle:
        lane_marking_dict = pickle.load(handle)
    lane_nodes = lane_marking_dict['lane_nodes']
    lane_nodes_f = lane_marking_dict['lane_nodes_frenet']
    merge_frenet_origin = lane_marking_dict['merge_origin_lane']
    main_frenet_origin = lane_marking_dict['main_origin_lane']
    merge_s_bias = lane_marking_dict['merge2main_s_bias']
    for id, track_data in enumerate(tracks_data):
        traj = np.stack((track_data[p.X], track_data[p.Y]), axis = 1)
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
        tracks_data[id][p.S] = traj_frenet[:,0] + x_bias
        tracks_data[id][p.D] = traj_frenet[:,1]
        if p.DEBUG_MODE:
            fig1, ax1 = plt.subplots()
            ax1.plot(tracks_data[id][p.S],tracks_data[id][p.D], '*')
            for i in range(len(lane_nodes)):
                ax1.plot(lane_nodes_f[i]['l'][:,0], lane_nodes_f[i]['l'][:,1])
                ax1.plot(lane_nodes_f[i]['r'][:,0], lane_nodes_f[i]['r'][:,1])
            
            fig2, ax2 = plt.subplots()
            ax2.plot(traj[:,0],traj[:,1], '*')
            for i in range(len(lane_nodes)):
                ax2.plot(lane_nodes[i]['l'][:,0], lane_nodes[i]['l'][:,1])
                ax2.plot(lane_nodes[i]['r'][:,0], lane_nodes[i]['r'][:,1])
                
            fig3, ax3 = plt.subplots()
            ax3.plot(frenet_ref[:,0],frenet_ref[:,1], '*')
            
            fig4, ax4 = plt.subplots()
            ax4.plot(np.arange(0,len(tracks_data[id][p.Y_VELOCITY])),tracks_data[id][p.Y_VELOCITY], '*')
            
            fig5, ax5 = plt.subplots()
            ax5.plot(np.arange(0,len(tracks_data[id][p.Y_VELOCITY])),tracks_data[id][p.X_VELOCITY], '*')
            
            fig6, ax6 = plt.subplots()
            # plot lane_nodes
            for i in range(len(lane_nodes)):
                ax6.plot(lane_nodes[i]['l'][:,0], lane_nodes[i]['l'][:,1], '*')
                ax6.plot(lane_nodes[i]['r'][:,0], lane_nodes[i]['r'][:,1], '*')
            # plot traj
            ax6.plot(tracks_data[id][p.X],tracks_data[id][p.Y], '*')
            #fig6, ax6 = plt.subplots()
            #ax6.plot(frenet_ref[:,0],frenet_ref[:,1], '*')
            
            plt.show()
        if p.DEBUG_MODE and id>3:
            exit()
        #pdb.set_trace()
    return {'configs': None, 'df': None, 'tracks_data': tracks_data,'frames_data': None}



def plot_traj(configs, df_itr, df_data, tracks_data = None, frames_data = None, plot_velocity = True):
  
    
    with open(configs['dataset']['map_export_dir'], 'rb') as handle:
        lane_marking_dict = pickle.load(handle)
    lane_nodes = lane_marking_dict['lane_nodes']
    lane_nodes_f = lane_marking_dict['lane_nodes_frenet']
    fig1, ax1 = plt.subplots()
    #fig2, ax2 = plt.subplots()
    #fig3, ax3 = plt.subplots()
    #fig4, ax4 = plt.subplots()

    for i in range(len(lane_nodes)):
            ax1.plot(lane_nodes_f[i]['l'][:,0], lane_nodes_f[i]['l'][:,1], 'k')
            ax1.plot(lane_nodes_f[i]['r'][:,0], lane_nodes_f[i]['r'][:,1], 'k')
    #    
    #for i in range(len(lane_nodes)):
    #        ax2.plot(lane_nodes[i]['l'][:,0], lane_nodes[i]['l'][:,1], '*')
    #        ax2.plot(lane_nodes[i]['r'][:,0], lane_nodes[i]['r'][:,1], '*')
            
    for id, track_data in enumerate(tracks_data):
        if id>10:
            break
        ax1.plot(tracks_data[id][p.S], tracks_data[id][p.D], '.')
        array_size = len(tracks_data[id][p.S])
        dx = tracks_data[id][p.S_VELOCITY][0]*np.arange(array_size)*configs['dataset']['desired_fps']
        dy = tracks_data[id][p.D_VELOCITY][0]*np.arange(array_size)*configs['dataset']['desired_fps']
        #ax1.plot(tracks_data[id][p.S][0]+dx, tracks_data[id][p.D][0]+dy, '.')
        
        if plot_velocity:
            ax1.quiver(tracks_data[id][p.S], tracks_data[id][p.D], tracks_data[id][p.S_VELOCITY], tracks_data[id][p.D_VELOCITY], width = 0.001, angles='xy', scale_units='xy')
          

        #ax3.plot(np.arange(0,len(tracks_data[id][p.D_VELOCITY])),tracks_data[id][p.D_VELOCITY], '*')
        #ax4.plot(np.arange(0,len(tracks_data[id][p.D_VELOCITY])),tracks_data[id][p.S_VELOCITY], '*')
        

            #fig6, ax6 = plt.subplots()
            #ax6.plot(frenet_ref[:,0],frenet_ref[:,1], '*')
            
    plt.show()
        
    return {'configs': None, 'df': None, 'tracks_data': None,'frames_data': None}


def hdmaps2lane_markings(configs,df_itr, df_data, tracks_data = None, frames_data = None):
    ll2p_yml_dir = configs['dataset']['lane_markings_yml_dir'] # lane markings ways extracted from lanelet2 and their types
    with open(ll2p_yml_dir) as f:
            lm_ways = yaml.load(f, Loader = yaml.SafeLoader)
    lanes_ways = []
    for key in lm_ways:
        lane = {}
        lane['r'] = lm_ways[key]['right']
        lane['l'] = lm_ways[key]['left']
        lane['rt'] = lm_ways[key]['right_type']
        lane['lt'] = lm_ways[key]['left_type']
        lanes_ways.append(lane)
        
    ll2p_file_dir = configs['dataset']['lanelet2_file_dir'] #original map data in lanelet2 format
    xml_tree = ET.parse(ll2p_file_dir)
    root = xml_tree.getroot()
    node_ids = []
    node_poses = []
    ways = {}
    for child in root:
        if child.tag == 'node':
            node_poses.append([float(child.attrib['lon']), float(child.attrib['lat'])])
            node_ids.append(child.attrib['id'])
        elif child.tag == 'way':
            way_nodes = []
            for g_child in child:
                if g_child.tag == 'nd':
                    way_nodes.append(g_child.attrib['ref'])
            ways[child.attrib['id']] = way_nodes
    lanes_nodes = []
    lanes_nodes_types = []
    utmZone = configs['dataset']['UTMZone']
    lonlat2utm = Proj("+proj=utm +zone={} +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(utmZone))
    utmXorig = configs['dataset']['xUtmOrigin']
    utmYorig = configs['dataset']['yUtmOrigin']  
    node_poses = np.array(node_poses)
    utmX, utmY = lonlat2utm(node_poses[:,0], node_poses[:,1])
    node_poses = np.stack((utmX - utmXorig, utmY - utmYorig), axis = 1)
    
    for lane_ways in lanes_ways:
        lane_nodes = {}
        lane_nodes_types = {}
        n_ways = len(lane_ways['r'])
        
        r_nodes = []
        l_nodes = []
        
        r_nodes_type = []
        l_nodes_type = []
        for way_itr in range(n_ways):
            # Right lane marking
            r_way = lane_ways['r'][way_itr]
            r_node_ids = ways[str(r_way)]
            
            r_nodes_xy = []
            for r_node_id in r_node_ids:
                node_itr = node_ids.index(r_node_id)
                r_nodes_xy.append(node_poses[node_itr])
            r_nodes_xy = np.array(r_nodes_xy)    
            # Intrapolate
            r_nodes_type.extend([lane_ways['rt'][way_itr]]*len(r_nodes_xy))
            r_nodes.append(r_nodes_xy)
        
        for way_itr in range(n_ways):
            # Left Lane Marking
            l_way = lane_ways['l'][way_itr]
            l_node_ids = ways[str(l_way)]
            
            l_nodes_xy = []
            for l_node_id in l_node_ids:
                node_itr = node_ids.index(l_node_id)
                l_nodes_xy.append(node_poses[node_itr])
            l_nodes_xy = np.array(l_nodes_xy)    

            l_nodes_type.extend([lane_ways['lt'][way_itr]]*len(l_nodes_xy))
            l_nodes.append(l_nodes_xy)
        
            
            '''
                #Intrapolate
            x = r_nodes_xy[:,0]
            y = r_nodes_xy[:,1]
            interpolate_fn = interp1d(x,y)
            new_x = x #np.linspace(x[0], x[-1], 100)
            new_y = y #interpolate_fn(new_x)
            r_nodes_xy_n = np.stack((new_x, new_y), axis = 1)
            for i in range(len(r_nodes_xy_n)):
                r_nodes.append([float(r_nodes_xy_n[i,0]), float(r_nodes_xy_n[i,1])])
            '''
        r_nodes_arr = np.concatenate(r_nodes, axis = 0)
        l_nodes_arr = np.concatenate(l_nodes, axis = 0)
        r_nodes_type = np.array(r_nodes_type)
        l_nodes_type = np.array(l_nodes_type)
        assert(len(r_nodes_arr)== len(r_nodes_type))
        assert(len(l_nodes_arr)== len(l_nodes_type))
        

        new_u = np.linspace(0,1, 5000)
        
        r_tck, r_u, r_duplicate = parametrise_polyline(r_nodes_arr)
        if r_duplicate:
            r_nodes_type = np.delete(r_nodes_type, r_duplicate, axis = 0)
        
        r_u_type = np.zeros_like(new_u)
        ru_ind = 0
        for i, u in enumerate(new_u):
            if u > r_u[ru_ind]:
                ru_ind+=1
            r_u_type[i] = r_nodes_type[ru_ind]
            


        r_nodes = interpolate_polyline(r_tck, new_u)
        r_nodes_type = r_u_type
        
        l_tck, l_u, l_duplicate = parametrise_polyline(l_nodes_arr)
        if l_duplicate:
            l_nodes_type = np.delete(l_nodes_type, l_duplicate, axis = 0)
        
        l_u_type = np.zeros_like(new_u)
        lu_ind = 0
        for i, u in enumerate(new_u):
            if u > l_u[lu_ind]:
                lu_ind+=1
            l_u_type[i] = l_nodes_type[lu_ind]
            
        l_nodes = interpolate_polyline(l_tck, new_u)
        l_nodes_type = l_u_type
        
        
        lane_nodes['r'] = r_nodes#cf.longlat2xy(np.array(r_nodes), longlat_origin)
        lane_nodes['l'] = l_nodes#cf.longlat2xy(np.array(l_nodes), longlat_origin)
        lane_nodes_types['r'] = r_nodes_type
        lane_nodes_types['l'] = l_nodes_type
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
    
    # lanes_nodes=>  array of dict, each dict is the nodes of right and left lane marking of lane i in array.
    n_lanes = len(lanes_nodes)
    
    merge_frenet_lm = lanes_nodes[-1]['l'] # -2 is index of low-speed main road
    main_frenet_lm = lanes_nodes[-2]['r'] # -1 is index of merging lane
    
    merge2main_node = merge_frenet_lm[-1]
    merge2main_itr = np.nonzero(main_frenet_lm==merge2main_node)[0]
    #assert(np.all(merge2main_itr == merge2main_itr[0]))
    merge2main_itr = merge2main_itr[0]
    lanes_nodes_frenet = []
    main_matched_point_s = cf.cart2frenet(main_frenet_lm, main_frenet_lm)[merge2main_itr,0]

        
    # convert 2 frenet
    for itr, lane_nodes in enumerate(lanes_nodes):
        if itr == (len(lanes_nodes)-1):
            break
        lane_nodes_frenet = {}
        l_nodes_frenet = cf.cart2frenet(lane_nodes['l'], main_frenet_lm)
        r_nodes_frenet = cf.cart2frenet(lane_nodes['r'], main_frenet_lm)
    
        lane_nodes_frenet['r'] = r_nodes_frenet
        lane_nodes_frenet['l'] = l_nodes_frenet
        lanes_nodes_frenet.append(lane_nodes_frenet)

    lane_nodes_frenet = {}
    lane_nodes = lanes_nodes[-1]
    l_nodes_frenet = cf.cart2frenet(lane_nodes['l'], merge_frenet_lm)
    r_nodes_frenet = cf.cart2frenet(lane_nodes['r'], merge_frenet_lm)
    merging_matched_point_s = l_nodes_frenet[-1,0]
    l_nodes_frenet[:,0] += main_matched_point_s - merging_matched_point_s
    r_nodes_frenet[:,0] += main_matched_point_s - merging_matched_point_s
    lane_nodes_frenet['r'] = r_nodes_frenet
    lane_nodes_frenet['l'] = l_nodes_frenet
    lanes_nodes_frenet.append(lane_nodes_frenet)

    # transform lane markigns in frenet to image coordinate (top left of image is the origin with y axis direction down and x axis direction to right)
    # assumption: driving dir = 2
    # assumption lane r x is not less/greater than lane l x
    #lane_y_max = max([max(lane['l'][:,1]) for lane in lane_nodes_frenet])
    #lane_y_min = min([min(lane['r'][:,1]) for lane in lane_nodes_frenet])
    #lane_x_max = max([max(lane['l'][:,0]) for lane in lane_nodes_frenet])
    #lane_x_min = min([min(lane['l'][:,0]) for lane in lane_nodes_frenet])
    
    # interpolate lane markings 
    '''
    for itr, lane_nodes_f in enumerate(lanes_nodes_frenet):
        s = lane_nodes_f['r'][:,0]
        d = lane_nodes_f['r'][:,1]
        interpolate_fn = interp1d(s,d)
        new_s = np.arange(min(s), max(s), 0.1)
        new_d = interpolate_fn(new_s)
        new_traj = np.stack((new_s, new_d), axis = 1)f
        lanes_nodes_frenet[itr]['r'] = new_traj

        s = lane_nodes_f['l'][:,0]
        d = lane_nodes_f['l'][:,1]
        interpolate_fn = interp1d(s,d)
        new_s = np.arange(min(s), max(s), 0.1)
        new_d = interpolate_fn(new_s)
        new_traj = np.stack((new_s, new_d), axis = 1)
        lanes_nodes_frenet[itr]['l'] = new_traj
    '''
    #Extroplolate lane markings for 400 meters
    for itr, lane_nodes_f in enumerate(lanes_nodes_frenet):
        lane_type = lanes_nodes_types[itr]['r'][-1]
        s = lane_nodes_f['r'][:,0]
        d = lane_nodes_f['r'][:,1]
        s_ext = np.arange(s[-1]+10, s[-1]+300, 0.1)
        d_ext = np.ones_like(s_ext)*d[-1]
        sd_ext = np.stack((s_ext, d_ext), axis=  1)
        
        t_ext = np.ones_like(s_ext)*lane_type
        lanes_nodes_frenet[itr]['r'] = np.append(lanes_nodes_frenet[itr]['r'], sd_ext, axis = 0)
        lanes_nodes_types[itr]['r'] = np.append(lanes_nodes_types[itr]['r'], t_ext, axis = 0)
        
        lane_type = lanes_nodes_types[itr]['l'][-1]
        s = lane_nodes_f['l'][:,0]
        d = lane_nodes_f['l'][:,1]
        s_ext = np.arange(s[-1]+10, s[-1]+300, 0.1)
        d_ext = np.ones_like(s_ext)*d[-1]
        sd_ext = np.stack((s_ext, d_ext), axis=  1)
        
        t_ext = np.ones_like(s_ext)*lane_type
        lanes_nodes_frenet[itr]['l'] = np.append(lanes_nodes_frenet[itr]['l'], sd_ext, axis = 0)
        lanes_nodes_types[itr]['l'] = np.append(lanes_nodes_types[itr]['l'], t_ext, axis = 0)
        


    lane_y_max = max([max(lane['l'][:,1]) for lane in lanes_nodes_frenet])
    lane_y_min = min([min(lane['r'][:,1]) for lane in lanes_nodes_frenet])
    lane_x_max = max([max(lane['l'][:,0]) for lane in lanes_nodes_frenet])
    lane_x_min = min([min(lane['l'][:,0]) for lane in lanes_nodes_frenet])
    image_width = lane_x_max - lane_x_min
    image_height = lane_y_max- lane_y_min
    #print(lane_markings_frenets)
    lane_marking_dict = {}
    lane_marking_dict['image_width'] = image_width
    lane_marking_dict['image_height'] = image_height
    
    lane_marking_dict['lane_types'] = lanes_nodes_types
    lane_marking_dict['lane_nodes'] = lanes_nodes
    
    lane_marking_dict['lane_nodes_frenet'] = lanes_nodes_frenet
    lane_marking_dict['merge_origin_lane'] = merge_frenet_lm
    lane_marking_dict['main_origin_lane'] = main_frenet_lm
    lane_marking_dict['merge2main_s_bias'] = main_matched_point_s - merging_matched_point_s
    lane_marking_dict['driving_dir'] = 2 #1: right to left in image plane, 2: left to right in image plane    
    with open(configs['dataset']['map_export_dir'], 'wb') as handle:
        pickle.dump(lane_marking_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    return {'configs': configs, 'df': None, 'tracks_data': None,'frames_data': None}



