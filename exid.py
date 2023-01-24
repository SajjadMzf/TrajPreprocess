import numpy as np
import os
import xml.etree.ElementTree as ET
import params as p 
import pdb
import yaml
def hdmaps2lane_markings(configs,itr, df_data, tracks_data = None, frames_data = None):
    ll2p_yml_dir = configs['meta_data']['lane_markings_yml_dir']
    with open(ll2p_yml_dir) as f:
            lm_nodes = yaml.load(f, Loader = yaml.SafeLoader)
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
            
    
    pdb.set_trace()
    a = 2
    return {'configs': configs, 'df': None, 'tracks_data': None,'frames_data': None}