import numpy as np
from numpy.linalg import norm
import math 
import sys

def frenet2cart(traj, ref):

    return 0

def cart2frenet(traj, ref):
    '''
    traj = np array of size [T,2]
    ref = np array of size [L,2]
    '''
    #print('CART2FRENET') TODO: 1. extend gamma 2. test with crossing traj and ref
    L = ref.shape[0]
    T = traj.shape[0]
    gamma = np.zeros((L)) 
    for i in range(1, L):
        gamma[i] = norm(ref[i]-ref[i-1])
    gamma = np.cumsum(gamma)
    ref_frenet = np.zeros((L, 2))
    ref_frenet[:,0] = gamma
    traj_frenet = np.zeros((T, 2))
    for i in range(T):
        traj2ref_dist = norm(ref-traj[i], axis=1)     
        itrs = list(np.argpartition(traj2ref_dist, 1)[0:2])
        itrs.sort()
        itr1 = itrs[0]
        itr2 = itrs[1]
        traj_frenet[i,0] =  gamma[itr1] + np.dot(ref[itr2]-ref[itr1], traj[i]-ref[itr1])/norm(ref[itr2]-ref[itr1])
        traj_frenet[i,1] = np.cross(ref[itr2] - ref[itr1], traj[i] - ref[itr1])/norm(ref[itr2]-ref[itr1])
        #print('it:{}, it1:{}'.format(it,it1))
    return traj_frenet


def frenet2cart( traj, ref):
        #print('FRENET2CART')
        epsilon=sys.float_info.epsilon
        L = ref.shape[0]
        T = traj.shape[0]
        cart_traj = np.zeros_like(traj)
        gamma = np.zeros((L))
        for i in range(L-1):
            gamma[i] = norm(ref[i+1]-ref[i])
        gamma[L-1] = gamma[L-2]
        gamma = np.cumsum(gamma)
        traj_cart = np.zeros((T,2))
        #assert(np.any(gamma>traj))
        for i in range(T):
            it2 = np.nonzero(gamma>traj[i,0])[0][0]
            it1 = it2-1
            assert(it1>=0)             

            thetha1 = np.arctan((ref[it2,1]-ref[it1,1])/(ref[it2,0]-ref[it1,0]+epsilon))
            
            thetha = np.arctan((traj[i,1])/(traj[i,0]-gamma[it1]+epsilon))
            
            thetha_cart = thetha1+thetha
            dist2origin = np.sqrt(np.power(traj[i,1], 2) + np.power((traj[i,0]- gamma[it1]), 2))
            #assert(np.sin(thetha_cart)>0)
            #assert(np.cos(thetha_cart)>0)
            cart_traj[i,0] = dist2origin * np.cos(thetha_cart) + ref[it1, 0]
            cart_traj[i,1] = dist2origin * np.sin(thetha_cart) + ref[it1, 1]
            #print('it1:{}, it2:{}, theta:{}, theta1:{},{},{},{}'.format(it1,it2,thetha*180/np.pi,thetha1*180/np.pi, (np.abs(traj[i,1]))/(np.abs(traj[i,0]- gamma[it1])+epsilon),np.abs(traj[i,1]),thetha_cart*180/np.pi) )
        return cart_traj

def asRadians(degrees):
    return degrees * math.pi / 180

def longlat2xy(data_coordinates, origin_coordinates):
    """ Calculates X and Y distances in meters.
    """
    
    long_array = data_coordinates[:,0]
    lat_array = data_coordinates[:,1]
    #(data_long, data_lat) = data_coordinates
    (long_origin, lat_origin) = origin_coordinates
    deltaLatitude = lat_array - lat_origin
    deltaLongitude = long_array - long_origin
    latitudeCircumference = 40075160 * math.cos(asRadians(lat_origin))
    resultX = deltaLongitude * latitudeCircumference / 360
    resultY = deltaLatitude * 40008000 / 360
    xy = np.stack((resultX, resultY),axis = 1)
    return xy 

def point2lane_dist(p, l1,l2):
    x = p-l1
    y = l2-l1
    return np.abs(np.cross(x,y)/np.linalg.norm(y))
