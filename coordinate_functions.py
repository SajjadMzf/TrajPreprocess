import numpy as np
from numpy.linalg import norm
import math 

def cart2frenet(traj, ref):
    '''
    traj = np array of size [T,2]
    ref = np array of size [L,2]
    '''
    L = ref.shape[0]
    T = traj.shape[0]
    gamma = np.zeros((L))
    for i in range(L-1):
        gamma[i] = norm(ref[i+1]-ref[i])
    gamma[L-1] = gamma[L-2]
    gamma = np.cumsum(gamma)
    ref_frenet = np.zeros((L, 2))
    ref_frenet[:,0] = gamma
    traj_frenet = np.zeros((T, 2))
    for i in range(T):
        min2itr = np.argpartition(norm(ref-traj[i], axis=1), 3)[0:2]
        it = np.min(min2itr)
        it1 = np.max(min2itr)
        traj_frenet[i,0] =  gamma[it] + norm(np.dot(ref[it1]-ref[it], ref[it]-traj[i]))/norm(ref[it1]-ref[it])
        traj_frenet[i,1] = norm(np.cross(ref[it1]-ref[it], ref[it]-traj[i]))/norm(ref[it1]-ref[it])
    return traj_frenet, ref_frenet

def asRadians(degrees):
    return degrees * math.pi / 180

def longlat2xy(data_coordinates, null_coordinates):
    """ Calculates X and Y distances in meters.
    """
    (data_long, data_lat) = data_coordinates
    (null_long, null_lat) = null_coordinates
    deltaLatitude = data_lat - null_lat
    deltaLongitude = data_long - null_long
    latitudeCircumference = 40075160 * math.cos(asRadians(null_lat))
    resultX = deltaLongitude * latitudeCircumference / 360
    resultY = deltaLatitude * 40008000 / 360
    
    return (resultX, resultY) 