import numpy as np
import scipy.interpolate as interp

def digital_filter(x, symmetric_coef, normalisation_factor, smoothing = False):
    y = np.zeros_like(x)
    if len(x)<len(symmetric_coef):
        return y
    past_future_dep = int(len(symmetric_coef)/2)
    filter_index = np.arange(-1*past_future_dep,past_future_dep+1)
    for i in range(past_future_dep, len(x)-past_future_dep):
        for j in range(len(symmetric_coef)):
            y[i] += symmetric_coef[j]*x[i+filter_index[j]]
    
    y   = y/normalisation_factor
    
    if smoothing:
        y[0:past_future_dep] = x[0:past_future_dep]
        y[len(y)-past_future_dep:] = x[len(y)-past_future_dep:]
    else:
        y[0:past_future_dep] = y[past_future_dep]
        y[len(y)-past_future_dep:] = y[len(y)-past_future_dep-1]
    return y

def parametrise_polyline(polyline):
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i-1]):
            duplicates.append(i)
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    return tck, u, duplicates
def interpolate_polyline(tck, u):
    return np.column_stack(interp.splev(u, tck))