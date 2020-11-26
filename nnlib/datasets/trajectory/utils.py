import math
import numpy as np

def l2_distance_to_trajectory_set(traj, traj_set):
    num_set, num_points = traj_set.shape[:2]
    traj = traj[:num_points,:2]
    min_dist = 1e5
    min_idx = 0
    for i, traj_ref in enumerate(traj_set):
        _,_,metric = interp_l2_distance(traj, traj_ref)
        d = metric['ADE']
        if d < min_dist:
            min_dist = d
            min_idx = i
    return min_idx, min_dist
    
def angle_to_trajectory_set(traj, traj_set):
    num_set, num_points = traj_set.shape[:2]
    traj = traj[:num_points,:2]
    angles = []
    for traj_ref in traj_set:
        angle = 0
        for i in range(num_points):
            angle += angle_between(traj[i,:], traj_ref[i,:])
        angles.append(angle / num_points)
    idx = np.argmin(np.array(angles))
    return idx, angles[idx]

def angle_between(v1, v2):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if math.isclose(norm, 0):
        return 0

    dot_product = v1.dot(v2)
    angle = math.degrees(math.acos(max(min(dot_product / norm, 1), -1)))
    if angle == 180:
        return angle - 1e-5

    return angle

def l2_distance(traj, traj_ref):
    metric = {}
    d = np.linalg.norm(traj - traj_ref, axis=1)
    metric['ADE'] = np.mean(d)
    metric['FDE'] = d[-1]
    metric['MAX'] = np.max(d)
    return metric

def interp_l2_distance(traj, traj_ref, samples=20):
    x1, y1 = traj_ref[-1,:2]
    if abs(y1) > abs(x1):
        ymin = np.min(traj_ref[:,1])
        ymax = np.max(traj_ref[:,1])
        yvals = np.linspace(ymin, ymax, samples)
        xinterp = np.interp(yvals, traj[:,1], traj[:,0])
        xinterp_ref = np.interp(yvals, traj_ref[:,1], traj_ref[:,0])
        traj_interp = np.vstack([xinterp, yvals]).T
        traj_interp_ref = np.vstack([xinterp_ref, yvals]).T
    else:
        xmin = np.min(traj_ref[:,0])
        xmax = np.max(traj_ref[:,0])
        xvals = np.linspace(xmin, xmax, samples)
        yinterp = np.interp(xvals, traj[:,0], traj[:,1])
        yinterp_ref = np.interp(xvals, traj_ref[:,0], traj_ref[:,1])
        traj_interp = np.vstack([xvals, yinterp]).T
        traj_interp_ref = np.vstack([xvals, yinterp_ref]).T
    
    metric = l2_distance(traj_interp, traj_interp_ref)
    return traj_interp, traj_interp_ref, metric
