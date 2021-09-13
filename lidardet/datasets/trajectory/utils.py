import math
import random
import numpy as np
from scipy.spatial.distance import directed_hausdorff

from lidardet.utils import geometry

def hausdorff_dist(traj1, traj2):
    """
    Double side hausdorff distance
    """
    dist1 = directed_hausdorff(traj1, traj2)[0]
    dist2 = directed_hausdorff(traj2, traj1)[0]
    return max(dist1, dist2)

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

def random_shift(traj_list, res, max_dist=1.0):
    rx = max_dist * 2 * (random.random() - 0.5) # [-1.0, 1.0)
    #print('rx1', rx)
    rx = rx / res
    for i in range(len(traj_list)):
        traj_list[i][:,0] += rx
    #print(rx)
    return traj_list

def random_flip(points, traj_list):
    """
    Random flip along y axis
    Args:
        points: (M, 4)
        traj_list: list of trajectories with shape of (N, 2)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 0] = -points[:, 0]
        for i in range(len(traj_list)):
            traj_list[i][:, 0] = - traj_list[i][:, 0]
    return points, traj_list

def random_rotate(points, traj_list, rot_range=[0, np.pi/4]):
    """
    Random rotate along z axis
    Args:
        points: (M, 4)
        traj_list: list of trajectories with shape of (N, 2)
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = geometry.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    for i, traj in enumerate(traj_list):
        traj_points = np.zeros((1, traj.shape[0], 3))
        traj_points[0,:,:2] = traj[:,:2]
        traj_list[i][:,:2] = geometry.rotate_points_along_z(traj_points, np.array([noise_rotation]))[0][:,:2]
    return points, traj_list

def random_drop(points, drop_ratio=0.5):
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        np.random.shuffle(points)
        points = points[:int(points.shape[0] * drop_ratio), :]
    return points

class HeatMapGenerator():
    def __init__(self, img_w, img_h, scale = 1.0):
        self.img_w = int(img_w * scale)
        self.img_h = int(img_h * scale)
        sigma = min(self.img_h, self.img_w) / 64.0
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.sigma = sigma

    def __call__(self, points):
        hms = np.zeros((points.shape[0], self.img_h, self.img_w), dtype=np.float32)
        sigma = self.sigma
        points[:,0] *= self.img_w
        points[:,1] *= self.img_h
        for idx, p in enumerate(points):
            x, y = int(p[0]), int(p[1])
            ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
            br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

            c,d = max(0, -ul[0]), min(br[0], self.img_w) - ul[0]
            a,b = max(0, -ul[1]), min(br[1], self.img_h) - ul[1]

            cc,dd = max(0, ul[0]), min(br[0], self.img_w)
            aa,bb = max(0, ul[1]), min(br[1], self.img_h)
            hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms
