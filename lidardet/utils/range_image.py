import numpy as np

class RangeImage(object):

    def __init__(self, cfg = {}):
        self.cfg = cfg
        self.max_range = cfg.get('max_range', 80.0)
        self.min_range = cfg.get('min_range', 2.0)
        self.fov_up = cfg.get('fov_up', 3.0) / 180.0 * np.pi
        self.fov_down = cfg.get('fov_down', -25.0) / 180.0 * np.pi
       
        # get vertical field of view total in radians
        self.fov = abs(self.fov_down) + abs(self.fov_up)  

        self.cols = cfg.get('cols', 1024) # 1800
        self.rows = cfg.get('rows', 64)

    def from_points(self, points_in, normalize=True):
        """
         Project points to range image
        """
        points = np.copy(points_in)
        # get depth of all points
        depth = np.linalg.norm(points[:, :3], 2, axis=1)
        # filter by range limit
        points = points[(depth > self.min_range) & (depth < self.max_range)]
        depth = depth[(depth > self.min_range) & (depth < self.max_range)]
       
        # extract x, y, z and intensity values
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        t = points[:,3]

        # get horizontal and vertical angles [radian]
        yaw = -np.arctan2(y, x) # [-pi, pi]
        pitch = np.arcsin(z / depth)
        
        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov  # in [0.0, 1.0]
        
        # scale to image size using angular resolution
        proj_x *= self.cols  # in [0.0, cols]
        proj_y *= self.rows  # in [0.0, rows]
        
        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.cols - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.rows - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        
        # sort depth in ascending order to keep more far distance points
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)
        depth = depth[order]
        indices = indices[order]
        points = points[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        proj_range = np.zeros((self.rows, self.cols, 5), dtype=np.float32)  # [H,W] range (0 is no data)

        if normalize:
            depth /= self.max_range

        proj_range[proj_y, proj_x, 0] = depth
        proj_range[proj_y, proj_x, 1] = points[:,3]
        proj_range[proj_y, proj_x, 2] = points[:,0]
        proj_range[proj_y, proj_x, 3] = points[:,1]
        proj_range[proj_y, proj_x, 4] = points[:,2]
        return proj_range
    
    def to_points(self, img_in, denormalize=True):
        img = np.copy(img_in)
        proj_y = np.float32(np.arange(self.rows)) / self.rows
        proj_x = np.float32(np.arange(self.cols)) / self.cols

        v_angles = (1.0 - proj_y) * self.fov - abs(self.fov_down)
        h_angles = (2 * proj_x - 1.0) * np.pi

        points = []
        coordinates = []

        if denormalize:
            img[:,:,0] *= self.max_range

        for i in range(self.rows):
            for j in range(self.cols):
                depth = img[i, j, 0]
                intensity = 0
                if img.shape[2] >= 2:
                    intensity = img[i, j, 1]

                if depth < self.min_range:
                    continue
                
                h_angle = h_angles[j]
                v_angle = v_angles[i]
                x = np.sin(h_angle) * np.cos(v_angle) * depth
                y = np.cos(h_angle) * np.cos(v_angle) * depth
                z = np.sin(v_angle) * depth
                point = np.array([x, y, z, intensity]).astype(np.float32)
                points.append(point)
                coordinates.append(np.array([i, j]).astype(np.int32))

        return np.array(points), np.array(coordinates) 
