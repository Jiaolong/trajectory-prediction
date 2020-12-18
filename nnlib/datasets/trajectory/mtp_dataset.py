import cv2
import math
import json
import pickle
import random
import numpy as np
from pathlib import Path
from matplotlib import cm
from torchvision.transforms import ToTensor

from nnlib.datasets.trajectory import bev
from nnlib.datasets.registry import DATASETS
from nnlib.datasets.base import PointCloudDataset

from .utils import interp_l2_distance

@DATASETS.register
class MTPDataset(PointCloudDataset):

    def __init__(self, cfg, logger=None):
        super(MTPDataset, self).__init__(cfg=cfg, logger=logger)
        
        assert(self.root_path.exists())
        
        self.label_files = []
        for subdir in cfg.subset: 
            data_dir = self.root_path / subdir
            assert(data_dir.exists())
            fs = [p for p in data_dir.glob('*.json')]
            self.label_files.extend(fs)

        self.label_files.sort()
        
        self.res = cfg.voxel_size[0] # 0.16 m / pixel
        self.road_width = cfg.road_width # meters
        xmin, ymin, zmin, xmax, ymax, zmax = cfg.point_cloud_range
        self.lidar_range_np = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        self.lidar_range = {'Left': xmin, 'Right': xmax, 'Front': ymax, 'Back': ymin, 'Bottom': zmin, 'Top': zmax}
        self.map_h = self.lidar_range['Front'] - self.lidar_range['Back']
        self.map_w = self.lidar_range['Right'] - self.lidar_range['Left']

        self.img_h = int(self.map_h / self.res) # 250
        self.img_w = int(self.map_w / self.res) # 400

        self.trajectory_colors = cm.get_cmap('viridis', 10)
        self.image_transform = ToTensor()
       
        if self.logger:
            self.logger.info('Loading TrajectoryPrediction {} dataset with {} samples'.format(cfg.mode, len(self.label_files)))
       
    def __len__(self):
        return len(self.label_files)
    
    def car2img(self, car_point):
        """
        Convert coordinates from ego-car to image space
        """

        cx, cy = car_point
        px = (cx - self.lidar_range['Left']) / self.res;
        py = (self.lidar_range['Front'] - cy) / self.res;
        return np.array([int(px), int(py)], np.int32)
    
    def lidar_bev_image(self, points):
        if self.cfg.lidar_intensity_max == 255:
            points[:,3] = points[:,3] / 255.0

        bev_map = bev.create_bev(points, self.lidar_range_np, self.img_h, self.img_w, self.res)
        return bev_map

    def generate_image(self, trajectory, rescale=1.0):
         
        thickness = int(rescale * self.road_width / self.res) # 16 pixels
        img = np.zeros((int(self.img_h * rescale), int(self.img_w * rescale), 1)).astype(np.uint8)
        for i in range(trajectory.shape[0] - 1):
            x1, y1 = self.car2img(trajectory[i][:2])
            x2, y2 = self.car2img(trajectory[i+1][:2])
                        
            #print(x1, y1, x2, y2)
            x1 = int(x1 * rescale)
            y1 = int(y1 * rescale)
            x2 = int(x2 * rescale)
            y2 = int(y2 * rescale)
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), thickness)

        return img

    def generate_prediction_dicts(self, batch_dict, pred_dicts, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_np: (B, M), numpy array
            output_path:

        Returns:
            result_dicts: Dict of evaluated metrics
        """
        result_dicts = {}
        
        bs = batch_dict['frame_id'].shape[0]
        num_modes = min(pred_dicts['pred_traj'][0].shape[0], 25)
        ADE = np.zeros((bs, num_modes))
        FDE = np.zeros((bs,))
        hit_rate = np.zeros((bs, num_modes))

        for index, pred_seg in enumerate(pred_dicts['pred_seg']):
            frame_id = batch_dict['frame_id'][index]
            traj_pred_all = pred_dicts['pred_traj'][index][:num_modes] # [num_modes, num_points, 2]

            label_file = Path(self.label_files[frame_id])     
            # load json
            with open(str(label_file)) as f:
                data = json.load(f)
             
            lidar_path = str(label_file).replace('json', 'bin')
            points = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 4])
            if self.cfg.use_lidar_bev:
                lidar_bev = self.lidar_bev_image(points)
       
            traj_hmi = np.array(data['trajectory_hmi'])
            traj_ins = np.array(data['trajectory_ins'])
            traj_ins_past = np.array(data['trajectory_ins_past'])

            assert traj_ins_past.shape[0] >= 2
            dx = traj_ins_past[0,0] - traj_ins_past[1,0] 
            dy = traj_ins_past[0,1] - traj_ins_past[1,1]
            theta = np.arctan2(dx, dy)
            # constant velocity & yaw model
            traj_const = []
            r = 2.0 # radius
            for i in range(self.cfg.num_points_per_trajectory):
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                if i > 0:
                    x1, y1 = traj_const[i - 1]
                    x += x1
                    y += y1
                traj_const.append(np.array([x, y]))
            traj_const = np.array(traj_const)

            assert traj_ins.shape[0] >= self.cfg.num_points_per_trajectory
            traj_ins = traj_ins[:self.cfg.num_points_per_trajectory,:2]
            
            assert traj_hmi.shape[0] >= self.cfg.num_points_per_trajectory
            traj_hmi = traj_hmi[:self.cfg.num_points_per_trajectory,:2]
           
            if self.cfg.normalize_trajectory:
                traj_pred_all *= (self.map_w / 2)
            for m in range(traj_pred_all.shape[0]):
                #traj_pred_all[m] = traj_const 
                traj_pred_interp, traj_ins_interp, metric = interp_l2_distance(traj_pred_all[m], traj_ins)
                if m == 0:
                    FDE[index] = metric['FDE']
                
                ADE[index, m] = metric['ADE']
                hit_rate[index, m] = 1 if metric['MAX'] < 2 else 0

                ADE[index, m] = np.min(ADE[index, :m+1])
                hit_rate[index, m] = np.max(hit_rate[index, :m+1])


            if output_path is not None:
                fname = label_file.stem
                save_dir = output_path / 'predictions' 
                save_dir.mkdir(parents=True, exist_ok=True)
                save_file = save_dir / ('%s.jpg' % fname)
                #print('Saving result {}'.format(save_file))
                
                img_bev = (lidar_bev[:,:,-1] * 255).astype(np.uint8)

                pred_seg *= 255
                pred_seg = cv2.resize(pred_seg, (int(self.img_w), int(self.img_h)))
                w = self.img_w
                blend = (img_bev * 0.5 + pred_seg * 0.8).astype(np.uint8)
                
                img_traj = np.zeros((self.img_h, self.img_w, 3), np.uint8)
                img_traj[...,0] = blend
                img_traj[...,1] = blend
                img_traj[...,2] = blend
                img_traj = self.plot_trajectory_set([traj_ins, traj_pred_all[0], traj_hmi], 
                        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)], img=img_traj)

                cv2.imwrite(str(save_file), img_traj)

        #print('ADE', ADE)
        #print('FDE', FDE)
        #print('HitRate', hit_rate)
        result_dicts['ADE'] = ADE
        result_dicts['FDE'] = FDE
        result_dicts['HitRate'] = hit_rate
        return result_dicts

    def evaluation(self, result_dicts_list):
        self.logger.info('Evaluating on {}'.format(self.cfg.subset))
        for i, result_dicts in enumerate(result_dicts_list):
            ADE_i = result_dicts['ADE']
            FDE_i = result_dicts['FDE']
            hit_rate_i = result_dicts['HitRate']
            if i == 0:
                ADE = ADE_i
                FDE = FDE_i
                hit_rate = hit_rate_i
            else:
                ADE = np.concatenate((ADE, ADE_i), axis=0)
                FDE = np.concatenate((FDE, FDE_i), axis=0)
                hit_rate = np.concatenate((hit_rate, hit_rate_i), axis=0)
        
        self.logger.info('FDE =  {}'.format(np.mean(FDE)))
        for i in range(ADE.shape[1]):
            self.logger.info('minADE@{} = {}'.format(i, np.mean(ADE[:,i])))

        for i in range(hit_rate.shape[1]):
            self.logger.info('HitRate@{} = {}'.format(i, np.sum(hit_rate[:,i])/hit_rate.shape[0]))

    def plot_trajectory_set(self, trajectory_set, colors=[], img=None, thickness=3):
        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3)).astype(np.uint8)
        
        for t, trajectory in enumerate(trajectory_set):
            if len(colors) > 0:
                color = colors[t]
            else:
                color = [int(255 * c) for c in self.trajectory_colors(1.0 * t / len(trajectory_set))]
            for i in range(trajectory.shape[0] - 1):
                x1, y1 = self.car2img(trajectory[i][:2])
                x2, y2 = self.car2img(trajectory[i+1][:2])
                           
                #if t == 0:
                #    cv2.circle(img, (x1, y1), 2, color, thickness)
                #else:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        return img

    def random_shift(self, traj_in, max_dist=1.0):
        rx = max_dist * 2 * (random.random() - 0.5) # [-1.0, 1.0)
        #print('rx1', rx)
        rx = rx / self.res
        traj_out = traj_in
        traj_out[:,0] += rx
        #print(rx)
        return traj_out

    def __getitem__(self, idx):
        label_file = Path(self.label_files[idx]) 
        self.curr_file = label_file
        # load json
        with open(str(label_file)) as f:
            data = json.load(f)

        lidar_path = str(label_file).replace('json', 'bin')
        points = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 4])

        if self.cfg.use_lidar_bev:
            lidar_bev = self.lidar_bev_image(points)

        traj_ins = np.array(data['trajectory_ins'])
        traj_hmi = np.array(data['trajectory_hmi'])
        
        traj_ins_past = np.array(data['trajectory_ins_past'])
        traj_hmi_past = np.array(data['trajectory_hmi_past'])
        
        traj_ins_all = np.vstack((traj_ins_past[::-1], traj_ins)) 
        traj_hmi_all = np.vstack((traj_hmi_past[::-1], traj_hmi)) 

        # random pertubate hmi trajectory
        if self.cfg.mode == 'train':
            traj_hmi_all = self.random_shift(traj_hmi_all, self.cfg.get('max_random_shift', 1.0))

        img_ins = self.generate_image(traj_ins_all, rescale=0.25) 
        img_hmi = self.generate_image(traj_hmi_all, rescale=1.0) 

        img_ins = img_ins[...,0]
        img_ins = img_ins / 255.0

        img_hmi = self.image_transform(img_hmi)
        
        # normalize traj_ins
        assert traj_ins.shape[0] >= self.cfg.num_points_per_trajectory
        traj_ins = traj_ins[:self.cfg.num_points_per_trajectory,:2]
        
        assert traj_hmi.shape[0] >= self.cfg.num_points_per_trajectory
        traj_hmi = traj_hmi[:self.cfg.num_points_per_trajectory,:2]
        
        n_past = min(traj_ins_past.shape[0], self.cfg.num_points_per_trajectory)
        traj_hist = np.zeros_like(traj_ins)
        traj_hist[:n_past] = traj_ins_past[:n_past,:2]

        if self.cfg.normalize_trajectory:
            traj_ins = traj_ins / (self.map_w / 2)
            traj_hmi = traj_hmi / (self.map_w / 2)
            traj_hist = traj_hist / (self.map_w / 2)

        data_dict = {
                'img_hmi': img_hmi,
                'img_ins': img_ins,
                'frame_id': idx,
                'traj_ins': traj_ins,
                'traj_hmi': traj_hmi,
                'traj_hist': traj_hist
                }
       
        if self.cfg.use_lidar_points:
            data_dict['points'] = points

        if self.cfg.use_lidar_bev:
            # H, W, C -> C, H, W
            data_dict['lidar_bev'] = lidar_bev.transpose(2, 0, 1)

        return data_dict  
