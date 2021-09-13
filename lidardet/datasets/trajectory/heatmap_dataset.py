import cv2
import math
import time
import json
import pickle
import random
import numpy as np
from matplotlib import cm
from pathlib import Path
from torchvision.transforms import ToTensor

from .utils import interp_l2_distance, hausdorff_dist, HeatMapGenerator
from .utils import l2_distance_to_trajectory_set, angle_to_trajectory_set
from .utils import random_shift, random_drop, random_flip, random_rotate
from lidardet.datasets.base import PointCloudDataset
from lidardet.datasets.registry import DATASETS

from lidardet.ops.lidar_bev import bev

@DATASETS.register
class HeatMapDataset(PointCloudDataset):

    def __init__(self, cfg, logger=None):
        super(HeatMapDataset, self).__init__(cfg=cfg, logger=logger)
        
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
        self.heatmap_generator = HeatMapGenerator(self.img_w, self.img_h, 0.25)
      
        if self.logger:
            self.logger.info('Loading TrajectoryPrediction {} dataset with {} samples'.format(cfg.mode, len(self.label_files)))
       
    def __len__(self):
        return len(self.label_files)
    
    def img2car(self, img_points, normalized=True):
        """
        Convert coordinates from image space to ego-car
        """
        car_points = img_points.copy()
        if normalized:
            img_points[:,0] *= self.img_w
            img_points[:,1] *= self.img_h

        car_points[:,0] = img_points[:,0] * self.res + self.lidar_range['Left']
        car_points[:,1] = self.lidar_range['Front'] - img_points[:,1] * self.res
        return car_points

    def car2img(self, car_point, normalize=False):
        """
        Convert coordinates from ego-car to image space
        """
        if car_point.ndim == 2: # ndarray
            img_points = car_point.copy()
            img_points[:,0] = (car_point[:,0] - self.lidar_range['Left']) / self.res
            img_points[:,1] = (self.lidar_range['Front'] - car_point[:,1]) / self.res
            if normalize:
                img_points[:,0] /= self.img_w
                img_points[:,1] /= self.img_h
            return img_points

        cx, cy = car_point
        px = (cx - self.lidar_range['Left']) / self.res;
        py = (self.lidar_range['Front'] - cy) / self.res;
        if normalize:
            px /= self.img_w
            py /= self.img_h
        return np.array([int(px), int(py)], np.int32)
     
    def lidar_bev(self, points):
        m = self.cfg.lidar_intensity_max
        assert np.max(points[:,3]) <= m
        points[:,3] = points[:,3] / m
        
        if self.cfg.lidar_bev_type == 'height_map':
            bev_map = bev.height_map(points[:,:4], self.lidar_range_np, self.res)
        elif self.cfg.lidar_bev_type == 'rgb_map':
            bev_map = bev.rgb_map(points[:,:4], self.lidar_range_np, self.res)
        elif self.cfg.lidar_bev_type == 'rgb_traversability_map':
            bev_map = bev.rgb_traversability_map(points, self.lidar_range_np, self.cfg.sensor_height, self.res)

        bev_map = bev_map.reshape((self.img_h, self.img_w, -1)).copy()
        return bev_map

    def generate_road_mask(self, trajectory, rescale=1.0, road_width=2.0):
         
        thickness = int(rescale * road_width / self.res) # 16 pixels
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
        result_dicts = {}
        
        bs = batch_dict['frame_id'].shape[0]
        num_modes = 1
        ADE = np.zeros((bs, num_modes)) # average displacement error
        HD = np.zeros((bs, num_modes)) # Hausdorff Distance
        FDE = np.zeros((bs,)) # final displacement error
        hit_rate = np.zeros((bs, num_modes))

        for index, pred_seg in enumerate(pred_dicts['pred_seg']):
            heatmap = pred_dicts['pred_heatmap'][index]
            frame_id = batch_dict['frame_id'][index]
            pred_points = pred_dicts['pred_traj'][index]
            traj_pred_all = self.img2car(pred_points, normalized=True) # [num_points, 2]

            label_file = Path(self.label_files[frame_id])     
            # load json
            with open(str(label_file)) as f:
                data = json.load(f)
             
            lidar_path = str(label_file).replace('json', 'bin')
            points = np.fromfile(lidar_path, dtype=np.float32)
            points = points.reshape([-1, 4])

            # remove nan
            points = points[~np.isnan(points).any(axis=1)]
            lidar_bev = self.lidar_bev(points)
            
            if self.cfg.lidar_bev_type == 'height_map':
                img_bev = (lidar_bev[..., -1] * 255).astype(np.uint8)
                img_bev = img_bev[..., None]
            else: 
                img_bev = (lidar_bev[:,:,:3] * 255).astype(np.uint8)
       
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
           
            #traj_pred_all = traj_const 
            traj_pred_interp, traj_ins_interp, metric = interp_l2_distance(traj_pred_all, traj_ins)
            FDE[index] = metric['FDE']
            
            ADE[index, 0] = metric['ADE']
            HD[index, 0] = hausdorff_dist(traj_pred_all, traj_ins)
            hit_rate[index, 0] = 1 if metric['MAX'] < 2 else 0

            if output_path is not None:
                fname = label_file.stem
                save_dir = output_path / 'predictions' 
                save_dir.mkdir(parents=True, exist_ok=True)
                save_file = save_dir / ('%s.jpg' % fname)
                #print('Saving result {}'.format(save_file))
                
                pred_seg *= 255
                pred_seg = cv2.resize(pred_seg, (int(self.img_w), int(self.img_h)))
                img_seg = (img_bev + 0.3 * pred_seg[..., None]).astype(np.uint8)

                heatmap *= 255
                heatmap = cv2.resize(heatmap, (int(self.img_w), int(self.img_h)))
                img_hm = (img_bev * 0.3 + heatmap[..., None]).astype(np.uint8)
                
                img_traj = np.zeros((self.img_h, self.img_w, 3), np.uint8)
                img_traj += img_bev
                img_traj = self.plot_trajectory_set([traj_ins, traj_pred_all, traj_hmi], 
                        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)], img=img_traj)

                img_all = np.concatenate([img_traj, img_hm, img_seg], axis=1)
                cv2.imwrite(str(save_file), img_all)
            
        result_dicts['ADE'] = ADE
        result_dicts['HD'] = HD
        result_dicts['FDE'] = FDE
        result_dicts['HitRate'] = hit_rate
        return result_dicts

    def evaluation(self, result_dicts_list):
        self.logger.info('Evaluating on {}'.format(self.cfg.subset))
        for i, result_dicts in enumerate(result_dicts_list):
            HD_i = result_dicts['HD']
            ADE_i = result_dicts['ADE']
            FDE_i = result_dicts['FDE']
            hit_rate_i = result_dicts['HitRate']
            if i == 0:
                HD = HD_i
                ADE = ADE_i
                FDE = FDE_i
                hit_rate = hit_rate_i
            else:
                HD = np.concatenate((HD, HD_i), axis=0)
                ADE = np.concatenate((ADE, ADE_i), axis=0)
                FDE = np.concatenate((FDE, FDE_i), axis=0)
                hit_rate = np.concatenate((hit_rate, hit_rate_i), axis=0)
        
        self.logger.info('FDE =  {}'.format(np.mean(FDE)))
        for i in range(HD.shape[1]):
            self.logger.info('HD@{} = {}'.format(i, np.mean(HD[:,i])))

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
 
    def __getitem__(self, idx):
        label_file = Path(self.label_files[idx]) 
        self.curr_file = label_file
        # load json
        with open(str(label_file)) as f:
            data = json.load(f)

        lidar_path = str(label_file).replace('json', 'bin')
        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape([-1, 4])
        
        # remove nan
        points = points[~np.isnan(points).any(axis=1)]
 
        traj_ins = np.array(data['trajectory_ins'])
        traj_hmi = np.array(data['trajectory_hmi'])
        
        traj_ins_past = np.array(data['trajectory_ins_past'])
        traj_hmi_past = np.array(data['trajectory_hmi_past'])
        
        if self.cfg.mode == 'train':
            # random drop points
            if self.cfg.random_drop:
                points = random_drop(points)

            # random rotate
            if self.cfg.random_rotate:
                points, traj_list = random_rotate(points, [traj_ins, traj_ins_past, traj_hmi, traj_hmi_past])
                traj_ins, traj_ins_past, traj_hmi, traj_hmi_past = traj_list

            # random flip
            if self.cfg.random_flip:
                points, traj_list = random_flip(points, [traj_ins, traj_ins_past, traj_hmi, traj_hmi_past])
                traj_ins, traj_ins_past, traj_hmi, traj_hmi_past = traj_list

            # random shift hmi trajectory
            if self.cfg.random_shift:
                traj_hmi, taj_hmi_past = random_shift([traj_hmi, traj_hmi_past], self.res, self.cfg.get('max_random_shift', 1.0))
        
        # set origin on the ground 
        points[:,2] += self.cfg.sensor_height

        traj_ins_all = np.vstack((traj_ins_past[::-1], traj_ins)) 
        traj_hmi_all = np.vstack((traj_hmi_past[::-1], traj_hmi)) 

        img_ins = self.generate_road_mask(traj_ins_all, rescale=0.25, road_width=self.road_width) 
        img_hmi = self.generate_road_mask(traj_hmi_all, rescale=1.0, road_width=self.road_width) 

        lidar_bev = self.lidar_bev(points)
 
        img_ins = img_ins[...,0]
        img_ins = img_ins / 255.0

        img_hmi = self.image_transform(img_hmi)
        
        assert traj_ins.shape[0] >= self.cfg.num_points_per_trajectory
        traj_ins = traj_ins[:self.cfg.num_points_per_trajectory,:2]
        
        assert traj_hmi.shape[0] >= self.cfg.num_points_per_trajectory
        traj_hmi = traj_hmi[:self.cfg.num_points_per_trajectory,:2]
        
        n_past = min(traj_ins_past.shape[0], self.cfg.num_points_per_trajectory)
        traj_hist = np.zeros_like(traj_ins)
        traj_hist[:n_past] = traj_ins_past[:n_past,:2]
        
        traj_ins_pixel_norm = self.car2img(traj_ins, normalize=True)
        heatmap = self.heatmap_generator(traj_ins_pixel_norm.copy())
        
        if False:
            
            if self.cfg.lidar_bev_type == 'rgb_traversability_map':
                obstacle_map1 = lidar_bev[...,-2] # infered traversability map
                img_obstacle1 = np.zeros((self.img_h, self.img_w, 3)).astype(np.uint8)
                img_obstacle1[obstacle_map1 == 0, 0] = 255 
                img_obstacle1[obstacle_map1 == 2, 1] = 255 
                img_obstacle1[obstacle_map1 == 1, 2] = 255 
                
                obstacle_map2 = lidar_bev[...,-1] # original
                img_obstacle2 = np.zeros((self.img_h, self.img_w, 3)).astype(np.uint8)
                img_obstacle2[obstacle_map2 == 0, 0] = 255 
                img_obstacle2[obstacle_map2 == 2, 1] = 255 
                img_obstacle2[obstacle_map2 == 1, 2] = 255 

                cv2.imshow('obstacle_map', np.concatenate([img_obstacle2, img_obstacle1], axis=1))
       
            img_bev = (lidar_bev[:,:,:3] * 255).astype('uint8')
            img_bev = self.plot_trajectory_set(trajectory_set=[traj_ins_all, traj_hmi_all], img=img_bev, thickness=3)
            cv2.imshow("lidar_bev", img_bev)
            cv2.waitKey(0)
            exit(0)

        if False:
            cv2.imshow("heatmap", np.sum(heatmap, axis=0))
            cv2.waitKey(0)
            exit(0)

        data_dict = {
                'img_hmi': img_hmi,
                'img_ins': img_ins,
                'frame_id': idx,
                'traj_ins': traj_ins,
                'traj_ins_pixel_norm': traj_ins_pixel_norm,
                'traj_hmi': traj_hmi,
                'traj_hist': traj_hist,
                'heatmap': heatmap
                }
        
        if self.cfg.use_lidar_points:
            data_dict['points'] = points

        if self.cfg.lidar_bev_type == 'rgb_traversability_map':
            lidar_bev[...,-2] -= 1 # convert to range [-1, 0, 1]
            data_dict['lidar_bev'] = lidar_bev[:,:,:-1].transpose(2, 0, 1)
            obstacle_map = lidar_bev[...,-2].copy() 
            obstacle_map[obstacle_map != 0] = 1;
            # rescale
            scale = 0.25
            obstacle_map = cv2.resize(obstacle_map, (int(self.img_w * scale), int(self.img_h * scale)))
            data_dict['obstacle_map'] = obstacle_map

            #cv2.imshow("obstacle_map", (obstacle_map * 255).astype(np.uint8))
            #cv2.waitKey(0)
            #exit(0)
        else:
            # H, W, C -> C, H, W
            data_dict['lidar_bev'] = lidar_bev[:,:,:3].transpose(2, 0, 1)

        if self.cfg.get('pre_processor', None):
            data_dict = self.pre_process(data_dict)

        return data_dict  
