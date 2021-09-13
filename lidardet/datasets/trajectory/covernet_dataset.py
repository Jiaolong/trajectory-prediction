import cv2
import math
import json
import pickle
import random
import numpy as np
from matplotlib import cm
from pathlib import Path
from torchvision.transforms import ToTensor

from .utils import interp_l2_distance
from .utils import l2_distance_to_trajectory_set, angle_to_trajectory_set
from lidardet.datasets.base import PointCloudDataset
from lidardet.datasets.registry import DATASETS


@DATASETS.register
class CoverNetDataset(PointCloudDataset):

    def __init__(self, cfg, logger=None):
        super(CoverNetDataset, self).__init__(cfg=cfg, logger=logger)
        
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
        self.lidar_range = {'Left': xmin, 'Right': xmax, 'Front': ymax, 'Back': ymin, 'Bottom': zmin, 'Top': zmax}
        self.map_h = self.lidar_range['Front'] - self.lidar_range['Back']
        self.map_w = self.lidar_range['Right'] - self.lidar_range['Left']

        self.img_h = int(self.map_h / self.res) # 250
        self.img_w = int(self.map_w / self.res) # 400

        self.trajectory_colors = cm.get_cmap('viridis', 1000)
        self.image_transform = ToTensor()
        
        traj_set_path = self.root_path / 'trajset.npy'
        if traj_set_path.exists():
            self.traj_set = np.load(traj_set_path)
            self.logger.info('Loaded {} trajectories.'.format(self.traj_set.shape[0]))
        else:
            self.logger.info('Creating trajectory set ...')
            self.create_trajectory_set(traj_set_path)

        
        self.logger.info('Calculating trajectory labels ...')
        self.calculate_traj_label()

        if self.logger:
            self.logger.info('Loading TrajectoryPrediction {} dataset with {} samples'.format(cfg.mode, len(self.label_files)))
       
    def __len__(self):
        return len(self.label_files)
    
    def calculate_traj_label(self):
        count = 0
        self.traj_labels = [0] * self.__len__()
        for idx in range(self.__len__()):
            label_file = Path(self.label_files[idx]) 
            # load json
            with open(str(label_file)) as f:
                data = json.load(f)
            
            count += 1
            if count % 100 == 0:
                print('Calculating trajectory labels {}/{}'.format(count, len(self.label_files)))

            traj_ins = np.array(data['trajectory_ins'])
            assert traj_ins.shape[0] >= self.cfg.num_points_per_trajectory
            traj_ins = traj_ins[:self.cfg.num_points_per_trajectory,:2]
            min_id, d = angle_to_trajectory_set(traj_ins, np.array(self.traj_set))
            #print(min_id, d)
            self.traj_labels[idx] = min_id
                
            #if count % 50 == 0:
            if False:
                img_traj = self.plot_trajectory_set([traj_ins, self.traj_set[min_id]])

                cv2.imshow("dist={}".format(d), img_traj)
                cv2.waitKey(0)
        
    def create_trajectory_set(self, traj_set_path):
        traj_set = []
        count = 0
        for idx in range(len(self.label_files)):
            label_file = Path(self.label_files[idx]) 
            # load json
            with open(str(label_file)) as f:
                data = json.load(f)
            
            count += 1
            if count % 100 == 0:
                print('Processing {}/{} trajset size = {}'.format(count, len(self.label_files), len(traj_set)))

            traj_ins = np.array(data['trajectory_ins'])
            # normalize traj_ins
            #traj_ins[:,:2] = traj_ins[:,:2] / (self.map_w * 0.5)
            assert traj_ins.shape[0] >= self.cfg.num_points_per_trajectory
            traj_ins = traj_ins[:self.cfg.num_points_per_trajectory,:2]
            if len(traj_set) == 0:
                traj_set.append(traj_ins)
            else: 
                min_id, d = l2_distance_to_trajectory_set(traj_ins, np.array(traj_set))
                #min_id, d = angle_to_trajectory_set(traj_ins, np.array(traj_set))
                print(min_id, d)
                if d > self.cfg.traj_thresh:
                    traj_set.append(traj_ins)

            if count % 50 == 0:
                img_traj = self.plot_trajectory_set(traj_set)

                cv2.imshow("traj_set", img_traj)
                cv2.waitKey(0)
        print('Created {} trajectories from {}'.format(len(traj_set), count))
        print('Trajectories are saved at {}'.format(str(traj_set_path)))
        self.traj_set = np.array(traj_set)
        np.save(str(traj_set_path), self.traj_set)

    def car2img(self, car_point):
        """
        Convert coordinates from ego-car to image space
        """

        cx, cy = car_point
        px = (cx - self.lidar_range['Left']) / self.res;
        py = (self.lidar_range['Front'] - cy) / self.res;
        return np.array([int(px), int(py)], np.int32)

    def lidar_bev_image(self, points):
        """
        Convert lidar points to BEV representation
        """
        x_points = points[:,0]
        y_points = points[:,1]
        z_points = points[:,2]
        i_points = points[:,3]
        if self.cfg.lidar_intensity_max == 255:
            i_points = i_points / 255.0

        s_filt = np.logical_and(x_points > self.lidar_range['Left'], x_points < self.lidar_range['Right'])
        f_filt = np.logical_and(y_points > self.lidar_range['Back'], y_points < self.lidar_range['Front'])
        z_filt = np.logical_and(z_points > self.lidar_range['Bottom'], z_points < self.lidar_range['Top'])
        fs_filt = np.logical_and(f_filt, s_filt)
        filt = np.logical_and(fs_filt, z_filt)
        indices = np.argwhere(filt).flatten()
        
        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]
        i_points = i_points[indices]
        
        map_z = (self.lidar_range['Top'] - self.lidar_range['Bottom']) / self.res
        bev_map = np.zeros((self.img_h, self.img_w, int(map_z) + 1), dtype=np.float32)
        intensity_map_count = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        # convert to pixel coordinate
        for i in range(x_points.shape[0]):
            px, py = self.car2img([x_points[i], y_points[i]])
            pz = (z_points[i] - self.lidar_range['Bottom']) / self.res
            bev_map[py, px, int(pz)] = 1
            bev_map[py, px, -1] += i_points[i]
            intensity_map_count[py, px] += 1
       
        intensity_map_count[intensity_map_count == 0] = 1
        bev_map[:,:,-1] = bev_map[:,:,-1] / intensity_map_count
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
            pred_idx = pred_dicts['pred_traj'][index]
            traj_pred_all = self.traj_set[pred_idx][:num_modes] # [num_modes, num_points, 2]

            label_file = Path(self.label_files[frame_id])     
            # load json
            with open(str(label_file)) as f:
                data = json.load(f)
             
            if self.cfg.use_lidar_bev:
                lidar_bev_path = str(label_file).replace('.json', '_bev.npy')
                if self.cfg.use_cache and  Path(lidar_bev_path).exists():
                    lidar_bev = np.load(lidar_bev_path)
                else:
                    lidar_bev = self.lidar_bev_image(points)
       
            traj_hmi = np.array(data['trajectory_hmi'])
            traj_ins = np.array(data['trajectory_ins'])
            
            assert traj_ins.shape[0] >= self.cfg.num_points_per_trajectory
            traj_ins = traj_ins[:self.cfg.num_points_per_trajectory,:2]
            
            assert traj_hmi.shape[0] >= self.cfg.num_points_per_trajectory
            traj_hmi = traj_hmi[:self.cfg.num_points_per_trajectory,:2]
            
            for m, traj_pred in enumerate(traj_pred_all):
                traj_pred_interp, traj_ins_interp, metric = interp_l2_distance(traj_pred, traj_ins)
                if m == 0:
                    FDE[index] = metric['FDE']
                
                ADE[index, m] = metric['ADE']
                hit_rate[index, m] = 1 if metric['MAX'] < 2 else 0

                ADE[index, m] = np.min(ADE[index, :m+1])
                hit_rate[index, m] = np.max(hit_rate[index, :m+1])

            if True:
                img_traj = self.plot_trajectory_set([traj_ins, traj_hmi, traj_pred_all[0]])
                #cv2.imshow("traj_pred", img_traj)
                #cv2.waitKey(0)
                #exit(0)

            if output_path is not None:
                fname = label_file.stem
                save_dir = output_path / 'predictions' 
                save_dir.mkdir(parents=True, exist_ok=True)
                save_file = save_dir / ('%s.png' % fname)
                #print('Saving result {}'.format(save_file))
                h, w = img_traj.shape[:2]
                img_save = np.zeros((h, w*3, 3), np.uint8)
                img_save[:,:w,:] = img_traj
                pred_seg *= 255
                pred_seg = cv2.resize(pred_seg, (int(self.img_w), int(self.img_h)))
                img_save[:,w:w*2,:] = pred_seg[...,np.newaxis]
                
                img_bev = (lidar_bev[:,:,-1] * 255).astype(np.uint8)
                img_save[:,w*2:,:] = img_bev[...,np.newaxis]

                cv2.imwrite(str(save_file), img_save)
        
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

    def plot_trajectory_set(self, trajectory_set, thickness=2):
        img = np.ones((self.img_h, self.img_w, 3)).astype(np.uint8)
        img *= 255
        for t, trajectory in enumerate(trajectory_set):
            color = [int(255 * c) for c in self.trajectory_colors(1.0 * t / len(trajectory_set))]
            for i in range(trajectory.shape[0] - 1):
                x1, y1 = self.car2img(trajectory[i][:2])
                x2, y2 = self.car2img(trajectory[i+1][:2])
                            
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
            lidar_bev_path = str(label_file).replace('.json', '_bev.npy')
            if self.cfg.use_cache and  Path(lidar_bev_path).exists():
                lidar_bev = np.load(lidar_bev_path)
            else:
                lidar_bev = self.lidar_bev_image(points)
                np.save(lidar_bev_path, lidar_bev)

        traj_ins = np.array(data['trajectory_ins'])
        traj_hmi = np.array(data['trajectory_hmi'])
        
        traj_ins_past = np.array(data['trajectory_ins_past'])
        traj_hmi_past = np.array(data['trajectory_hmi_past'])
        
        traj_ins_all = np.vstack((traj_ins_past[::-1], traj_ins)) 
        traj_hmi_all = np.vstack((traj_hmi_past[::-1], traj_hmi)) 

        # random shit hmi trajectory
        if self.cfg.mode == 'train':
            traj_hmi_all = self.random_shift(traj_hmi_all)

        img_ins = self.generate_image(traj_ins_all, rescale=0.25) 
        img_hmi = self.generate_image(traj_hmi_all, rescale=1.0) 
        
        assert traj_ins.shape[0] >= self.cfg.num_points_per_trajectory
        traj_ins = traj_ins[:self.cfg.num_points_per_trajectory,:2]
        
        #min_id, d = l2_distance_to_trajectory_set(traj_ins, self.traj_set)
        #min_id, d = angle_to_trajectory_set(traj_ins, self.traj_set)

        if False:
            traj_ins_tmp = traj_ins.copy()
            traj_ins_tmp[:,0] -= 0.0
            img_traj = self.plot_trajectory_set([traj_ins_tmp, self.traj_set[min_id]])
            cv2.imshow("dist: {}".format(d), img_traj)
            #cv2.imshow("img_ins", img_ins)
            #cv2.imshow("img_hmi", img_hmi)
            #img_bev = (lidar_bev[:,:,-1] * 255).astype(np.uint8)
            #cv2.imshow("lidar_bev", img_bev)
        
            cv2.waitKey(0)
            exit(0)

        img_ins = img_ins[...,0]
        img_ins = img_ins / 255.0

        img_hmi = self.image_transform(img_hmi)
        
        data_dict = {
                'img_hmi': img_hmi,
                'img_ins': img_ins,
                'frame_id': idx,
                'traj_ins': traj_ins,
                'traj_label': self.traj_labels[idx]
                }
       
        if self.cfg.use_lidar_points:
            data_dict['points'] = points

        if self.cfg.use_lidar_bev:
            # H, W, C -> C, H, W
            data_dict['lidar_bev'] = lidar_bev.transpose(2, 0, 1)

        if self.cfg.get('pre_processor', None):
            data_dict = self.pre_process(data_dict)
            #for k in data_dict.keys():
            #    if k in ['voxels', 'voxel_coords', 'voxel_num_points']:
            #        print(k, data_dict[k].shape)

        return data_dict  
