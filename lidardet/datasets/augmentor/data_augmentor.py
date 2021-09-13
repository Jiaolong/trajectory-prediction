import torch
import numpy as np
from functools import partial

from lidardet.utils.geometry import limit_period
from lidardet.utils.common import check_numpy_to_torch, scan_downsample, scan_upsample
from . import augmentor_utils, database_sampler

class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        for cur_cfg in augmentor_configs:
            cur_augmentor = getattr(self, cur_cfg['name'])(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)       
     
    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
   
    def scan_down_up(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.scan_down_up, config=config)
        points = data_dict['points']
        if config.random:
            enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
            if enable:
                points_lr = scan_downsample(points, output_rings = 'even_or_odd')
                points_hr = scan_upsample(points_lr)
                data_dict['points'] = points_hr
        else:
            points_lr = scan_downsample(points, output_rings = 'even_or_odd')
            points_hr = scan_upsample(points_lr)
            data_dict['points'] = points_hr

        return data_dict
    
    def scan_downsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.scan_downsample, config=config)
        points = data_dict['points']
        if config.random:
            enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
            if enable:
                points_lr = scan_downsample(points, output_rings = config.output_rings)
                data_dict['points'] = points_lr
        else:
            points_lr = scan_downsample(points, output_rings = config.output_rings)
            data_dict['points'] = points_lr

        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)

        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            img = data_dict['range_image_in'][:,:,::-1]
            gt = data_dict['range_image_gt'][:,:,::-1]
            data_dict['range_image_in'] = img
            data_dict['range_image_gt'] = gt

        return data_dict
    
    def random_noise(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_noise, config=config)

        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            img = data_dict['range_image_in']
            noise = np.random.normal(0, config.sigma, img.shape) # mu, sigma, size
            for i in range(img.shape[0]):
                noise[i, img[0] == 0] = 0

            img += noise
            data_dict['range_image_in'] = img

        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['axis']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

   
    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['angle']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

 
    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['scale']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'][:, 6] = limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict
