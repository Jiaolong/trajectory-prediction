from functools import partial
import numpy as np
#from ...utils import box_utils, geometry
#from ...utils.common import scan_downsample, scan_upsample, scan_to_range

class DataProcessor(object):
    def __init__(self, processor_configs):
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg['name'])(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)
 
    def remove_points_and_boxes_outside_range(self, data_dict=None, config=None):

        if data_dict is None:
            return partial(self.remove_points_and_boxes_outside_range, config=config)
        
        point_cloud_range = np.array(config['point_cloud_range'], dtype=np.float32)
        mask = geometry.mask_points_by_range(data_dict['points'], point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config['remove_outside_boxes']:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def normalization(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.normalization, config=config)
        
        img = data_dict['range_image_in']
        for i in range(img.shape[0]):
            img[i,...] = (img[i,...] - config['mean'][i]) / config['std'][i]
        
        data_dict['range_image_in'] = img
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        points = data_dict['points']
        shuffle_idx = np.random.permutation(points.shape[0])
        points = points[shuffle_idx]
        data_dict['points'] = points

        return data_dict

    def get_range_image(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.get_range_image, config=config)
        points = data_dict['points']
        range_image, _, _ = scan_to_range(points, normalize=True)
        range_image = range_image[:,:,:2]
        data_dict['range_image_even'] = range_image[::2, :, :].transpose((2, 0, 1))
        data_dict['range_image_odd'] = range_image[1::2, :, :].transpose((2, 0, 1))

        if 'points' in data_dict:
            data_dict.pop('points')
        if 'gt_boxes' in data_dict:
            data_dict.pop('gt_boxes')
        return data_dict

    def scan_downsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.scan_downsample, config=config)
        points = data_dict['points']
        points_lr = scan_downsample(points)
        data_dict['points'] = points_lr
        return data_dict

    def scan_upsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.scan_upsample, config=config)
        points = data_dict['points']
        points_dense = scan_upsample(points)
        data_dict['points'] = points_dense
        return data_dict

    def voxelization(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            from spconv.utils import VoxelGenerator
            point_cloud_range = np.array(config['point_cloud_range'], dtype=np.float32)
            voxel_generator = VoxelGenerator(
                voxel_size=config['voxel_size'],
                point_cloud_range=point_cloud_range,
                max_num_points=config['max_points_per_voxel'],
                max_voxels=config['max_num_voxels'], full_mean=False
            )
            grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(config['voxel_size'])
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config['voxel_size']
            return partial(self.voxelization, voxel_generator=voxel_generator)
        points = data_dict['points']
        voxels, coordinates, num_points = voxel_generator.generate(points)
        data_dict['use_lead_xyz'] = True
        #if not data_dict['use_lead_xyz']:
        #    voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
