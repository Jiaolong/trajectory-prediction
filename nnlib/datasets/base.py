import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset

from .registry import DATASETS

@DATASETS.register
class PointCloudDataset(Dataset):
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.class_names = cfg.class_names
        self.root_path = Path(cfg.root_path)

    def __len__(self):
        raise NotImplementedError

    def forward(self, index):
        raise NotImplementedError

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            if key in ['voxels', 'voxel_num_points']:
                ret[key] = np.concatenate(val, axis=0)
            elif key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['gt_boxes']:
                max_gt = max([len(x) for x in val])
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                ret[key] = batch_gt_boxes3d
            else:
                ret[key] = np.stack(val, axis=0)

        ret['batch_size'] = batch_size
        return ret
