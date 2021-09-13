import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset

from .registry import DATASETS
from .augmentor import DataAugmentor
from .processor import DataProcessor

@DATASETS.register
class PointCloudDataset(Dataset):
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.class_names = cfg.class_names
        self.root_path = Path(cfg.root_path)

        if self.cfg.get('augmentor', None):
            self.data_augmentor = DataAugmentor(self.root_path, cfg.augmentor, self.class_names, logger)

        if self.cfg.get('pre_processor', None):
            self.pre_processor = DataProcessor(cfg.pre_processor)

    def __len__(self):
        raise NotImplementedError

    def forward(self, index):
        raise NotImplementedError

    def augment_data(self, data_dict):
        if data_dict.get('gt_names', None) is not None:
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
        else:
            data_dict = self.data_augmentor.forward(
                data_dict={**data_dict})

        if data_dict.get('gt_boxes', None) is not None:
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        return data_dict

    def pre_process(self, data_dict):
        data_dict = self.pre_processor.forward(data_dict)
        return data_dict

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
