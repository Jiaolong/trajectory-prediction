import math
import torch
import torch.nn as nn
import numpy as np

from .conv_header import ConvHeader
from .backbone import Bottleneck, BackBone
from .predictor_base import PredictorBase
from ..registry import TRAJECTORY_PREDICTOR

@TRAJECTORY_PREDICTOR.register
class ConvTrajPredictor(PredictorBase):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        
        self.backbone = BackBone(Bottleneck, cfg.backbone)
        self.header = ConvHeader(cfg.header)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, batch_dict):
        x = batch_dict['lidar_bev']
        img_hmi = batch_dict['img_hmi']
        x = torch.cat([x, img_hmi], dim=1)

        batch_dict['input'] = x
        batch_dict = self.backbone(batch_dict)
        batch_dict = self.header(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
