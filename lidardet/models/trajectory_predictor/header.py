import torch
from torch import nn

from .mtp import MTP
from .covernet import CoverNet
from .backbone import conv3x3

class Header(nn.Module):

    def __init__(self, cfg):
        super(Header, self).__init__()
        self.cfg = cfg
        self.use_bn = cfg.use_bn
        self.with_attention = cfg.with_attention
        bias = not self.use_bn
        self.conv1 = conv3x3(96, 96, bias=bias)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = conv3x3(96, 96, bias=bias)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)
        
        dim_conv5 = 384
        if self.with_attention:
            self.conv7 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=bias)
            self.bn7 = nn.BatchNorm2d(16)
            self.conv8 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=bias)
            self.bn8 = nn.BatchNorm2d(32)
            #self.pool = nn.MaxPool2d(kernel_size=4, padding=(1,0))
            dim_conv5 += 32

        self.conv5 = nn.Conv2d(dim_conv5, 512, kernel_size=3, stride=2, padding=0, bias=bias)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=0, bias=bias)
        self.bn6 = nn.BatchNorm2d(1024)
        self.seg_head = conv3x3(96, 1, bias=True)

        if self.cfg.get('mtp'):
            self.pred_head = MTP(cfg.mtp)
        elif self.cfg.get('covernet'):
            self.pred_head = CoverNet(cfg.covernet)

        self.seg_loss_func = nn.BCELoss()

    def forward(self, batch_dict):
        x = batch_dict['seg_features']

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)
        
        seg = torch.sigmoid(self.seg_head(x)) # [b, 1, 75, 100]
       
        y = batch_dict['reg_features'] # [b, 384, 19, 25]

        if self.with_attention:
            z = self.conv7(seg)
            z = self.bn7(z)
            z = self.conv8(z)
            z = self.bn8(z)
            y = torch.cat([y, z], dim=1)

        y = self.conv5(y)
        if self.use_bn:
            y = self.bn5(y)
        y = self.conv6(y)
        if self.use_bn:
            y = self.bn6(y)
        y = y.mean([2, 3])
        batch_dict['reg_features'] = y 
        
        batch_dict = self.pred_head(batch_dict)

        if self.training:
            self.seg_pred = seg.squeeze(1)
            self.seg_target = batch_dict['img_ins']

        batch_dict['pred_seg'] = seg.squeeze(1)
        return batch_dict

    def get_loss(self):
        loss_pred = self.pred_head.get_loss()
        loss_seg = self.seg_loss_func(self.seg_pred, self.seg_target)

        loss = self.cfg.weight_loss_seg * loss_seg + loss_pred
        tb_dict = {'loss_pred': loss_pred, 'loss_seg': loss_seg}
        return loss, tb_dict

    def get_prediction(self, batch_dict):
        pred_traj_list = []
        if self.cfg.get('mtp'):
            pred_traj_batch = batch_dict['trajectory_predictions']
            mode_prob_batch = batch_dict['mode_probabilities']
            for i, mode_prob in enumerate(mode_prob_batch):
                order = torch.argsort(mode_prob, dim=0, descending=True)
                traj_sorted = pred_traj_batch[i][order].cpu().numpy()
                pred_traj_list.append(traj_sorted)
        elif self.cfg.get('covernet'):
            logits = batch_dict['logits']
            for i, mode_prob in enumerate(logits):
                order = torch.argsort(mode_prob, dim=0, descending=True)
                pred_traj_list.append(order.cpu().numpy())
        
        return pred_traj_list
