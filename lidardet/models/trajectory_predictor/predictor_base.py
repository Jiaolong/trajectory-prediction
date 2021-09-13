import torch
import os
import torch.nn as nn

from ..registry import TRAJECTORY_PREDICTOR

@TRAJECTORY_PREDICTOR.register
class PredictorBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg = cfg
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1
        
    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        pred_dicts = {}
        ret_dicts = {}
        pred_dicts['pred_seg'] = batch_dict['pred_seg'].cpu().numpy()
        if 'pred_heatmap' in batch_dict:
            pred_dicts['pred_heatmap'] = batch_dict['pred_heatmap'].cpu().numpy()
        pred_dicts['pred_traj'] = self.header.get_prediction(batch_dict)
        return pred_dicts, ret_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss, tb_dict = self.header.get_loss()

        tb_dict = {
            'loss': loss.item(),
            **tb_dict
        }

        return loss, tb_dict, disp_dict
       
    def load_params_from_file(self, filename, logger=None, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        
        if logger:
            logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if logger and 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict():
                if self.state_dict()[key].shape == model_state_disk[key].shape:
                    update_model_state[key] = val
                    # logger.info('Update weight %s: %s' % (key, str(val.shape)))
                #else:
                #    logger.info('Shape not matched %s: self --> %s vs disk --> %s ' % (key, str(self.state_dict()[key].shape), str(val.shape)))


        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state and logger:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
        
        if logger:
            logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
        else:
            print('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)
        self.load_state_dict(checkpoint['model_state'])
        
        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
