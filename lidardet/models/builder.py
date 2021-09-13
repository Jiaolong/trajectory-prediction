from ..utils.registry import build_from_cfg
from torch import nn

from .registry import (
    TRAJECTORY_PREDICTOR
)

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_trajectory_predictor(cfg):
    return build(cfg, TRAJECTORY_PREDICTOR)

def build_model(cfg, train_cfg=None, test_cfg=None):
    if cfg.task == 'trajectory_prediction':
        return build_trajectory_predictor(cfg)
    else:
        raise ValueError("Unknown task {}".format(cfg.task))
