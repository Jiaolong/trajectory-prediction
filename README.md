# Trajectory Prediction for Autonomous Driving with Topometric Map

![image](https://github.com/Jiaolong/trajectory-prediction/tree/main/data/kitti/traj_pred_kitti10.gif)

Repository for the paper ["Trajectory Prediction for Autonomous Driving with Topometric Map"](https://arxiv.org/abs/2105.03869).
```
@inproceedings{traj-pred:2022,
  title={Trajectory Prediction for Autonomous Driving with Topometric Map},
  author={J. Xu, L. Xiao, D. Zhao etal},
  booktitle={ICRA},
  year={2022}
}
```

## Requirements

- python 3.6+

- pytorch 1.4+

## Install build requirements

```shell
pip install -v -e .  # or "python setup.py develop"
```

## Pretrained models

Pre-trained weights can be downloaded [here](https://pan.baidu.com/s/1Ns7qjW352rMXJhleGJN2TQ)(code: uf9g)

## Dataset

```
├── datasets
│   └── KITTI_RAW
        └── trajectory_prediction
            ├── 07
            └── 10
```

Testing dataset kitti-10 can be downloaded [here](https://pan.baidu.com/s/1DrPRNWfMOy7JMc_TOzdV7w)(code: kbuf)

## Train & Test

### Train

* Train with multiple GPUs:
```shell script
sh tools/scripts/dist_train.sh ${NUM_GPUS} -c ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python tools/train.py --cfg config/trajectory_prediction/transformer.yaml
```

### Test

* Test with a pretrained model:
```shell script
python tools/test.py --cfg config/trajectory_prediction/transformer.yaml --ckpt cache/transformer_epoch_120.pth
```
