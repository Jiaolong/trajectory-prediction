# Trajectory Prediction with Noisy Topometric Map for Autonomous Navigation

Repository for the paper ["Trajectory Prediction with Noisy Topometric Map for Autonomous Navigation"](https://arxiv.org/).
```
@inproceedings{traj-pred:2020,
  title={Trajectory Prediction with Noisy Topometric Map for Autonomous Navigation},
  author={J. Xu, L. Xiao, D. Zhao etal},
  booktitle={arxiv},
  year={2020}
}
```

## Requirements

- python 3.6

- pytorch 1.4+

- other requirements: `pip install -r requirements.txt`

## Pretrained models

Pre-trained weights can be downloaded [here](https://pan.baidu.com/s/1riU1Wu2lhYn5FJsaAoPB7A)(code:6ol2)

## Build C++ extensions

```shell script
cd nnlib/datasets/trajectory/
bash make.sh
```

## ROS demo

See ros package [README](ros/README.md)

## Train & Test

Export current directory to `PYTHONPATH`:

```bash
export PYTHONPATH=`pwd`
```
### Train

* Train with multiple GPUs:
```shell script
sh tools/scripts/dist_train.sh ${NUM_GPUS} -c ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python3 tools/train.py -c ${CONFIG_FILE}
```

### Test

* Test with a pretrained model:
```shell script
python3 tools/test.py -c ${CONFIG_FILE} -m ${MODEL_FILE}
```
