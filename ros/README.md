Trajectory Prediction
=====================

## Requirements

- Ubuntu 20.04 & [ROS noetic](http://wiki.ros.org/noetic)
- Pytorch 1.4+

## Build

- Install catkin tools

```bash
sudo pip3 install osrf-pycommon
sudo apt install python3-catkin-tools
```

- Build

```
catkin build
source devel/setup.bash
```

## Parameters

Parameters can be configured in `src/traj_pred/config/traj_pred_node.yaml`.

## Prepare KTTI rosbag

Use [kitti2bag](https://github.com/tomas789/kitti2bag) to create rosbag from KTIIT raw data.

```bash
kitti2bag -t 2011_10_03 -r 0027 raw_synced . # 00
kitti2bag -t 2011_10_03 -r 0034 raw_synced . # 02
kitti2bag -t 2011_09_30 -r 0018 raw_synced . # 05
kitti2bag -t 2011_09_30 -r 0027 raw_synced . # 07
kitti2bag -t 2011_09_30 -r 0028 raw_synced . # 08
kitti2bag -t 2011_09_30 -r 0033 raw_synced . # 09
kitti2bag -t 2011_09_30 -r 0034 raw_synced . # 10
```
## How to launch

```bash
roslaunch traj_pred traj_pred.launch
rosbag play kitti_10.bag
```
