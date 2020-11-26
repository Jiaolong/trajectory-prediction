#!/usr/bin/env python3
import os
import cv2
import sys
import math
import time
import torch
import threading
import numpy as np
from pathlib import Path
from torchvision.transforms import ToTensor

# ros
import rospy
import rospkg
import sensor_msgs.point_cloud2 as pc2
from geodesy import utm
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, NavSatFix, Imu
from tf.transformations import euler_from_quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber

# nnlib
PWD = Path(__file__).parent
nnlib_dir = PWD / '..' / '..' / '..' / '..'
sys.path.insert(0, str(nnlib_dir))

from nnlib.datasets.trajectory import bev
from nnlib.utils.config import cfg_from_file
from nnlib.models.builder import build_model
from nnlib.datasets.trajectory import MTPDataset
from nnlib.utils.satellite_map import get_satellite_map

ROS_INFO = rospy.loginfo
ROS_WARN = rospy.logwarn
ROS_ERR  = rospy.logerr
APP_NAME = 'TrajPred'

# ROS App Node
class AppNode(object):
    def __init__(self):
        super(AppNode, self).__init__()

        self.use_kitti_bag = rospy.get_param("use_kitti_bag") 
        model_cfg_file = rospy.get_param("model_cfg_file")

        model_ckpt_file = nnlib_dir / 'cache' / rospy.get_param("model_ckpt_file")
        config_file = PWD / '..' / 'config' / model_cfg_file

        cfg = cfg_from_file(str(config_file))
        self.cfg = cfg

        # initialize detector
        cfg.model.header.mtp.anchor_path = str(nnlib_dir) + '/' + cfg.model.header.mtp.anchor_path
        model = build_model(cfg.model)
        ROS_INFO("[{}] Loading from checkpoint '{}'".format(APP_NAME, model_ckpt_file))
        if not model_ckpt_file.exists():
            ROS_ERR('Model weight file {} does not exist!'.format(model_ckpt_file))
            exit(0)

        model.load_params_from_file(filename=model_ckpt_file, to_cpu=False)
        model.cuda()
        model.eval()
        self.model = model
        
        self.img_bev = None
        self.img_map = None
        self.img_gloabl_path = None
        self.lock_img_map = threading.Lock()
        self.lock_lidar = threading.Lock()
        self.lidar_queue_size = 5
        self.lidar_queue = {}

        self.gps_utm_points = None
        self.hmi_utm_points = None
        self.bev_res = cfg.data.val.voxel_size[0] # 0.16 m / pixel
        self.road_width = cfg.data.val.road_width # meters
        xmin, ymin, zmin, xmax, ymax, zmax = cfg.point_cloud_range
        self.lidar_range_np = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        self.lidar_range = {'Left': xmin, 'Right': xmax, 'Front': ymax, 'Back': ymin, 'Bottom': zmin, 'Top': zmax}
        self.bev_h = self.lidar_range['Front'] - self.lidar_range['Back']
        self.bev_w = self.lidar_range['Right'] - self.lidar_range['Left']

        self.img_h = int(self.bev_h / self.bev_res)
        self.img_w = int(self.bev_w / self.bev_res)
        
        self.path_search_size = 30
        self.last_hmi_idx = -1
        self.last_gps_idx = -1
        self.traj_len = 50
        
        self.map_res = 1.0 # 1.0 m / pixel
        self.utm_range = None # {'xmin': -1.0, 'ymin': -1.0, 'xmax': -1.0, 'ymax': -1.0}
        self.map_h = -1.0
        self.map_w = -1.0
        self.image_transform = ToTensor()
        self.db = MTPDataset(cfg.data.val)

        if self.use_kitti_bag:
            gps_topic_sub = rospy.get_param("kitti/topic_gps")
            imu_topic_sub = rospy.get_param("kitti/topic_imu")
            lidar_topic_sub = rospy.get_param("kitti/topic_lidar")

            # global path
            gps_csv = nnlib_dir / 'data' / 'kitti' / rospy.get_param("kitti/gps_path")
            hmi_csv = nnlib_dir / 'data' / 'kitti' / rospy.get_param("kitti/hmi_path")
            self.load_gps_path(gps_csv)
            self.load_hmi_path(hmi_csv)

            sync = ApproximateTimeSynchronizer([Subscriber(gps_topic_sub, NavSatFix), Subscriber(imu_topic_sub, Imu)], slop=0.1, queue_size=5) 
            sync.registerCallback(self.gps_imu_callback)
        
        self.lidar_sub = rospy.Subscriber(
            lidar_topic_sub, PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)
        
        self.hmi_local_markers_pub = rospy.Publisher('/hmi/local_markers', MarkerArray, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def get_synchronized(self, stamp, queue):
        item = None
        min_delta = 1e5
        for t in queue:
            delta =  (stamp - t).to_sec()
            if delta < min_delta:
                min_delta = delta
                item = queue[t]

        if min_delta > 0.2:
            ROS_WARN('[%s]: get_synchronized: time gap is over 0.2 second!' % APP_NAME)

        return item

    def predict(self, data_dict, num_modes=3):

        with torch.no_grad():

            # create batch data
            batch_dict = self.db.collate_batch([data_dict])

            for key, val in batch_dict.items():
                if not isinstance(val, np.ndarray):
                    continue
                batch_dict[key] = torch.from_numpy(val).float().cuda()

            pred_dicts, ret_dict = self.model(batch_dict)

            pred_seg = pred_dicts['pred_seg'][0]
            traj_pred_all = pred_dicts['pred_traj'][0][:num_modes] # [num_modes, num_points, 2]
            
            if self.cfg.data.val.normalize_trajectory:
                traj_pred_all *= (self.map_w / 2)

            return pred_seg, traj_pred_all[0]
   
    def utm2img(self, utm_points):
        """
        Convert from utm to image coordinates
        """
        assert self.utm_range is not None
        assert self.map_res > 0
        xmin = self.utm_range['xmin']
        ymax = self.utm_range['ymax']

        img_points = utm_points.copy()
        if utm_points.ndim == 1:
            img_points[0] = (utm_points[0] - xmin) / self.map_res
            img_points[1] = (ymax - utm_points[1]) / self.map_res
        else:
            img_points[:,0] = (utm_points[:,0] - xmin) / self.map_res
            img_points[:,1] = (ymax - utm_points[:,1]) / self.map_res
        return img_points.astype(np.int32)

    def car2img(self, car_point):
        """
        Convert coordinates from ego-car to image space
        """

        cx, cy = car_point
        px = (cx - self.lidar_range['Left']) / self.bev_res;
        py = (self.lidar_range['Front'] - cy) / self.bev_res;
        return np.array([int(px), int(py)], np.int32)
   

    def process_lidar(self, lidar_msg):
        points = []
        #for p in pc2.read_points(lidar_msg, field_names = ("x", "y", "z", "intensity"), skip_nans=True):
        for p in pc2.read_points(lidar_msg, field_names = ("x", "y", "z", "i"), skip_nans=True):
            x, y, z, i = p[:4]
            if self.use_kitti_bag:
                points.append(np.array([-y, x, z, i]))
            else:
                points.append(np.array([x, y, z, i]))

        points = np.array(points, dtype=np.float32)
        if self.cfg.data.val.lidar_intensity_max == 255:
            points[:,3] = points[:,3] / 255.0
        
        bev_map = bev.create_bev(points, self.lidar_range_np, self.img_h, self.img_w, self.bev_res)
        return bev_map

    def process_lidar_rosnumpy(self, lidar_msg):
        cloud_array = ros_numpy.numpify(lidar_msg)
        points = np.zeros((cloud_array.shape[0], 4))
        points[:, 2] = cloud_array['z']
        points[:, 3] = cloud_array['intensity']
        if self.use_kitti_bag:
            points[:, 0] = cloud_array['y'] * (-1)
            points[:, 1] = cloud_array['x']
        else:
            points[:, 0] = cloud_array['x']
            points[:, 1] = cloud_array['y']

        if self.cfg.data.val.lidar_intensity_max == 255:
            points[:,3] = points[:,3] / 255.0

        bev_map = bev.create_bev(points, self.lidar_range_np, self.img_h, self.img_w, self.bev_res)
        return bev_map

    def gps_imu_callback(self, gps_msg, imu_msg):

        traj_hmi_forward = None
        traj_gps_forward = None
        
        stamp = gps_msg.header.stamp
        
        current_pose = self.get_current_pose(gps_msg, imu_msg)

        lidar_msg = self.get_synchronized(stamp, self.lidar_queue)
        if lidar_msg is None:
            return

        delta = (stamp - lidar_msg.header.stamp).to_sec()
        if delta > 0.5:
            ROS_WARN('[%s] GPS and Lidar timestamp diff = %f sec' % (APP_NAME, delta))
 
        # convert to BEV map
        st = time.time()
        lidar_bev = self.process_lidar(lidar_msg)
        ROS_INFO("[%s]: Lidar BEV proccessing time: %f", APP_NAME, time.time()-st)
        
        if self.use_kitti_bag:
            if self.hmi_utm_points is not None:
                traj_hmi_forward, traj_hmi_backward, search_range = self.get_guide_trajectory(current_pose, self.hmi_utm_points, self.last_hmi_idx)
                self.last_hmi_idx = search_range[0]
            
            if self.gps_utm_points is not None:
                traj_gps_forward, traj_gps_backward, tmp = self.get_guide_trajectory(current_pose, self.gps_utm_points, self.last_gps_idx)
                self.last_gps_idx = tmp[0]

        if traj_hmi_forward is not None:
            img_bev = (lidar_bev[:,:,-1] * 255).astype(np.uint8)
            
            if traj_gps_forward is not None:
                traj_gps = np.vstack([traj_gps_backward, traj_gps_forward])
                traj_gps_car = self.utm2car(current_pose, traj_gps)

            traj_hmi = np.vstack([traj_hmi_backward, traj_hmi_forward])
            traj_hmi_car = self.utm2car(current_pose, traj_hmi)
            img_hmi = self.generate_image(traj_hmi_car, rescale=1.0) 
           
            # predict trajectory
            data_dict = {'img_hmi': self.image_transform(img_hmi)}
            data_dict['lidar_bev'] = lidar_bev.transpose(2, 0, 1) # H, W, C -> C, H, W
            st = time.time()
            pred_seg, pred_traj = self.predict(data_dict)
            ROS_INFO("[%s]: Trajectory prediction inference time: %f", APP_NAME, time.time()-st)
 
            img_traj = np.zeros((self.img_h, self.img_w, 3), np.uint8)
            img_traj[...,0] = img_bev
            img_traj[...,1] = img_bev
            img_traj[...,2] = img_bev
            img_traj = self.plot_trajectory(traj_hmi_car, img_traj)
            if traj_gps_forward is not None:
                img_traj = self.plot_trajectory(traj_gps_car, img_traj, color=(0,0,255))
            
            img_traj = self.plot_trajectory(pred_traj, img_traj, color=(0,255,0))

            pred_seg *= 255
            pred_seg = cv2.resize(pred_seg, (int(self.img_w), int(self.img_h)))
            img_seg = np.zeros_like(img_traj)
            img_seg[...,1] = pred_seg
            img_seg[...,2] = img_bev
            self.img_bev = np.concatenate([img_seg, img_traj], axis=1)

            #self.publish_hmi_local_markers(msg.header, traj_hmi_car)

        if self.img_map is not None:
            self.img_gloabl_path = self.draw_current_pose(self.img_map, current_pose, search_range)

    def timer_callback(self, timer):
        if self.img_bev is not None:
            cv2.imshow('BEV', self.img_bev)
        
        if self.img_gloabl_path is not None:
            cv2.imshow('Map', self.img_gloabl_path)
        cv2.waitKey(1)

    def load_gps_path(self, csv_file):
        utm_points = self.load_csv_path(csv_file)
        self.gps_utm_points = utm_points
        self.img_map = self.draw_global_path(self.img_map, utm_points, color=(0, 0, 255))
        ROS_INFO("[%s]: Loaded %d points from GPS path", APP_NAME, utm_points.shape[0])

    def load_hmi_path(self, csv_file):
        utm_points = self.load_csv_path(csv_file)
        self.hmi_utm_points = utm_points
        self.img_map = self.draw_global_path(self.img_map, utm_points, color=(255, 0, 0))
        ROS_INFO("[%s]: Loaded %d points from HMI path", APP_NAME, utm_points.shape[0])

    def load_csv_path(self, csv_file):
        assert csv_file.exists()
        utm_points = []
        latlon_points = []
        with open(str(csv_file), 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            ss = line.split(',')
            lat, lon = float(ss[1]), float(ss[2])
            latlon_points.append(np.array([lat, lon]))
            utm_point = utm.fromLatLong(lat, lon)
            x, y = utm_point.easting, utm_point.northing
            utm_points.append(np.array([x, y]))

        latlon_points = np.array(latlon_points)
        utm_points = np.array(utm_points)
        self.lock_img_map.acquire()
        if self.utm_range is None:
            self.utm_range = {}
            xmax = np.max(utm_points[:,0])
            ymax = np.max(utm_points[:,1])
            xmin = np.min(utm_points[:,0])
            ymin = np.min(utm_points[:,1])
            self.utm_range['xmax'] = xmax
            self.utm_range['ymax'] = ymax
            self.utm_range['xmin'] = xmin
            self.utm_range['ymin'] = ymin

            self.map_w = int((xmax - xmin) / self.map_res)
            self.map_h = int((ymax - ymin) / self.map_res)

            lat_max = np.max(latlon_points[:,0])
            lat_min = np.min(latlon_points[:,0])
            lon_max = np.max(latlon_points[:,1])
            lon_min = np.min(latlon_points[:,1])
            satellite_map, _ = get_satellite_map(lat_min, lat_max, lon_min, lon_max)
            if satellite_map is not None:
                self.img_map = cv2.resize(satellite_map, (self.map_w, self.map_h))
            else:
                self.img_map = np.zeros((self.map_h, self.map_w, 3), np.uint8)

        self.lock_img_map.release()
        return utm_points

    def get_current_pose(self, gps, imu):
        lat = gps.latitude
        lon = gps.longitude
        utm_point = utm.fromLatLong(lat, lon)
        x, y = utm_point.easting, utm_point.northing

        quaternion = (
            imu.orientation.x,
            imu.orientation.y,
            imu.orientation.z,
            imu.orientation.w)
        
        euler = euler_from_quaternion(quaternion)
        roll, pitch, yaw = euler[:3] # yaw is the angle (radius) to x axis
        # convert to degree
        print('Current position: ', x, y, math.degrees(yaw))
        yaw = 0.5 * np.pi - yaw  # convert to the angles between car-heading and the y (north) axis
        return np.array([x, y, yaw])
        
    def get_guide_trajectory(self, curr_pose, utm_points, last_idx):
        # find closest point in utm_points
        search_dist = self.path_search_size
        start, end = 0, utm_points.shape[0] - 1
        if last_idx == -1:
            end = min(2 * search_dist, end)
        else:
            start = max(last_idx - search_dist, start)
            end = min(last_idx + search_dist, end)
        
        ds = curr_pose[:2] - utm_points[start:end]
        ds = np.linalg.norm(ds, axis=1)
        idx = np.argmin(ds)
        if ds[idx] > 5:
            ROS_WARN("[%s] guide path is more than 5 meters away" % APP_NAME)

        idx += start

        # get forward trajectory
        traj_len = min(self.traj_len, utm_points.shape[0] - idx)
        traj_forward = utm_points[idx:idx+traj_len]
        # get backward trajectory
        traj_start = max(idx - self.traj_len, 0)
        traj_len = min(idx - traj_start, utm_points.shape[0] - idx)
        traj_backward = utm_points[traj_start:traj_start + traj_len]
        
        return traj_forward, traj_backward, (idx, start, end)
    
    def utm2car(self, curr_pose, utm_points):
        """
        Convert to ego-car coordinate
        """
        x, y, yaw = curr_pose
        points = utm_points.copy()
        points -= curr_pose[:2]
        sin, cos = np.sin(yaw), np.cos(yaw)
        x1 = points[:,0]
        y1 = points[:,1]
        x2 = x1 * cos - y1 * sin
        y2 = x1 * sin + y1 * cos
        points[:,0] = x2
        points[:,1] = y2

        return points

    def lidar_callback(self, msg):
        stamp = msg.header.stamp 
        
        self.lock_lidar.acquire()
        self.lidar_queue[stamp] = msg
        while len(self.lidar_queue) > self.lidar_queue_size:
            del self.lidar_queue[min(self.lidar_queue)]

        self.lock_lidar.release()
    
    def generate_image(self, trajectory, rescale=1.0):
         
        thickness = int(rescale * self.road_width / self.bev_res) # 16 pixels
        img = np.zeros((int(self.img_h * rescale), int(self.img_w * rescale), 1)).astype(np.uint8)
        for i in range(trajectory.shape[0] - 1):
            x1, y1 = self.car2img(trajectory[i][:2])
            x2, y2 = self.car2img(trajectory[i+1][:2])
                        
            x1 = int(x1 * rescale)
            y1 = int(y1 * rescale)
            x2 = int(x2 * rescale)
            y2 = int(y2 * rescale)
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), thickness)

        return img
    
    def draw_global_path(self, img_map, utm_points, color=(0, 255, 0), thickness=2):
        img_points = self.utm2img(utm_points)
        for i in range(img_points.shape[0] - 1):
            x1, y1 = img_points[i, :] 
            x2, y2 = img_points[i+1, :] 
            cv2.line(img_map, (x1, y1), (x2, y2), color, thickness)
        
        return img_map

    def draw_current_pose(self, img_map_in, curr_pose, search_range, color=(0, 255, 0), thickness=2):
        img_map = img_map_in.copy() 
        target_idx, start, end = search_range
        x1, y1 = self.utm2img(self.hmi_utm_points[start])
        x2, y2 = self.utm2img(self.hmi_utm_points[target_idx])
        x3, y3 = self.utm2img(self.hmi_utm_points[end])

        cv2.line(img_map, (x1, y1), (x2, y2), (255, 255, 0), thickness)
        cv2.line(img_map, (x2, y2), (x3, y3), (255, 255, 0), thickness)

        if curr_pose is not None:
            xt, yt, yaw = curr_pose
            # head point
            xt1 = xt + int(10 * np.sin(yaw))
            yt1 = yt + int(10 * np.cos(yaw))

            x, y, yaw = self.utm2img(curr_pose)
            cv2.circle(img_map, (x, y), 5, color, thickness)
            # draw heading
            x1, y1 = self.utm2img(np.array([xt1, yt1]))
            cv2.line(img_map, (x, y), (x1, y1), color, thickness)

        return img_map

    def plot_trajectory(self, traj, img, color=(255, 0, 0), thickness=2):
        for i in range(traj.shape[0] - 1):
            x1, y1 = self.car2img(traj[i][:2])
            x2, y2 = self.car2img(traj[i+1][:2])
                            
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def publish_hmi_local_markers(self, header, points):
        #if not self.hmi_local_markers_pub.getNumSubscribers():
        #    return
        
        marker_arr = MarkerArray()
        i = 0
        for x, y in points:
            marker = Marker()
            marker.header.frame_id = header.frame_id
            marker.header.stamp = header.stamp
            marker.ns = 'hmi'
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            if self.use_kitti_bag:
                marker.pose.position.x = y
                marker.pose.position.y = -x
            marker.pose.position.z = 1.0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(0.1)
            i += 1

            marker_arr.markers.append(marker)

        self.hmi_local_markers_pub.publish(marker_arr)

def main(args):
    rospy.init_node('traj_pred_node')

    node = AppNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ShutDown")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
