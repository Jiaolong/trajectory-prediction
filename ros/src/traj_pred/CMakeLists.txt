cmake_minimum_required(VERSION 2.8.3)
project(traj_pred)

find_package(catkin REQUIRED COMPONENTS
    rospy
    sensor_msgs
    std_msgs
    )

catkin_python_setup()


catkin_package(
    #  INCLUDE_DIRS include
    #  LIBRARIES tf_object_detect
    #  CATKIN_DEPENDS  rospy sensor_msgs std_msgs
    #  DEPENDS system_lib
    )

catkin_install_python(PROGRAMS src/traj_pred_node.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    # include
    ${catkin_INCLUDE_DIRS}
    )
