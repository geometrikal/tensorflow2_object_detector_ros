# TensorFlow 2 Object Detector ROS Node

This repository is a fork of https://github.com/mohammedari/tensorflow_object_detector_ros which has been modified to work with Tensorflow 2.

The main differences are:

- loading from saved_model format
- different tensor names for the outputs due to TF2 obj det API not naming them properly
- uses Tensorflow 2.3.1 C API (included)
- needs CUDA 10.2 installed despite Tensorflow 2.3.1 claiming not to need it.

A ROS node using TensorFlow Object Detection C API.
This repository includes Tensorflow 2.3.1 library and requires CUDA 10.2 as well as CUDA 10.1

![image](./image.png)

## How to build?

### setup ROS 
see http://wiki.ros.org/Installation

### setup catkin workspace
```
mkdir -P ~/catkin_ws/src
cd ~/catkin_ws
catkin init
```

### clone this repository
```
cd ~/catkin_ws/src
git clone <repository_path>
```

### download tensorflow
Create a `tensorflow` folder in the `thirdparty` folder and save the Tensorflow 2 C API folders (`include` and `lib`) inside

Download Tensorflow C API from https://www.tensorflow.org/install/lang_c

### install prerequisite packages
Install USB camera node.
```
sudo apt-get install ros-<ros_distro>-usb-cam
```

### build
```
catkin build
```

## How to run?

Needs CUDA 10.2 added to the library path otherwise you will get an error:

_Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10_ 

```
cd ~/catkin_ws
source devel/setup.bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64 roslaunch tensorflow_object_detector object_detection.launch
```
