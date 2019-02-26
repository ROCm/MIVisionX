# RGBDSLAM_V2-GPU

This is an implementation of RGBDSLAM-V2 that leverages the parallel processing power of AMD Radeon GPUs using OpenCL.

## Prerequisites:

* Ubuntu 16.04
* Radeon Open Compute (ROCm)
* AMD MIVisionX
* OpenCV 3
* OpenVX
* ROS Kinetic 
* g2o
* Eigen
* PCL 1.8

## Docker:

* Dockerfile.rgbdslam builds an image with all the pre-requisites installed
* Build
```bash
cd MIVisionX/apps/rgbdslam_v2/src/rgbdslam_mivisionx/docker
./build
```

* Run
```bash
cd MIVisionX/apps/rgbdslam_v2/src/rgbdslam_mivisionx/docker
./run
```

* Build rgbdslamv2_mivisionx in docker container
```
cd ~;
git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git;
cd MIVisionX/apps/rgbdslam_v2;
catkin_make;
```

## Installation Instructions

* Make a catkin workspace and clone this repo inside the source folder of the workspace

* Follow these steps to clone and install the pre-requisites of RGBDSLAM_v2 on ROS Kinetic: [Link](https://github.com/felixendres/rgbdslam_v2/wiki/Instructions-for-Compiling-Rgbdslam-(V2)-on-a-Fresh-Ubuntu-16.04-Install-(Ros-Kinetic)-in-Virtualbox)
```bash
cd MIVisionX/apps/rgbdslam_v2;
catkin_make
```

## Running RGBDSLAM_v2
* ROSbag
```
source MIVisionX/apps/rgbdslam_v2/devel/setup.bash
roslaunch rgbdslam test_settings.launch bagfile_name:='path/to/rosbag'
```
* Live
```
source MIVisionX/apps/rgbdslam_v2/devel/setup.bash
roslaunch rgbdslam rgbdslam.launch
```
