[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# OpenVX Optical Flow

A sample application that runs Pyramidal Lucas-Kanade Optical Flow on a video file or live camera stream using an OpenVX graph. OpenCV is used for video decode and display.

<p align="center"><img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/optical_flow_video.gif"></p>

## Prerequisites

* MIVisionX installed (see [installation instructions](../../README.md#installation-instructions))
* OpenCV `3.4` or later

## Build

```shell
export OPENVX_DIR=/opt/rocm

mkdir opticalFlow-build && cd opticalFlow-build
cmake -DOPENVX_INCLUDES=$OPENVX_DIR/include/mivisionx \
      -DOPENVX_LIBRARIES=$OPENVX_DIR/lib/libopenvx.so \
      ../MIVisionX/apps/optical_flow/
make
```

## Run

```shell
# From a video file
./opticalFlow --video <path/to/video.mp4>

# From a live camera
./opticalFlow --live <capture-device-id>
```

Example using the included sample video (adjust the path to where MIVisionX was cloned):

```shell
./opticalFlow --video /path/to/MIVisionX/data/videos/AMD_driving_virtual_20.mp4
```
