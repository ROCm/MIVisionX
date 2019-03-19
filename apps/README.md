# Applications

MIVisionX has a number of applications built on top of OpenVX modules, it uses AMD optimized libraries to build applications which can be used to prototype or used as models to develop a product.

## Cloud Application

[![Radeon Inference](../docs/images/inferenceVideo.png)](http://www.youtube.com/watch?v=0GLmnrpMSYs)

## DG Test

A simple application used to recognize hand written digits using openvx.

<p align="center">
  <img src="../docs/images/digits1.png">
  <img src="../docs/images/digits2.png">
  <img src="../docs/images/DGtest.png">
</p>

## MIVisionX RGBDSLAM_v2

A Visual SLAM framework used to create 3D maps for applications such as robotics, mapping terrain, and exploration of unknown territory. This particular implementation utilizes OpenVX and OpenCL to take advantage of AMD Radeon GPUs.

RGBDSLAM_v2 original framework: https://github.com/felixendres/rgbdslam_v2

[![MIVisionX RGBDSLAM_v2](../docs/images/mivisionx_rgbdslamv2.png)](https://www.youtube.com/watch?v=zO-ZWHcFcFY)

## MIVisionX WinML YoloV2

<p align="center">
  <img src="./mivisionx_winml_yolov2/image/cat-yolo.jpg">
</p>

## External Application
* [MIVisionX-Classifier](https://github.com/kiritigowda/MIVisionX-Classifier) - This application runs know CNN image classifiers on live/pre-recorded video stream.
* [YOLOv2](https://github.com/kiritigowda/YoloV2NCS) - Run tiny yolov2 (20 classes) with AMD's MIVisionX
* [Traffic Vision](https://github.com/srohit0/trafficVision#traffic-vision) - This app detects cars/buses in a live traffic at a phenomenal 50 frames/sec with HD resolution (1920x1080) using deep learning network Yolo-V2. The model unsed in the app is optimized for inferencing performnce on AMD-GPUs using MIVisionX toolkit.
