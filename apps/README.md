# Applications

MIVisionX has a number of applications built on top of OpenVX and its modules, it uses AMD optimized libraries to build applications which can be used as prototypes or used as models to develop products.

## Cloud Application
This sample [application](./cloud_inference#cloud-inference-application) does inference using a client-server system.

[![Radeon Inference](../docs/images/inferenceVideo.png)](http://www.youtube.com/watch?v=0GLmnrpMSYs)

## DG Test
This sample [application](./dg_test#amd-dgtest) is used to recognize hand written digits.

<p align="center">
  <img src="../docs/images/DGtest.gif">
</p>

## Image Augmentation
This sample [application](./image_augmentation#image-augmentation-application) demonstrates a basic usage of RALI's C API to load JPEG images from the disk and modify them in different possible ways and displays the output images.

<p align="center">
	<img width="90%" src="../docs/images/image_augmentation.png" />
</p>

# MIVisionX Inference Analyzer

[MIVisionX Inference Analyzer Application](./mivisionx_inference_analyzer#mivisionx-python-inference-analyzer) using pre-trained `ONNX`/`NNEF`/`Caffe` models to analyze and summarize images.

<p align="center"><img width="60%" src="../docs/images/inference_analyzer.gif" /></p>

## MIVisionX OpenVX Classsification
This sample [application](./mivisionx_openvx_classifier/README.md) shows how to run supported pre-trained caffe models with MIVisionX RunTime.

<p align="center"> <img src="../docs/images/mivisionx_openvx_classifier_imageClassification.png"></p>

# MIVisionX Validation Tool

[MIVisionX ML Model Validation Tool](./mivisionx_validation_tool#mivisionx-python-ml-model-validation-tool) using pre-trained `ONNX`/`NNEF`/`Caffe` models to analyze, summarize, & validate.

<p align="center"><img width="100%" src="../docs/images/validation-2.png" /></p>

## MIVisionX WinML Classification
This sample [application](./mivisionx_winml_classifier/README.md) shows how to run supported ONNX models with MIVisionX RunTime on Windows.

<p align="center"> <img src="./mivisionx_winml_classifier/images/MIVisionX-ImageClassification-WinML.png"> </p>

## MIVisionX WinML YoloV2
This sample [application](./mivisionx_winml_yolov2#yolov2-using-amd-winml-extension) shows how to run tiny yolov2(20 classes) with MIVisionX RunTime on Windows.

<p align="center"> <img src="./mivisionx_winml_yolov2/image/cat-yolo.jpg"> </p>

## External Application

* [MIVisionX-Classifier](https://github.com/kiritigowda/MIVisionX-Classifier) - This application runs know CNN image classifiers on live/pre-recorded video stream.
* [YOLOv2](https://github.com/kiritigowda/YoloV2NCS) - Run tiny yolov2 (20 classes) with AMD's MIVisionX
* [Traffic Vision](https://github.com/srohit0/trafficVision#traffic-vision) - This app detects cars/buses in live traffic at a phenomenal 50 frames/sec with HD resolution (1920x1080) using deep learning network Yolo-V2. The model used in the app is optimized for inferencing performance on AMD-GPUs using MIVisionX toolkit.
* [RGBDSLAMv2-MIVisionX](https://github.com/ICURO-AI-LAB/RGBDSLAMv2-MIVisionX) - This is an implementation of RGBDSLAM_V2 that utilizes AMD MIVisionX for feature detection and ROCm OpenCL for offloading computations to Radeon GPUs. This application is used to create 3D maps using RGB-D Cameras.
