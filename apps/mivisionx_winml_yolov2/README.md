# YoloV2 using AMD WinML Extension

*This application shows how to run tiny yolov2 (20 classes) with MIVisionX RunTime:*
+ A c/c++ implementation for region layer of yolov2
+ A sample for running yolov2 with MIVisionX


## Pre-requisites

* Windows 10, [version `1809` or later](https://www.microsoft.com/software-download/windows10)
* Windows SDK, build `17763` or later
* Visual Studio 2017, [version `15.7.4` or later](https://developer.microsoft.com/en-us/windows/downloads)
    * Visual Studio extension for C++/WinRT
* Install [OpenCL SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0)
* [OpenCV 3.4+](https://github.com/opencv/opencv/releases/tag/3.4.0)
  * Set `OpenCV_DIR` environment variable to `OpenCV/build` folder
  * Add `OpenCV_DIR\x64\vc14\bin` or `OpenCV_DIR\x64\vc15\bin` to your `PATH`


## Run

### Step 1. Get ONNX model
Train your own YoloV2 ONNX model or get it from [onnx github](https://github.com/onnx/models/tree/master/tiny_yolov2).
ONNX version 1.3 is recommended.

### Step 2. Build the app using MIVisionX_winml_YoloV2.sln on Visual Studio.

### Step 3. Run tests
Open up the command line prompt or Windows Powershell and use the following commands to run the tests.
#### Usage:

* Image

      .\MIVisionX_winml_YoloV2.exe --image [image - required] --modelLoc [modelLocation - required] --confidence [1-100 - optional, default: 20]
* Camera Capture

      .\MIVisionX_winml_YoloV2.exe --capture [0 - required] --modelLoc [modelLocation - required] --confidence [1-100 - optional, default: 20]

* Video

      .\MIVisionX_winml_YoloV2.exe --video [video - required] --modelLoc [modelLocation - required] --confidence [1-100 - optional, default: 20]

The confidence parameter is an optional parameter which sets the confidence level of detection.
Lower the confidence level if the detection is not good enough.

### Example

.\MIVisionX_winml_YoloV2.exe --image image\cat.jpg --modelLoc model.onnx


<p align="center">
  <img src="./image/cat-yolo.jpg">
</p>

### Update parameters

Please update parameters (biases, object names, etc) in /source/Region.cpp, and parameters (dim, blockwd, targetBlockwd, classes, etc) in /source/AnnieYoloDetect.cpp
