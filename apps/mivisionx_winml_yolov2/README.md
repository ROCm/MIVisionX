# YoloV2 using AMD WinML Extension

*This project shows how to run tiny yolov2 (20 classes) with MIVisionX RunTime:*
+ A c/c++ implementation for region layer of yolov2
+ A sample for running yolov2 with Annie

---

## Pre-requisites

### Option 1: Using pre-built installer
[MIVisionX_WinML-installer.msi](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/releases) 

* The pre-requisites and instructions are under the tab "Install Packages on Windows".

### Option 2: Build MIVisionX winML extension library

[MIVisionX winML extension library](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/amd_openvx_extensions/amd_winml#amd-winml-extension)

* Use the above link for building instructions.


## Run

### Step 1. Get ONNX model
Train your own YoloV2 ONNX model or get it from [onnx github](https://github.com/onnx/models/tree/master/tiny_yolov2).
ONNX version 1.3 is recommended.

### Step 2. Build the app using MIVisionX_winml_YoloV2.sln on Visual Studio.

### Step 3. Run tests
In order to run the MIVisionX_winml_YolV2.exe, you need OpenVX.dll and vx_winml.dll files on the same directory. You can either copy it from $(MIVisionX_ROOT)\bin or add the $(MIVisionX_ROOT)\bin directory to the $PATH$ environment variable.

#### Usage:

* Image

      MIVisionX_winml_YoloV2.exe --image [image] --modelLoc [modelLocation] --confidence(default = 20)
* Camera Capture

      MIVisionX_winml_YoloV2.exe --capture 0     --modelLoc [modelLocation](Live Capture) --confidence(default = 20)
* Video

      MIVisionX_winml_YoloV2.exe --video [video] --modelLoc [modelLocation] --confidence(default = 20)

The confidence parameter is an optional parameter which sets the confidence level of detection.
Lower the confidence level if the detection is not good enough.

### Example

MIVisionX_winml_YoloV2.exe --image image\cat.jpg --modelLoc model.onnx


<p align="center">
  <img src="./image/cat-yolo.jpg">
</p>

### Update parameters

Please update parameters (biases, object names, etc) in /source/Region.cpp, and parameters (dim, blockwd, targetBlockwd, classes, etc) in /source/AnnieYoloDetect.cpp

