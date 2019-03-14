# YOLOv2 using AMD winml (AMD NN Inference Engine)

*This project shows how to run tiny yolov2 (20 classes) with AMD's NN inference engine(Annie):*
+ A c/c++ implementation for region layer of yolov2
+ A sample for running yolov2 with Annie

---

### Preliminaries
Install MIVisionX_WinML uing the [installer](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX#build--install-mivisionx)
This will create a environment variable $MIVisionX_ROOT. 


### Step 1. Get ONNX model
Train your own YoloV2 ONNX model or get it from [onnx github](https://github.com/onnx/models/tree/master/tiny_yolov2).
ONNX version 1.3 is recommended.

### Step 2. Build the app using MIVisionX_winml_YoloV2.sln on Visual Studio.

### Step 3. Run tests
In order to run the MIVisionX_winml_YolV2.exe, you need OpenVX.dll and vx_winml.dll files on the same directory. You can either copy it from $(MIVisionX_ROOT)\bin or add the $(MIVisionX_ROOT)\bin directory to the $PATH$ environment variable.

```	
Usage: MIVisionX_winml_YoloV2.exe --image [image] --modelLoc [modelLocation]
       MIVisionX_winml_YoloV2.exe --capture 0     --modelLoc [modelLocation](Live Capture)
       MIVisionX_winml_YoloV2.exe --video [video] --modelLoc [modelLocation]
```

### Update parameters

Please update parameters (biases, object names, etc) in /source/Region.cpp, and parameters (dim, blockwd, targetBlockwd, classe, etc) in /source/AnnieYoloDetect.cpp

