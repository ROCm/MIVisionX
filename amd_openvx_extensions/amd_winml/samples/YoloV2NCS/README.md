# YOLOv2 for Annie (AMD NN Inference Engine)

*This project shows how to run tiny yolov2 (20 classes) with AMD's NN inference engine(Annie):*
+ A c/c++ implementation for region layer of yolov2
+ A sample for running yolov2 with Annie

---

### Step 1. Get YoloV2 ONNX model
Train your own YoloV2 ONNX model or get it from [onnx github](https://github.com/onnx/models/tree/master/tiny_yolov2). ONNX version 1.3 is recommended.

### Step 2. Run tests

		Usage: YoloV2NCS.exe --image [image] --modelLoc [modelLocation]
		       YoloV2NCS.exe --capture 0     --modelLoc [modelLocation](Live Capture)
		       YoloV2NCS.exe --video [video] --modelLoc [modelLocation]

This runs inference and detections on a given input (image, camera capture, video).

### Update parameters

Please update parameters (biases, object names, etc) in Region.cpp, and parameters (dim, blockwd, targetBlockwd, classe, etc) in AnnieObjectWrapper.cpp
