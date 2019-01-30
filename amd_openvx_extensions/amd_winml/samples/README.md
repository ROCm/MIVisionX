# Sample

Get ONNX models from [ONNX Model Zoo](https://github.com/onnx/models)

## Sample - SqeezeNet

* Download the [SqueezeNet](https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz) ONNX Model
* Use [Netron](https://lutzroeder.github.io/netron/) to open the model.onnx
	* Look at Model Properties to find Input & Output Tensor Name (data_0 - input; softmaxout_1 - output)
	* Look at output tensor dimensions (n,c,h,w  - [1,1000,1,1] for softmaxout_1)
* Use the label file - Labels.txt and sample image - car.JPEG to run samples

### winML-image.gdf - Single Image Inference

This sample is in [Graph Description Format](../../../utilities/runvx#amd-runvx) (gdf)

#### usage
````
runvx.exe -v winML-image.gdf
````

**NOTE:**
Make the below changes in the `winML-image.gdf` file to run the inference

* Add full path to the car.JPEG image provided in this folder in line 11
````
read input_image FULL_PATH_TO\car.JPEG
````

* Add full path to the SqueezeNet ONNX model downloaded in line 21
````
data modelLocation = scalar:STRING,FULL_PATH_TO\squeezenet\model.onnx:view,resultWindow
````

* Add full path to the Labels.txt provided in this folder in line 34
````
data labelLocation = scalar:STRING,FULL_PATH_TO\Labels.txt
````

### winML-Live.gdf - Live Inference using a camera

This sample is in [Graph Description Format](../../../utilities/runvx#amd-runvx) (gdf)

#### usage
````
runvx.exe -v winML-Live.gdf
````

**NOTE:**
Make the below changes in the `winML-Live.gdf` file to run the inference

* Add full path to the SqueezeNet ONNX model downloaded in line 16
````
data modelLocation = scalar:STRING,FULL_PATH_TO\squeezenet\model.onnx:view,resultWindow
````

* Add full path to the Labels.txt provided in this folder in line 25
````
data labelLocation = scalar:STRING,FULL_PATH_TO\Labels.txt
````
