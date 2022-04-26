* To run the mnist_migraphx GDF, 
** Download the mnist model to this directory, then run the sample.
```
wget https://github.com/onnx/models/blob/main/vision/classification/mnist/model/mnist-8.onnx?raw=true
/opt/rocm/bin/runvx mnist_migraphx.gdf
```
