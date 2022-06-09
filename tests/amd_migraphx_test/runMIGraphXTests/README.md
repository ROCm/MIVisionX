* This script is used for testing multiple onnx models using ```amdMIGraphXnode``` node extension for inference.
* The script takes one image for some of the models which use an imagenet dataset and one image for the mnist model.
* Models tested in this script (needed as input). All models can be found [here](https://github.com/onnx/models).
  1. mnist
  2. resnet50
  3. googlenet
  4. alexnet
  5. densenet
  6. squeezenet
  7. vgg19

* Usage:
```
mkdir build
cd build
cmake ../
make -j
./runMIGraphXTests --imagenet_image <path to image from imagenet dataset> --mnist_image <path to image from mnist dataset> --mnist <path to mnist model> --googlenet <path to googlenet model> --alexnet <path to alexnet model> --squeezenet <path to squeezenet model> --resnet50 <path to resnet50 model> --vgg19 <path to vgg19 model> --densenet <path to densenet model>
```
