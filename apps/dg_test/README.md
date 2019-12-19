# MIVisionX DGtest

MIVisionX DGtest is a tutorial program for those who are new to MIOpen & OpenVX. It runs inference on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset using MIOpen & OpenVX.
    
<p align="center">
  <img src="../../docs/images/DGtest.gif">
</p>

### Explanation
The program is divided into four parts:

##### 1. annmodule

annmodule is created by the AMD model compiler. You can convert your own trained caffemodel to openvx graph using the model-compiler. See the below section for the detailed instruction.
Once the conversion is done, the corresponding c++ codes will be generated including annmodule which contains the information about your pre-trained caffemodel.

##### 2. Userinterface

Userinterface class is for the gui functionality of the DGtest.
It has the palette and the result window to draw the image and show the result output.

##### 3. DGtest

DGtest class is where you actually run the inference.
Using annAddToGraph() function in annmodule, it adds weights, input tensor, and output tensor to the graph.
If it was successful, it preprocess the image and process the graph (runs the inference) using vxProcessGraph() function.

See the [OpenVX documentation](https://www.khronos.org/registry/OpenVX/specs/1.0/html/index.html) for detailed explanation about OpenVX API calls.

### Pre-requisites
1. Build & Install [MIVisionX](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX#build--install-mivisionx)
2. [OpenCV 3.1](https://opencv.org/opencv-3-1.html) or higher

### Build using Cmake on Linux (Ubuntu 16.04 64bit)
     mkdir build
     cd build
     cmake ..
     make

### Usage
     Usage: ./DGtest [weights.bin]\n"
     
     [weights.bin]
         The name of the weights file to be used for the inference. It is created by running caffemodel converter.
         See the belows section for using your own caffemodel.
     
### Testing with your own Caffemodel

You can test your own trained MNIST caffemodel using the [model compiler](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules/tree/develop/utils/model_compiler)
    
    1. Convert your caffemodel->NNIR->openvx using the model compiler.
    2. From the generated files, copy 
        
         annmodule.cpp
         annmodule.h
         weights.bin
         
       to the DGtest folder.
    3. Build the program again.
         make
         
### Example
    ./DGTest ../data/weights.bin
    
<p align="center">
  <img src="../../docs/images/dg_test_sample.png">
</p>

