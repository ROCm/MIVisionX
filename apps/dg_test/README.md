# AMD DGtest

The AMD DGtest is a tutorial program for those who are new to [MIVisionX](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX). It runs inference on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset using MIVisionX.

### Explanation
The program is divided into four parts:

##### 1. annmodule

annmodule is created by the AMD model compiler. You can convert your own trained caffemodel to openvx graph using the model-compiler. See the below section for the detailed instruction.
Once the conversion is done, the corresponding c++ codes will be generated including annmodule which contains the information about your pre-trained caffemodel.

##### 2. VXtensor

VXtensor class holds the openvx tensors, its attributes, and the functions related to it.
You can use readTensor() function to read-in the specified input tensor to run the inference on.
Also, you can write-out the inference result to the specified output tensor using writeTensor() function.

##### 3. DGtest

DGtest class is where you are actually running the inference.
Using annAddToGraph() function in annmodule, it adds weights, input tensor, and output tensor to the graph.
If it was successful, it will go ahead and process the graph (runs the inference) using vxProcessGraph() function.

##### 4. Argmax

Argmax class is used to read the output tensor and print out the result so that one can actually see the accuracy & performance. 
It will calculate the maximum probability from the output tensor and return the corresponding label from the label text file.

See the [OpenVX documentation](https://www.khronos.org/registry/OpenVX/specs/1.0/html/index.html) for detailed explanation about OpenVX API calls.

### Pre-requisites
1. ubuntu 16.04
2. [rocm supported hardware](https://rocm.github.io/hardware.html)
3. [rocm](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)
4. [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) with OpenCL backend
   Alternatively, you can simply run the [MIVisionX-Setup](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/MIVisionX-setup.py) which will install all the dependencies for MIOpen
5. [OpenCV 3.3](https://opencv.org/opencv-3-3.html) or higher
6. cmake git

       sudo apt-get install cmake git

### Build using Cmake on Linux (Ubuntu 16.04 64bit)
     mkdir build
     cd build
     cmake ..
     make

### Usage
     Usage: ./DGtest [weights.bin] [input-tensor] [output-tensor] [labels.txt] [imagetag.txt] \n"
     
     1. [weights.bin]
         The name of the weights file to be used for the inference. It is created by running caffemodel converter.
         See the belows section for using your own caffemodel.
     2. [input-tensor]
         The name of the input tensor to run the inference on. It is created by the img2tensor.py
     3. [output-tensor]
         The name of the ouput tensor that will contian the inference result.
     4. [labels.txt]
         The text file containing the labels of each classes.
     5. [imagetag.txt]
         The text file containing each images' directories. It is created by the img2tensor.py. 
            
### Guideline for Image Preparation & Converting it to tensor
You can prepare your own handwritten digits by using the ParseDigit application that is in this repository.
After the image preparation is done, convert it to a tensor using the python script in this repository

    python img2tensor.py -d <image_directory> -i <imagetag.txt> -o <output_tensor.f32>
         
         1. <image_directory>
              Path to the diectory containing prepared images.
         2. <imagetag.txt>
              The name of the text file that will be created.
              It contains each images' directories which will be used for DGtest's result.
         3. <output_tensor>
              The name of the tensor that will be created.
     
### Testing with your own Caffe / ONNX model

You can test your own trained MNIST caffe / ONNX model using the [model compiler](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/model_compiler)
    
    1. Convert your caffemodel->NNIR->openvx / ONNX->NNIR->openvx using the model compiler.
    2. From the generated files, copy 
        
         cmake folder
         annmodule.cpp
         annmodule.h
         weights.bin
         
       to the DGtest folder.
    3. Build the program again.
         make
         
***Make sure that the batch size of weights.bin, input tensor and image tag matches.
For example, when you trained your caffemodel/ONNX with a batch size of 64 and convert it to weights.bin file, your input tensor should also have 64 images in it.***

### Example
    ./DGTest Examples/weights.bin Examples/input.f32 Examples/output.f32 Examples/labels.txt Examples/imagelist.txt 
    
    The output images will be stored in ../Examples/Cropped folder as digits001-1.jpg, digits001-2.jpg, ... digits009-5.jpg.
    Make sure the destination folder is created before running the program.
   
