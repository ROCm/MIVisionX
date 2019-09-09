# Image Augmentation Unit Tests
This application can be used to verify the functionality of the API offered by Rali.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `16.04` or later](https://www.microsoft.com/software-download/windows10)
* RALI library (Part of the MIVisionX toolkit)
* [OpenCV 3.4+](https://github.com/opencv/opencv/releases/tag/3.4.0)
* Radeon Performance Primitives (RPP)

### build
  ````
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/rpp/lib
  mkdir build
  cd build
  cmake ../
  make 
  ````
### running the application  
  ````
  ./augmentation_unittest <image-dataset-folder> <node_idx> <display_on=0/off=1> <gpu=1/cpu=0> <rgb=1/grayscale=0> num_decode_threads
  ````
