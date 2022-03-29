# rali_dataloader application
This application demonstrates a basic usage of rocAL's C API to load images from external source and add augmentations and and displays the output images.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `18.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library (Part of the MIVisionX toolkit)
* [OpenCV 3.1](https://github.com/opencv/opencv/releases) or higher
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
  rali_dataloader <path-to-image-dataset>
  ````
