# rocal_dataloader application
This application demonstrates a basic usage of rocAL's C API to load RAW images from the disk and modify them in different possible ways and displays the output images.
<p align="center"><img width="90%" src="https://raw.githubusercontent.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/master/docs/data/image_augmentation.png" /></p>

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `16.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library (Part of the MIVisionX toolkit)
* [OpenCV 3.1](https://github.com/opencv/opencv/releases) or higher
* ROCm Performance Primitives (RPP)

### build
  ````
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
  mkdir build
  cd build
  cmake ../
  make
  ````
### running the application
  ````
  rocal_dataloader <path-to-image-dataset>
  ````
