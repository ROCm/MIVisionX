# rocAL Performance Tests
This application is used to run performance tests on the rocAL API for graphs of depth size 1.


## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `16.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library (Part of the MIVisionX toolkit)
* [OpenCV 3.4+](https://github.com/opencv/opencv/releases/tag/3.4.0)
* Radeon Performance Primitives (RPP)

### build
  ````
  mkdir build
  cd build
  cmake ../
  make 
  ````
### running the application  
  ````
rali_performance_tests [test image folder] [image width] [image height] [test case] [batch size] [0 for CPU, 1 for GPU] [0 for grayscale, 1 for RGB]
  ````
