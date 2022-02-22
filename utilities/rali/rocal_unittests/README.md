# rocAL Unit Tests
This application can be used to verify the functionality of the API offered by rocAL.

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
Go to MIVisionX-tests/rali-unittests
sh run-rali-unittests.sh
  ````
