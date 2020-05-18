# RALI Performance Tests with multiple epochs
This application can be used to run performance tests on RALI graphs with with multiple epochs.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `16.04` or later](https://www.microsoft.com/software-download/windows10)
* RALI library (Part of the MIVisionX toolkit)
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
 ./rali_image_tests_for_epoch <image_dataset_folder>  <test_case:0/1>  <fuction-case> <num_of_epochs> <batch_size> <processing_device=1/cpu=0>  decode_width decode_height <gray_scale:0/rgb:1> decode_shard_counts 
  ````
