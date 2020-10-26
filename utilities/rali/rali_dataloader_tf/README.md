# rali_dataloader_tf application
This application demonstrates a basic usage of RALI's C API to load TfRecords from the disk and modify them in different possible ways and displays the output images.

## Build Instructions

### Pre-requisites
*  Ubuntu 16.04/18.04 Linux 
*  RALI library (Part of the MIVisionX toolkit)
*  [OpenCV 3.4+](https://github.com/opencv/opencv/releases/tag/3.4.0)
*  Google protobuf 3.11.1 or higher
*  Radeon Performance Primitives (RPP)

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
  rali_dataloader <path-to-TFRecord> <proc_dev> <decode_width> <decode_height> <batch_size> <grayscale/rgb> <dispay_on_or_off>
  ````
