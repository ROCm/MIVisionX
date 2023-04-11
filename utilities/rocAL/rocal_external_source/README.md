# rocal_external_source application
This application demonstrates a basic usage of rocAL's C API to load images from external source and add augmentations and and displays the output images.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `18.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library (Part of the MIVisionX toolkit)
* [OpenCV 3.1](https://github.com/opencv/opencv/releases) or higher
* ROCm Performance Primitives (RPP)

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
  rocal_external_source <image_dataset_folder/video_file> <processing_device=1/cpu=0>  decode_width decode_height batch_size gray_scale/rgb/rgbplanar display_on_off external_source_mode

  ````
