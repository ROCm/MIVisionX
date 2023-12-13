# rocAL_dataloader_mt application
This application demonstrates a basic usage of rocAL's C API to use sharded data_loader  in a multithreaded application.
<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/image_augmentation.png" /></p>

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
  rocAL_dataloader_mt <image_dataset_folder> <num_gpus(gpu:>=1)/(cpu:0)>  <num_shards> <decode_width> <decode_height> <batch_size> <shuffle> <display>
  ````
