# rocAL Introduction
Today’s deep learning applications require loading and pre-processing data efficiently to achieve high processing throughput.  This requires creating efficient processing pipelines fully utilizing the underlying hardware capabilities. Some examples are load and decode data, do a variety of augmentations, color-format conversions, etc. 
Deep learning frameworks require supporting multiple data formats and augmentations to adapt to a variety of data-sets and models.

AMD ROCm Augmentation Library (rocAL) is designed to efficiently do such processing pipelines from both images and video as well as from a variety of storage formats.  
These pipelines are programmable by the user using both C++ and Python APIs. 

## Key Components of rocAL
*    Full processing pipeline support for data_loading, meta-data loading, augmentations, and data-format conversions for training and inference.
*    Being able to do processing on CPU or Radeon GPU (with OpenCL or HIP backend) 
*    Ease of integration with framework plugins in Python
*    Support variety of augmentation operations through AMD’s Radeon Performance Primitives (RPP).
*    All available public and open-sourced under ROCm.

## Prerequisites
Refer to the [rocAL](../README.md) to follow and install pre-requisites.

## Build instructions
Follow the build instructions in [rocAL](../README.md)

## rocAL Python
*   rocAL Python package has been created using Pybind11 which enables data transfer between rocAL C++ API and Python API.
*   Module imports are made similar to other data loaders like NVidia's DALI.
*   rali_pybind package has both PyTorch and TensorFlow framework support.
*   Various reader format support including FileReader, COCOReader, and TFRecordReader.
*   example folder contains sample implementations for each reader variation as well as sample training script for PyTorch
*   rocAL is integrated into MLPerf Resnet-50 Pytorch classification example on the ImageNet dataset.

## rocAL Python API

### amd.rali.ops
*  Contains the image augmentations & file read and decode operations which are linked to rocAL C++ API
*  All ops (listed below) are supported for the single input image and batched inputs.

|Image Augmentation | Reader and Decoder  | Geometric Ops |
| :------------------: |:--------------------:| :-------------:|
| ColorTwist          | File Reader         | CropMirrorNormalize |
| Brightness          | ImageDecoder        | Resize |
| Gamma Correction    | ImageDecoderRandomCrop        |    ResizeCrop |
| Snow                | COCOReader        |    WarpAffine |
| Rain                | TFRecordReader        |    FishEye |
| Blur                |         |    LensCorrection |
| Jitter |         |    Rotate |
| Hue     |         |    |
| Saturation |         |    |
| Fog  |         |     |
| Contrast  |         |     |
| Vignette  |         |     |
| SNPNoise  |         |     |
| Pixelate  |         |     |
| Blend  |        |     |

### amd.rali.pipeline 
* Contains Pipeline class which has all the data needed to build and run the rocAL graph.
* Contains support for context/graph creation, verify and run the graph.
* Has data transfer functions to exchange data between frameworks and rocAL
* define_graph functionality has been implemented to add nodes to build a pipeline graph.

### amd.rali.types
rali.types are enums exported from C++ API to python. Some examples include CPU, GPU, FLOAT, FLOAT16, RGB, GRAY, etc..

### amd.rali.plugin.pytorch
*  Contains RaliGenericIterator for Pytorch.
*  RaliClassificationIterator class implements iterator for image classification and return images with corresponding labels.
*  From the above classes, any hybrid iterator pipeline can be created by adding augmentations.
*  see example [PyTorch Simple Example](./examples). Requires PyTorch.

### installing rocAL python plugin (Python 3.6)
*  Build and install RPP
*  Build and install MIVisionX which installs rocAL c++ lib
*  Go to [rali_pybind](../rali_pybind) folder
*  sudo ./run.sh

### Steps to run MLPerf Resnet50 classification training with rocAL on a system with MI50 and ROCm
* Step 1: Ensure you have downloaded ILSVRC2012_img_val.tar (6.3GB) and ILSVRC2012_img_train.tar (138 GB) files and unzip into train and val folders
* Step 2: Pull and install [ROCm PyTorch Docker].(https://hub.docker.com/r/rocm/pytorch) 
```
sudo docker pull rocm/pytorch:rocm3.3_ubuntu16.04_py3.6_pytorch
```
* Step 3: Install RPP on the docker
* Step 4: Install MIVisionX on the docker
* Step 5: Install rocAL python_pybind plugin
* Step 6: Clone [MLPerf](https://github.com/rrawther/MLPerf-mGPU) branch and checkout mlperf-rali branch
```
git clone -b mlperf-rali https://github.com/rrawther/MLPerf-mGPU
```
* Step 7: Modify SMC_RN50_FP32_50E_1GPU_MI50_16GB.sh to reflect correct path for imagenet directory
* Step 8: Run SMC_RN50_FP32_50E_1GPU_MI50_16GB.sh
```
sh ./SMC_RN50_FP32_50E_1GPU_MI50_16GB.sh
```

### Steps to run MLPerf training on rali_pytorch docker
* Step 1: Ensure you have downloaded ILSVRC2012_img_val.tar (6.3GB) and ILSVRC2012_img_train.tar (138 GB) files and unzip into the train and Val folders
* Step 2: Pull and run  [MIVisionX rali_pytorch docker](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX#docker). The docker already installed with pre-built packages for rocAL
* Step 3: Clone [MLPerf](https://github.com/rrawther/MLPerf-mGPU) branch and checkout mlperf-rali branch
```
git clone -b mlperf-rali https://github.com/rrawther/MLPerf-mGPU
```
* Step 4: Modify SMC_RN50_FP32_50E_1GPU_MI50_16GB.sh to reflect correct path for imagenet directory
* Step 5: Run SMC_RN50_FP32_50E_1GPU_MI50_16GB.sh
```
sh ./SMC_RN50_FP32_50E_1GPU_MI50_16GB.sh
```

### MIVisionX Pytorch Docker
* Refer to the [docker](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX#docker) page for prerequisites and information on docker
* Step 1: Get the [docker image for mivisionx pytorch](https://hub.docker.com/r/mivisionx/pytorch-ubuntu-16.04)
```
sudo docker pull mivisionx/pytorch-ubuntu-16.04
```
* Step 2: *Run the docker image*
````
sudo docker run -it --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/pytorch-ubuntu-16.04
````
  * Optional: Map localhost directory on the docker image
    * option to map the localhost directory with imagenet dataset folder to be accessed on the docker image.
    * usage: -v {LOCAL_HOST_DIRECTORY_PATH}:{DOCKER_DIRECTORY_PATH} 
````
sudo docker run -it -v /home/:/root/hostDrive/ --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/pytorch-ubuntu-16.04
````

