# rocAL Introduction
Today’s deep learning applications require loading and pre-processing data efficiently to achieve high processing throughput.  This requires creating efficient processing pipelines fully utilizing the underlying hardware capabilities. Some examples are load and decode data, do a variety of augmentations, color-format conversions, etc.
Deep learning frameworks require supporting multiple data formats and augmentations to adapt to a variety of data-sets and models.

AMD ROCm Augmentation Library (rocAL) is designed to efficiently do such processing pipelines from both images and video as well as from a variety of storage formats.
These pipelines are programmable by the user using both C++ and Python APIs.

## Key Components of rocAL
*    Full processing pipeline support for data_loading, meta-data loading, augmentations, and data-format conversions for training and inference.
*    Being able to do processing on CPU or Radeon GPU (with OpenCL or HIP backend)
*    Ease of integration with framework plugins in Python
*    Support variety of augmentation operations through AMD’s ROCm Performance Primitives (RPP).
*    All available public and open-sourced under ROCm.

## Prerequisites
Refer to the [rocAL](../README.md) to follow and install pre-requisites.

## Build instructions
Follow the build instructions in [rocAL](../README.md)

## rocAL Python
*   rocAL Python package has been created using Pybind11 which enables data transfer between rocAL C++ API and Python API.
*   Module imports are made similar to other data loaders like NVidia's DALI.
*   rocal_pybind package has both PyTorch and TensorFlow framework support.
*   Various reader format support including FileReader, COCOReader, and TFRecordReader.
*   example folder contains sample implementations for each reader variation as well as sample training script for PyTorch
*   rocAL is integrated into MLPerf Resnet-50 Pytorch classification example on the ImageNet dataset.

## rocAL Python API

### amd.rocal.fn
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

### amd.rocal.pipeline
* Contains Pipeline class which has all the data needed to build and run the rocAL graph.
* Contains support for context/graph creation, verify and run the graph.
* Has data transfer functions to exchange data between frameworks and rocAL
* define_graph functionality has been implemented to add nodes to build a pipeline graph.

### amd.rocal.types
amd.rocal.types are enums exported from C++ API to python. Some examples include CPU, GPU, FLOAT, FLOAT16, RGB, GRAY, etc..

### amd.rocal.plugin.pytorch
*  Contains ROCALGenericIterator for Pytorch.
*  ROCALClassificationIterator class implements iterator for image classification and return images with corresponding labels.
*  From the above classes, any hybrid iterator pipeline can be created by adding augmentations.
*  see example [PyTorch Simple Example](./examples). Requires PyTorch.

### installing rocAL python plugin (Python 3.6)
*  Build and install RPP
*  Build and install MIVisionX which installs rocAL c++ lib
*  Go to [rocal_pybind](../rocal_pybind) folder
*  sudo ./run.sh

### Steps to run MLPerf Resnet50 classification training with rocAL on a system with MI50+ and ROCm
* Step 1: Ensure you have downloaded ILSVRC2012_img_val.tar (6.3GB) and ILSVRC2012_img_train.tar (138 GB) files and unzip into train and val folders
* Step 2: Build [MIVisionX Pytorch docker](https://github.com/ROCm/MIVisionX/tree/master/docker/pytorch)
* Step 3: Install rocAL python_pybind plugin as described above
* Step 4: Clone [MLPerf](https://github.com/rrawther/MLPerf-mGPU) branch and checkout mlperf-v1.1-rocal branch
```
git clone -b mlperf-v1.1-rocal https://github.com/rrawther/MLPerf-mGPU
```
* Step 5: Modify RN50_AMP_LARS_8GPUS_NCHW.sh or RN50_AMP_LARS_8GPUS_NHWC.sh to reflect correct path for imagenet directory
* Step 8: Run RN50_AMP_LARS_8GPUS_NCHC.sh or RN50_AMP_LARS_8GPUS_NHWC.sh
```
./RN50_AMP_LARS_8GPUS_NCHW.sh 
(or)
./RN50_AMP_LARS_8GPUS_NHWC.sh
```

### MIVisionX Pytorch Docker
* Refer to the [docker](https://github.com/ROCm/MIVisionX#docker) page for prerequisites and information on building the docker
* Step 1: Run the docker image*
````
sudo docker run -it -v <Path-To-Data-HostSystem>:/data -v /<Path-to-GitRepo>:/dockerx -w /dockerx --privileged --device=/dev/kfd --device=/dev/dri --group-add video --shm-size=4g --ipc="host" --network=host <docker-name>
````
  * Optional: Map localhost directory on the docker image
    * option to map the localhost directory with imagenet dataset folder to be accessed on the docker image.
    * usage: -v {LOCAL_HOST_DIRECTORY_PATH}:{DOCKER_DIRECTORY_PATH}
