[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.svg?branch=master)](https://travis-ci.org/GPUOpen-ProfessionalCompute-Libraries/MIVisionX)

# MIVisionX

MIVisionX toolkit is a comprehensive computer vision and machine intelligence libraries, utilities and applications bundled into a single toolkit.

* [AMD OpenVX](amd_openvx#amd-openvx-amd_openvx)
* [AMD OpenVX Extensions](amd_openvx_extensions#amd-openvx-extensions-amd_openvx_extensions)
  * [Loom 360 Stitch Library](amd_openvx_extensions/amd_loomsl#radeon-loom-stitching-library-vx_loomsl)
  * [Neural Net Library](amd_openvx_extensions/amd_nn#openvx-neural-network-extension-library-vx_nn)
  * [OpenCV Extensions](amd_openvx_extensions/amd_opencv#amd-module-for-opencv-interop-from-openvx-vx_opencv)
* [Applications](apps#applications)
* [Neural Net Model Compiler](model_compiler#model-compiler)
* [Samples](samples#samples)
* [Toolkit](toolkit#mivisionx-toolkit)
* [Utilities](utilities#utilities)
  * [Inference Generator](utilities/inference_generator#inference-generator)
  * [Radeon Loom Shell](utilities/loom_shell#radeon-loomshell)
  * [RunCL](utilities/runcl#amd-runcl)
  * [RunVX](utilities/runvx#amd-runvx)
* [Pre-requisites](#pre-requisites)
* [Build MIVisionX](#build-mivisionx)
* [Docker](#docker)
* [Release Notes](#release-notes)

## AMD OpenVX
AMD OpenVX (amd_openvx) is a highly optimized open source implementation of the <a href="https://www.khronos.org/openvx/" target="_blank">Khronos OpenVX</a> computer vision specification. It allows for rapid prototyping as well as fast execution on a wide range of computer hardware, including small embedded x86 CPUs and large workstation discrete GPUs.

## AMD OpenVX Extensions
The OpenVX framework provides a mechanism to add new vision functions to OpenVX by 3rd party vendors. This project has below OpenVX modules and utilities to extend [amd_openvx](amd_openvx#amd-openvx-amd_openvx) project, which contains the AMD OpenVX Core Engine.

* [amd_loomsl](amd_openvx_extensions/amd_loomsl#radeon-loom-stitching-library-vx_loomsl): AMD Radeon LOOM stitching library for live 360 degree video applications
* [amd_nn](amd_openvx_extensions/amd_nn#openvx-neural-network-extension-library-vx_nn): OpenVX neural network module
* [amd_opencv](amd_openvx_extensions/amd_opencv#amd-module-for-opencv-interop-from-openvx-vx_opencv): OpenVX module that implements a mechanism to access OpenCV functionality as OpenVX kernels

## Applications
MIVisionX has a number of applications (apps) built on top of OpenVX modules, it uses AMD optimized libraries to build applications which can be used to prototype or used as models to develop a product.  

* Cloud Inference Application
* External Applications

## Model Compiler
[Model compiler](model_compiler#convert-onnx-models-into-amd-nnir-and-openvx-code) generates efficient inference libraries from pre-trained neural net models.

## Toolkit

MIVisionX Toolkit, is a comprehensive set of help tools for neural net creation, development, training and deployment. The Toolkit provides you with help tools to design, develop, quantize, prune, retrain, and infer your neural network work in any framework. The Toolkit is designed to help you deploy your work to any AMD or 3rd party hardware, from embedded to servers.

MIVisionX provides you with tools for accomplishing your tasks throughout the whole neural net life-cycle, from creating a model to deploying them for your target platforms.

## Utilities
* [inference_generator](utilities/inference_generator#inference-generator): generate inference library from pre-trained CAFFE models
* [loom_shell](utilities/loom_shell/README.md#radeon-loomsh): an interpreter to prototype 360 degree video stitching applications using a script
* [RunVX](utilities/runvx/README.md#amd-runvx): command-line utility to execute OpenVX graph described in GDF text file
* [RunCL](utilities/runcl/README.md#amd-runcl): command-line utility to build, execute, and debug OpenCL programs

If you're interested in Neural Network Inference, start with the sample cloud inference application in apps folder.

Inference Application Development Workflow |  Sample Inference Application
:-------------------------:|:-------------------------:
[![Block-Diagram-Inference-Workflow](docs/images/block_diagram_inference_workflow.png)](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules/wiki#neural-network-inference-workflow-for-caffe-users)  |  [![Block-Diagram-Inference-Sample](docs/images/block_diagram_inference_sample.png)](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules/wiki#getting-started-with-neural-network-inference-sample)

Refer to [Wiki](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules/wiki) page for further details.

## Pre-requisites
* CPU: SSE4.1 or above CPU, 64-bit
* GPU: Radeon Instinct or Vega Family of Products (16GB recommended)
  * Linux: install [ROCm](https://rocm.github.io/ROCmInstall.html) with OpenCL development kit
  * Windows: install the latest drivers and OpenCL SDK [download](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases)
* CMake 2.8 or newer [download](http://cmake.org/download/)
* Qt Creator for [annInferenceApp](apps/cloud_inference/client_app/README.md)
* [protobuf](https://github.com/google/protobuf) for [inference_generator](utilities/inference_generator#inference-generator)
  * install `libprotobuf-dev` and `protobuf-compiler` needed for vx_nn
* OpenCV 3 (optional) [download](https://github.com/opencv/opencv/releases) for vx_opencv
  * Set OpenCV_DIR environment variable to OpenCV/build folder
  
### Pre-requisites setup script - MIVisionX-setup.py

For convenience of the developer, we here provide the setup script which will install all the dependencies required by this project. 

#### Prerequisites for running the scripts
1. Ubuntu `16.04`/`18.04` or CentOS `7.5`/`7.6`
2. [ROCm supported hardware](https://rocm.github.io/hardware.html)
3. [ROCm](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)

**MIVisionX-setup.py** - This scipts builds all the prerequisites required by MIVisionX. The setup script creates a deps folder and installs all the prerequisites, this script only needs to be executed once. If -d option for directory is not given the script will install deps folder in '~/' directory by default, else in the user specified folder.

usage:

````
python MIVisionX-setup.py -s [sudo password - required] -d [setup directory - optional (default:~/)] -m [MIOpen Version - optional (default:1.6.0)]
```` 
Refer to [Wiki](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules/wiki) page for developer instructions.

## Build MIVisionX

### Using CMake on Linux (Ubuntu 16.04/18.04 64-bit / CentOS 7.5/7.6) with ROCm
* Install [ROCm](https://rocm.github.io/ROCmInstall.html)
* git clone, build and install other ROCm projects (using `cmake` and `% make install`) in the below order for vx_nn.
  * [rocm-cmake](https://github.com/RadeonOpenCompute/rocm-cmake)
  * [MIOpenGEMM](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM)
  * [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) -- make sure to use `-DMIOPEN_BACKEND=OpenCL` option with cmake
* install [protobuf](https://github.com/protocolbuffers/protobuf/releases/tag/v3.5.2)
* install [OpenCV](https://github.com/opencv/opencv/releases/tag/3.3.0)
* git clone this project using `--recursive` option so that correct branch of the deps project is cloned automatically.
* build and install (using `cmake` and `% make install`)
  * executables will be placed in `bin` folder
  * libraries will be placed in `lib` folder
  * the installer will copy all executables into `/opt/rocm/mivisionx/bin` and libraries into `/opt/rocm/mivisionx/lib`
  * the installer also copies all the OpenVX and module header files into `/opt/rocm/mivisionx/include` folder
* add the installed library path to LD_LIBRARY_PATH environment variable (default `/opt/rocm/mivisionx/lib`)
* add the installed executable path to PATH environment variable (default `/opt/rocm/mivisionx/bin`)

### Using Setup Script and CMake on Linux (Ubuntu 16.04/18.04 64-bit / CentOS 7.5/7.6) with ROCm
* Install [ROCm](https://rocm.github.io/ROCmInstall.html)
* Use the below commands to setup and build MIVisionX
````
git clone --recursive https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git
cd MIVisionX
python MIVisionX-setup.py -s [sudo password - required] -d [setup directory - optional (default:~/)] -m [MIOpen Version - optional (default:1.6.0)]
mkdir build
cd build
cmake ../
make -j8
````

#### Build annInferenceApp using Qt Creator
* build [annInferenceApp.pro](apps/cloud_inference/client_app/annInferenceApp.pro) using Qt Creator
* or use [annInferenceApp.py](apps/cloud_inference/client_app/annInferenceApp.py) for simple tests

#### Build Radeon LOOM using Visual Studio Professional 2013 on 64-bit Windows 10/8.1/7
* Use [loom.sln](amd_openvx_extensions/amd_loomsl/vx_loomsl.sln) to build x64 platform

## Docker

MIVisionX provides developers with docker images for Ubuntu 16.04, Ubuntu 18.04, CentOS 7.5, & CentOS 7.6. Using docker images developers can quickly prototype and build applications without having to be locked into a single system setup or lose valuable time figuring out the dependencies of the underlying software.

### MIVisionX Docker
* [Ubuntu 14.04](https://hub.docker.com/r/kiritigowda/mivisionx-ubuntu-14.04)
* [Ubuntu 16.04](https://hub.docker.com/r/kiritigowda/mivisionx-ubuntu-16.04)
* [Ubuntu 18.04](https://hub.docker.com/r/kiritigowda/mivisionx-ubuntu-18.04)
* [CentOS 7.5](https://hub.docker.com/r/kiritigowda/centos)
* [CentOS 7.6](https://hub.docker.com/r/kiritigowda/centos)

### Docker Workflow Sample on Ubuntu 16.04

#### Prerequisites
* Ubuntu `16.04`
* [rocm supported hardware](https://rocm.github.io/hardware.html)

#### Workflow
* Step 1 - *Install rocm-dkms*
````
sudo apt update
sudo apt dist-upgrade
sudo apt install libnuma-dev
sudo reboot
````
````
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms
sudo reboot
````

* Step 2 - *Setup Docker*
````
sudo apt-get install curl
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
apt-cache policy docker-ce
sudo apt-get install -y docker-ce
sudo systemctl status docker
````

* Step 3 - *Get Docker Image*
````
sudo docker pull kiritigowda/mivisionx-ubuntu-16.04
````

* Step 4 - *Run the docker image*
````
sudo docker run -it --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host kiritigowda/mivisionx-ubuntu-16.04
````
  * Optional: Map localhost directory on the docker image
    * option to map the localhost directory with trained caffe models to be accessed on the docker image.
    * usage: -v {LOCAL_HOST_DIRECTORY_PATH}:{DOCKER_DIRECTORY_PATH} 
````
sudo docker run -it -v /home/:/root/hostDrive/ --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host kiritigowda/mivisionx-ubuntu-16.04
````

## Release Notes

### Supported Neural Net Layers

Layer name |					
------|			
Activation|				
Argmax|			
Batch Normalization|
Concat|
Convolution|			
Deconvolution|			
Fully Connected|			
Local Response Normalization (LRN)|
Pooling|
Scale|
Slice|
Softmax|
Tensor Add|
Tensor Convert Depth|
Tensor Convert from Image|
Tensor Convert to Image|
Tensor Multiply|
Tensor Subtract|
Upsample Nearest Neighborhood|

### Known issues
* Package (.deb & .rpm) install requires OpenCV v3.3.0 to execute AMD OpenCV extensions

### Tested configurations
* Linux: Ubuntu - `16.04`/`18.04` & CentOS - `7.5`/`7.6`
* ROCm: rocm-dkms - `2.0.89`
* rocm-cmake - [github master:ac45c6e](https://github.com/RadeonOpenCompute/rocm-cmake/tree/master)
* MIOpenGEMM - [1.1.5](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5)
* MIOpen - [1.7.0](https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/1.7.0)
* Protobuf - [V3.5.2](https://github.com/protocolbuffers/protobuf/releases/tag/v3.5.2)
* OpenCV - [3.3.0](https://github.com/opencv/opencv/releases/tag/3.3.0)
* Dependencies for all the above packages
