[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="70%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/MIVisionX.png" /></p>

**MIVisionX Lite** toolkit is a set of comprehensive computer vision, utilities, and applications bundled into a single toolkit. AMD MIVisionX delivers highly optimized open-source implementation of the <a href="https://www.khronos.org/openvx/" target="_blank">Khronos OpenVX™ 1.0.1</a>. The toolkit allows for rapid prototyping and deployment of optimized computer vision workloads on a wide range of computer hardware, including small embedded x86 CPUs, APUs, discrete GPUs, and heterogeneous servers.

## Table of Contents

* [AMD OpenVX](#amd-openvx)
* [AMD OpenVX Extensions](#amd-openvx-extensions)
  * [OpenCV Extension](amd_openvx_extensions/amd_opencv#amd-opencv-extension)
* [Applications](#applications)
* [Samples](samples#samples)
* [Utilities](#utilities)
  * [RunCL](utilities/runcl#amd-runcl)
  * [RunVX](utilities/runvx#amd-runvx)
* [Prerequisites](#prerequisites)
* [Build & Install MIVisionX](#build--install-mivisionx-lite)
* [Verify the Installation](#verify-the-installation)
* [Docker](#docker)
* [Release Notes](#release-notes)

## AMD OpenVX

<p align="center"><img width="30%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/OpenVX_logo.png" /></p>

[AMD OpenVX 1.0.1](amd_openvx#amd-openvx-amd_openvx) is a highly optimized open source implementation of the <a href="https://www.khronos.org/openvx/" target="_blank">Khronos OpenVX™</a> computer vision specification. It allows for rapid prototyping as well as fast execution on a wide range of computer hardware, including small embedded x86 CPUs and large workstation discrete GPUs.

## AMD OpenVX Extensions
The OpenVX framework provides a mechanism to add new vision functions to OpenVX by 3rd party vendors. This project has below mentioned OpenVX [modules](amd_openvx_extensions#amd-openvx-extensions-amd_openvx_extensions) and utilities to extend [amd_openvx](amd_openvx#amd-openvx-amd_openvx) project, which contains the AMD OpenVX Core Engine.

<p align="center"><img width="5%" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/OpenCV_Logo_with_text_svg_version.svg/1920px-OpenCV_Logo_with_text_svg_version.svg.png" /></p>

* [amd_opencv](amd_openvx_extensions/amd_opencv#amd-module-for-opencv-interop-from-openvx-vx_opencv): OpenVX module that implements a mechanism to access OpenCV functionality as OpenVX kernels

## Applications
MIVisionX has several [applications](apps#applications) built on top of OpenVX modules, it uses AMD optimized libraries to build applications which can be used to prototype or used as models to develop a product.

<p align="center"><img width="60%" src="docs/images/vx-pop-app.gif" /></p>

* [Bubble Pop](apps/bubble_pop#vx-bubble-pop-sample): This sample application creates bubbles and donuts to pop using OpenVX & OpenCV functionality.

## Utilities
* [RunVX](utilities/runvx/README.md#amd-runvx): command-line utility to execute OpenVX graph described in GDF text file
* [RunCL](utilities/runcl/README.md#amd-runcl): command-line utility to build, execute, and debug OpenCL programs

## Prerequisites

### Hardware

* **CPU**: [64-bit SSE4.2 or later](https://github.com/RadeonOpenCompute/ROCm#hardware-and-software-support)
* **GPU**: [GFX7 or later](https://github.com/RadeonOpenCompute/ROCm#hardware-and-software-support) [optional]
* **APU**: [Carrizo or later](https://github.com/RadeonOpenCompute/ROCm#hardware-and-software-support) [optional]

  **Note:** Some modules in MIVisionX can be built for `CPU ONLY`. To take advantage of `Advanced Features And Modules` we recommend using `AMD GPUs` or `AMD APUs`.

### Operating System

### Windows
* Windows 10
* Windows SDK
* Visual Studio 2017 or later
* Install the latest AMD [drivers](https://www.amd.com/en/support)
* **Optional:** Install [OpenCL SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0)
* **Optional:** Install [OpenCV 3.4](https://github.com/opencv/opencv/releases/tag/3.4.0)
  * Set `OpenCV_DIR` environment variable to `OpenCV/build` folder
  * Add `%OpenCV_DIR%\x64\vc14\bin` or `%OpenCV_DIR%\x64\vc15\bin` to your `PATH`

### MacOS
* Install [Homebrew](https://brew.sh)
* Install CMake - `brew install cmake`
* **Optional:** Install [OpenCV 3.4](https://github.com/opencv/opencv/releases/tag/3.4.0) - `brew install opencv@3`
* **Optional:** Install [Apple OpenCL](https://developer.apple.com/opencl/) for your [device](https://support.apple.com/en-us/HT202823)

### Linux
* CMake 2.8 or later [download](http://cmake.org/download/)
* **Optional:** Install [ROCm OpenCL](https://rocm.github.io/ROCmInstall.html) 
* **Optional:** [OpenCV 3.4](https://github.com/opencv/opencv/releases/tag/3.4.0)
  * Set `OpenCV_DIR` environment variable to `OpenCV/build` folder
  
#### Prerequisites setup script - `MIVisionX-Lite-setup.py`
For the convenience of the developer, we here provide the setup script which will install all the dependencies required by this project.

  **NOTE:** This script only needs to be executed once. 

#### Prerequisites for running the script
* Linux distribution
  + Ubuntu - `16.04` / `18.04` / `20.04`
  + CentOS - `7` / `8`
* [ROCm supported hardware](https://rocm.github.io/hardware.html)
* [ROCm](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)

  **usage:**
  ```
  python MIVisionX-Lite-setup.py --directory [setup directory - optional]
                                 --installer [Package management tool - optional (default:apt-get) [options: Ubuntu:apt-get;CentOS:yum]]
                                 --reinstall [Remove previous setup and reinstall - optional (default:no)[options:yes/no]]
  ```

  **Note:** 
  * use `--installer yum` for **CentOS**
  * **Upgrade ROCm** with `sudo apt upgrade`

##### Refer to [Wiki](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/wiki/Suggested-development-workflow) page for developer instructions.

## Build & Install MIVisionX-Lite

### Windows

#### Using `Visual Studio`
* Install [Windows Prerequisites](#windows)
* Use `MIVisionX-Lite.sln` to build for x64 platform

### MacOS

#### Using `CMake`
* Install [macOS Prerequisites](#macos)
* Use the below commands to set up and build MIVisionX

  ```
  git clone -b openvx-1.0.1 https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git
  cd MIVisionX && mkdir build && cd build
  cmake ../
  make -j8
  sudo make install
  ```

### Linux

#### Using `MIVisionX-Lite-setup.py`

* Install [ROCm OpenCL](https://rocm.github.io/ROCmInstall.html)
* Use the below commands to set up and build MIVisionX

  ```
  git clone -b openvx-1.0.1 https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git
  cd MIVisionX
  python MIVisionX-Lite-setup.py 
  ```
  **Note:** Use `--installer yum` for **CentOS**

  ```
  mkdir build && cd build
  cmake ../
  make -j8
  sudo make install
  ```

## Verify the Installation

### Linux & MacOS
* The installer will copy 
  + Executables into `/opt/rocm/mivisionx_lite/bin` 
  + Libraries into `/opt/rocm/mivisionx_lite/lib`
  + OpenVX and module header files into `/opt/rocm/mivisionx_lite/include`
* Run samples to verify the installation

  **Canny Edge Detection**
  
  <p align="center"><img width="60%" src="samples/images/canny_image.PNG" /></p>
  
  ```
  export PATH=$PATH:/opt/rocm/mivisionx_lite/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx_lite/lib
  runvx file /opt/rocm/mivisionx_lite/samples/gdf/canny.gdf
  ```
  **Note:** More samples are available [here](samples#samples)

### Windows
* MIVisionX-Lite.sln builds the libraries & executables in the folder `MIVisionX/x64`
* Use RunVX to test the build
  ```
  ./runvx.exe file PATH_TO/MIVisionX/samples/gdf/skintonedetect.gdf
  ```

## Docker

MIVisionX provides developers with docker images for Ubuntu 16.04, Ubuntu 18.04, CentOS 7.5, & CentOS 7.6. Using docker images developers can quickly prototype and build applications without having to be locked into a single system setup or lose valuable time figuring out the dependencies of the underlying software.

### MIVisionX Docker
* [Ubuntu 16.04](https://hub.docker.com/r/mivisionx/ubuntu-16.04)
* [Ubuntu 18.04](https://hub.docker.com/r/mivisionx/ubuntu-18.04)
* [CentOS 7.5](https://hub.docker.com/r/mivisionx/centos-7.5)
* [CentOS 7.6](https://hub.docker.com/r/mivisionx/centos-7.6)

### Docker Workflow Sample on Ubuntu 18.04

#### Prerequisites
* Ubuntu `18.04`
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
sudo docker pull mivisionx/ubuntu-18.04
````

* Step 4 - *Run the docker image*
````
sudo docker run -it --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/ubuntu-18.04
````
  * Optional: Map localhost directory on the docker image
    * option to map the localhost directory with trained caffe models to be accessed on the docker image.
    * usage: -v {LOCAL_HOST_DIRECTORY_PATH}:{DOCKER_DIRECTORY_PATH} 
````
sudo docker run -it -v /home/:/root/hostDrive/ --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/ubuntu-18.04
````

**Note:** **Display option with docker**
* Using host display
````
xhost +local:root
sudo docker run -it --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host --env DISPLAY=unix$DISPLAY --privileged --volume $XAUTH:/root/.Xauthority --volume /tmp/.X11-unix/:/tmp/.X11-unix mivisionx/ubuntu-18.04:latest
````
* Test display with MIVisionX sample
````
export PATH=$PATH:/opt/rocm/mivisionx_lite/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx_lite/lib
runvx file /opt/rocm/mivisionx_lite/samples/gdf/canny.gdf 
````

## Release Notes

### Known issues
* Package (.deb & .rpm) install requires **OpenCV `v3.4.0`** to execute AMD OpenCV extensions
* If OpenCL failure occurs on macOS, set environment variable to run on CPU by default
  ```
  export AGO_DEFAULT_TARGET=CPU
  ```

### Tested configurations
* Windows 10
* Linux distribution
  + **Ubuntu** - `16.04` / `18.04` / `20.04`
  + **CentOS** - `7` / `8`
* macOS 
* ROCm: rocm-opencl-dev - `2.0.20191`
* OpenCV - [3.4.0](https://github.com/opencv/opencv/releases/tag/3.4.0)
* Dependencies for all the above packages
