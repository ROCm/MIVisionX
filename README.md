[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.svg?branch=master)](https://travis-ci.org/GPUOpen-ProfessionalCompute-Libraries/MIVisionX)

<p align="center"><img width="70%" src="docs/images/MIVisionX.png" /></p>

MIVisionX Lite toolkit is a set of comprehensive computer vision, utilities, and applications bundled into a single toolkit. AMD MIVisionX delivers highly optimized open source implementation of the <a href="https://www.khronos.org/openvx/" target="_blank">Khronos OpenVX™ 1.0.1</a>. The toolkit allows for rapid prototyping and deployment of optimized workloads on a wide range of computer hardware, including small embedded x86 CPUs, APUs, discrete GPUs, and heterogeneous servers.

* [AMD OpenVX](#amd-openvx)
* [AMD OpenVX Extensions](#amd-openvx-extensions)
  * [OpenCV Extension](amd_openvx_extensions/amd_opencv#amd-opencv-extension)
* [Applications](#applications)
* [Samples](samples#samples)
* [Utilities](#utilities)
  * [RunCL](utilities/runcl#amd-runcl)
  * [RunVX](utilities/runvx#amd-runvx)
* [Prerequisites](#prerequisites)
* [Build & Install MIVisionX](#build--install-mivisionx)
* [Verify the Installation](#verify-the-installation)
* [Docker](#docker)
* [Release Notes](#release-notes)

## AMD OpenVX 1.0.1

<p align="center"><img width="30%" src="https://upload.wikimedia.org/wikipedia/en/thumb/d/dd/OpenVX_logo.svg/1920px-OpenVX_logo.svg.png" /></p>

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
* CPU: [SSE4.1 or above CPU, 64-bit](https://rocm.github.io/hardware.html)
* GPU: [GFX7 or above](https://rocm.github.io/hardware.html) [optional]
* APU: Carrizo or above [optional]

**Note:** Some modules in MIVisionX can be built for CPU only. To take advantage of advanced features and modules we recommend using AMD GPUs or AMD APUs.

### Windows
* Windows 10
* Windows SDK
* Visual Studio 2017 and above
* Install the latest AMD [drivers](https://www.amd.com/en/support)
* Install [OpenCL SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0)
* Install [OpenCV 3.4](https://github.com/opencv/opencv/releases/tag/3.4.0)
  * Set `OpenCV_DIR` environment variable to `OpenCV/build` folder
  * Add `%OpenCV_DIR%\x64\vc14\bin` or `%OpenCV_DIR%\x64\vc15\bin` to your `PATH`

### Linux
* Install [ROCm OpenCL](https://rocm.github.io/ROCmInstall.html) 
* CMake 2.8 or newer [download](http://cmake.org/download/)
* [OpenCV 3.4](https://github.com/opencv/opencv/releases/tag/3.4.0)
  * Set `OpenCV_DIR` environment variable to `OpenCV/build` folder
  
#### Prerequisites setup script for Linux - `MIVisionX-Lite-setup.py`

For the convenience of the developer, we here provide the setup script which will install all the dependencies required by this project.

**MIVisionX-Lite-setup.py** builds all the prerequisites required by MIVisionX. The setup script creates a deps folder and installs all the prerequisites, this script only needs to be executed once. If directory option is not given, the script will install deps folder in the home directory(~/) by default, else in the user specified location.

##### Prerequisites for running the script
1. Ubuntu `16.04`/`18.04` or CentOS `7.5`/`7.6`
2. X Window
3. [ROCm supported hardware](https://rocm.github.io/hardware.html)
4. [ROCm](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)

**usage:**
````
python MIVisionX-Lite-setup.py --directory [setup directory - optional]
                               --installer [Package management tool - optional (default:apt-get) [options: Ubuntu:apt-get;CentOS:yum]]
                               --reinstall [Remove previous setup and reinstall (default:no)[options:yes/no]]
````

**Note:** use `--installer yum` for **CentOS**
**Note:** Upgrade ROCm with `sudo apt upgrade`

##### Refer to [Wiki](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/wiki/Suggested-development-workflow) page for developer instructions.

## Build & Install MIVisionX

### Windows

#### Using .msi packages

* [MIVisionX-Lite-installer.msi](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/releases): MIVisionX

#### Using `Visual Studio 2017` on 64-bit `Windows 10`
* Install [Windows Prerequisites](#windows)
* Use `MIVisionX-Lite.sln` to build for x64 platform

### Linux

#### Using `apt-get`/`yum`

##### Prerequisites
1. Ubuntu `16.04`/`18.04` or CentOS `7.5`/`7.6`
2. [ROCm supported hardware](https://rocm.github.io/hardware.html)
3. [ROCm OpenCL](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)

###### Ubuntu
````
sudo apt-get install mivisionx_lite
````
###### CentOS
````
sudo yum install mivisionx_lite
````
 
 **Note:**
  * source code will not available with apt-get/yum install
  * executables placed in `/opt/rocm/mivisionx_lite/bin` and libraries in `/opt/rocm/mivisionx_/lib`
  * OpenVX and module header files into `/opt/rocm/mivisionx_lite/include`
  * Samples placed in `/opt/rocm/mivisionx`
  * Package (.deb & .rpm) install requires OpenCV v3.4.0 to execute AMD OpenCV extensions

#### Using `MIVisionX-Lite-setup.py` and `CMake` on Linux (Ubuntu `16.04`/`18.04` or CentOS `7.5`/`7.6`) with ROCm

* Install [ROCm OpenCL](https://rocm.github.io/ROCmInstall.html)
* Use the below commands to setup and build MIVisionX

````
git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git
cd MIVisionX
git checkout openvx-1.0.1
````

```
python MIVisionX-Lite-setup.py --directory [setup directory - optional]
                               --installer [Package management tool - optional (default:apt-get) [options: Ubuntu:apt-get;CentOS:yum]]
                               --reinstall [Remove previous setup and reinstall (default:no)[options:yes/no]]
```
**Note:** Use `--installer yum` for **CentOS**

````
mkdir build
cd build
cmake ../
make -j8
sudo make install
````
  **Note:**
   * the installer will copy all executables into `/opt/rocm/mivisionx_lite/bin` and libraries into `/opt/rocm/mivisionx_lite/lib`
   * the installer also copies all the OpenVX and module header files into `/opt/rocm/mivisionx_lite/include` folder

## Verify the Installation

### Linux
* The installer will copy all executables into `/opt/rocm/mivisionx_lite/bin` and libraries into `/opt/rocm/mivisionx_lite/lib`
* The installer also copies all the OpenVX and OpenVX module header files into `/opt/rocm/mivisionx_lite/include` folder
* Apps, Samples, & Documents are placed into `/opt/rocm/mivisionx`
* Run samples to verify the installation
  * **Canny Edge Detection**
  
  <p align="center"><img width="60%" src="samples/images/canny_image.PNG" /></p>
  
  ````
  export PATH=$PATH:/opt/rocm/mivisionx_lite/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx_lite/lib
  runvx /opt/rocm/mivisionx_lite/samples/gdf/canny.gdf 
  ````
**Note:** More samples are available [here](samples#samples)

### Windows
* MIVisionX-Lite.sln builds the libraries & executables in the folder `MIVisionX/x64`
* Use RunVX to test the build
```
./runvx.exe PATH_TO/MIVisionX/samples/gdf/skintonedetect.gdf
```

## Docker

MIVisionX provides developers with docker images for Ubuntu 16.04, Ubuntu 18.04, CentOS 7.5, & CentOS 7.6. Using docker images developers can quickly prototype and build applications without having to be locked into a single system setup or lose valuable time figuring out the dependencies of the underlying software.

### MIVisionX Docker
* [Ubuntu 16.04](https://hub.docker.com/r/mivisionx/ubuntu-16.04)
* [Ubuntu 18.04](https://hub.docker.com/r/mivisionx/ubuntu-18.04)
* [CentOS 7.5](https://hub.docker.com/r/mivisionx/centos-7.5)
* [CentOS 7.6](https://hub.docker.com/r/mivisionx/centos-7.6)

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
sudo docker pull mivisionx/ubuntu-16.04
````

* Step 4 - *Run the docker image*
````
sudo docker run -it --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/ubuntu-16.04
````
  * Optional: Map localhost directory on the docker image
    * option to map the localhost directory with trained caffe models to be accessed on the docker image.
    * usage: -v {LOCAL_HOST_DIRECTORY_PATH}:{DOCKER_DIRECTORY_PATH} 
````
sudo docker run -it -v /home/:/root/hostDrive/ --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/ubuntu-16.04
````

**Note:** **Display option with docker**
* Using host display
````
xhost +local:root
sudo docker run -it --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host --env DISPLAY=unix$DISPLAY --privileged --volume $XAUTH:/root/.Xauthority --volume /tmp/.X11-unix/:/tmp/.X11-unix mivisionx/ubuntu-16.04:latest
````
* Test display with MIVisionX sample
````
export PATH=$PATH:/opt/rocm/mivisionx/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib
runvx /opt/rocm/mivisionx/samples/gdf/canny.gdf 
````

## Release Notes

### Known issues
* Package (.deb & .rpm) install requires **OpenCV `v3.4.0`** to execute AMD OpenCV extensions
* ROCm `3.0` and above has known to slow down OpenCL kernels.

### Tested configurations
* Windows 10
* Linux: Ubuntu - `16.04`/`18.04` & CentOS - `7.5`/`7.6`
* ROCm: rocm-opencl-dev - `2.0.20191`
* OpenCV - [3.4.0](https://github.com/opencv/opencv/releases/tag/3.4.0)
* Dependencies for all the above packages
