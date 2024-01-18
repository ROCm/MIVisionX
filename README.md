[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![doc](https://img.shields.io/badge/doc-readthedocs-blueviolet)](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/doxygen/html/index.html)

<p align="center"><img width="70%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/MIVisionX.png" /></p>

MIVisionX toolkit is a set of comprehensive computer vision and machine intelligence libraries, utilities, and applications bundled into a single toolkit. AMD MIVisionX delivers highly optimized conformant open-source implementation of the <a href="https://www.khronos.org/openvx/" target="_blank">Khronos OpenVX&trade;</a> and OpenVX&trade; Extensions along with Convolution Neural Net Model Compiler & Optimizer supporting <a href="https://onnx.ai/" target="_blank">ONNX</a>, and <a href="https://www.khronos.org/nnef" target="_blank">Khronos NNEF&trade;</a> exchange formats. The toolkit allows for rapid prototyping and deployment of optimized computer vision and machine learning inference workloads on a wide range of computer hardware, including small embedded x86 CPUs, APUs, discrete GPUs, and heterogeneous servers.

#### Latest release

[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/ROCm/MIVisionX?style=for-the-badge)](https://github.com/ROCm/MIVisionX/releases)

## AMD OpenVX&trade;

<p align="center"><img width="30%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/OpenVX_logo.png" /></p>

[AMD OpenVX&trade;](amd_openvx/README.md) is a highly optimized conformant open source implementation of the <a href="https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html" target="_blank">Khronos OpenVX&trade; 1.3</a> computer vision specification. It allows for rapid prototyping as well as fast execution on a wide range of computer hardware, including small embedded x86 CPUs and large workstation discrete GPUs.

<a href="https://www.khronos.org/registry/OpenVX/specs/1.0.1/html/index.html" target="_blank">Khronos OpenVX&trade; 1.0.1</a> conformant implementation is available in [MIVisionX Lite](https://github.com/ROCm/MIVisionX/tree/openvx-1.0.1)

## AMD OpenVX&trade; Extensions

The OpenVX framework provides a mechanism to add new vision functionality to OpenVX by vendors. This project has below mentioned OpenVX [modules](amd_openvx_extensions/README.md) and utilities to extend [amd_openvx](amd_openvx/README.md), which contains the AMD OpenVX&trade; Core Engine.

<p align="center"><img width="70%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/MIVisionX-OpenVX-Extensions.png" /></p>

* [amd_loomsl](amd_openvx_extensions/amd_loomsl/README.md): AMD Radeon Loom stitching library for live 360 degree video applications
* [amd_media](amd_openvx_extensions/amd_media/README.md): AMD media extension module is for encode and decode applications
* [amd_migraphx](amd_openvx_extensions/amd_migraphx/README.md): amd_migraphx extension integrates the <a href="https://github.com/ROCmSoftwarePlatform/AMDMIGraphX#amd-migraphx" target="_blank"> AMD's MIGraphx </a> into an OpenVX graph. This extension allows developers to combine the vision funcions in OpenVX with the MIGraphX and build an end-to-end application for inference.
* [amd_nn](amd_openvx_extensions/amd_nn/README.md): OpenVX neural network module
* [amd_opencv](amd_openvx_extensions/amd_opencv/README.md): OpenVX module that implements a mechanism to access OpenCV functionality as OpenVX kernels
* [amd_rpp](amd_openvx_extensions/amd_rpp/README.md): OpenVX extension providing an interface to some of the [RPP](https://github.com/ROCm/rpp)'s (ROCm Performance Primitives) functions. This extension is used to enable [rocAL](rocAL/README.md) to perform image augmentation.
* [amd_winml](amd_openvx_extensions/amd_winml/README.md): WinML extension will allow developers to import a pre-trained ONNX model into an OpenVX graph and add hundreds of different pre & post processing `vision` / `generic` / `user-defined` functions, available in OpenVX and OpenCV interop, to the input and output of the neural net model. This will allow developers to build an end to end application for inference.

## Applications

MIVisionX has several [applications](apps/README.md#applications) built on top of OpenVX modules, it uses AMD optimized libraries to build applications that can be used to prototype or use as a model to develop products.

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/MIVisionX-applications.png" /></p>

## Neural network model compiler and optimizer

<p align="center"><img width="80%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/modelCompilerWorkflow.png" /></p>

[Neural net model compiler and optimizer](model_compiler/README.md#neural-net-model-compiler--optimizer) converts pre-trained neural net models to MIVisionX runtime code for optimized inference.

## rocAL

The ROCm Augmentation Library - [rocAL](rocAL/README.md) is designed to efficiently decode and process images and videos from a variety of storage formats and modify them through a processing graph programmable by the user.

rocAL is now available as an independent module at [https://github.com/ROCm/rocAL](https://github.com/ROCm/rocAL). rocAL will be deprecated in MIVisionX with `ROCm 6.2.0`.

## Toolkit

[MIVisionX Toolkit](toolkit/README.md), is a comprehensive set of helpful tools for neural net creation, development, training, and deployment. The Toolkit provides you with helpful tools to design, develop, quantize, prune, retrain, and infer your neural network work in any framework. The Toolkit is designed to help you deploy your work to any AMD or 3rd party hardware, from embedded to servers.

MIVisionX provides you with tools for accomplishing your tasks throughout the whole neural net life-cycle, from creating a model to deploying them for your target platforms.

## Utilities

* [loom_shell](utilities/loom_shell/README.md#radeon-loomsh): an interpreter to prototype 360 degree video stitching applications using a script
* [mv_deploy](utilities/mv_deploy/README.md): consists of a model-compiler and necessary header/.cpp files which are required to run inference for a specific NeuralNet model
* [RunCL](utilities/runcl/README.md#amd-runcl): command-line utility to build, execute, and debug OpenCL programs
* [RunVX](utilities/runvx/README.md#amd-runvx): command-line utility to execute OpenVX graph described in GDF text file

## Prerequisites

### Hardware

* **CPU**: [AMD64](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* **GPU**: [AMD Radeon&trade; Graphics](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) [optional]
* **APU**: [AMD Radeon&trade; `Mobile`/`Embedded`](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) [optional]

  **Note:** Some modules in MIVisionX can be built for `CPU ONLY`. To take advantage of `Advanced Features And Modules` we recommend using `AMD GPUs` or `AMD APUs`.

### Operating System

#### Linux
* **Ubuntu** - `20.04` / `22.04`
* **CentOS** - `7`
* **RedHat** - `8` / `9`
* **SLES** - `15-SP4`

#### Windows
* Windows `10` / `11`

#### macOS
* macOS - Ventura `13` / Sonoma `14`

## Build and install instructions

### Linux

* [ROCm supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* Install ROCm with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) with `--usecase=graphics,rocm --no-32`

#### Package install

Install MIVisionX runtime, development, and test packages. 
* Runtime package - `mivisionx` only provides the dynamic libraries and executables
* Development package - `mivisionx-dev`/`mivisionx-devel` provides the libraries, executables, header files, and samples
* Test package - `mivisionx-test` provides ctest to verify installation

##### On `Ubuntu`
  ```shell
  sudo apt-get install mivisionx mivisionx-dev mivisionx-test
  ```
##### On `CentOS`/`RedHat`
  ```shell
  sudo yum install mivisionx mivisionx-devel mivisionx-test
  ```
##### On `SLES`
  ```shell
  sudo zypper install mivisionx mivisionx-devel mivisionx-test
  ```

  **Note:**
  * Package install requires `OpenCV V4.6` and `Protobuf V3.12.4`  manual install
  * `CentOS`/`RedHat`/`SLES` requires `FFMPEG Dev` package manual install

#### Source build and install

##### Prerequisites setup script for Linux

For the convenience of the developer, we provide the setup script `MIVisionX-setup.py` which will install all the dependencies required by this project.

**NOTE:** This script only needs to be executed once.

##### Prerequisites for running the script

* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`
  + RedHat - `8` / `9`
  + SLES - `15-SP4`
* [ROCm supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* Install ROCm with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) with `--usecase=graphics,rocm --no-32`

  **usage:**
  ```shell
  python MIVisionX-setup.py --directory [setup directory - optional (default:~/)]
                            --opencv    [OpenCV Version - optional (default:4.6.0)]
                            --protobuf  [ProtoBuf Version - optional (default:3.12.4)]
                            --pybind11  [PyBind11 Version - optional (default:v2.10.4)]
                            --ffmpeg    [FFMPEG V4.4.2 Installation - optional (default:ON) [options:ON/OFF]]
                            --rocal     [MIVisionX rocAL Dependency Install - optional (default:ON) [options:ON/OFF]]
                            --neural_net[MIVisionX Neural Net Dependency Install - optional (default:ON) [options:ON/OFF]]
                            --inference [MIVisionX Neural Net Inference Dependency Install - optional (default:ON) [options:ON/OFF]]
                            --developer [Setup Developer Options - optional (default:OFF) [options:ON/OFF]]
                            --reinstall [Remove previous setup and reinstall (default:OFF)[options:ON/OFF]]
                            --backend   [MIVisionX Dependency Backend - optional (default:HIP) [options:HIP/OCL/CPU]]
                            --rocm_path [ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required]
  ```
  **Note:**
    * **ROCm upgrade** requires the setup script **rerun**

##### Using MIVisionX-setup.py 

* Clone MIVisionX git repository

  ```shell
  git clone https://github.com/ROCm/MIVisionX.git
  ```

  **Note:** MIVisionX has support for two GPU backends: **OPENCL** and **HIP**:

* Instructions for building MIVisionX with the **HIP** GPU backend (default GPU backend):

    + run the setup script to install all the dependencies required by the **HIP** GPU backend:
  
  ```shell
  cd MIVisionX
  python MIVisionX-setup.py
  ```

    + run the below commands to build MIVisionX with the **HIP** GPU backend:

  ```shell
  mkdir build-hip
  cd build-hip
  cmake ../
  make -j8
  sudo cmake --build . --target PyPackageInstall
  sudo make install
  ```

    + run tests - [test option instructions](https://github.com/ROCm/MIVisionX/wiki/CTest)

  ```shell
  make test
  ```

    **Note:**
    + `PyPackageInstall` used for rocal_pybind installation
    + `sudo` required for pybind installation


* Instructions for building MIVisionX with [**OPENCL** GPU backend](https://github.com/ROCm/MIVisionX/wiki/OpenCL-Backend)

### Windows

* Windows SDK
* Visual Studio 2019 or later
* Install the latest AMD [drivers](https://www.amd.com/en/support)
* Install [OpenCL SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0)
* Install [OpenCV 4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
  + Set `OpenCV_DIR` environment variable to `OpenCV/build` folder
  + Add `%OpenCV_DIR%\x64\vc14\bin` or `%OpenCV_DIR%\x64\vc15\bin` to your `PATH`

#### Using Visual Studio
* Use `MIVisionX.sln` to build for x64 platform

**NOTE:** Some modules in MIVisionX are only supported on Linux

### macOS

macOS [build instructions](https://github.com/ROCm/MIVisionX/wiki/macOS#macos-build-instructions)

**NOTE:** MIVisionX CPU backend is supported in macOS

## Verify installation

### Linux / macOS

* The installer will copy
  + Executables into `/opt/rocm/bin`
  + Libraries into `/opt/rocm/lib`
  + Header files into `/opt/rocm/include/mivisionx`
  + Apps, & Samples folder into `/opt/rocm/share/mivisionx`
  + Documents folder into `/opt/rocm/share/doc/mivisionx`
  + Model Compiler, and Toolkit folder into `/opt/rocm/libexec/mivisionx`

#### Verify with sample application
  **Canny Edge Detection**

  <p align="center"><img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/canny_image.PNG" /></p>

  ```shell
  export PATH=$PATH:/opt/rocm/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
  runvx /opt/rocm/share/mivisionx/samples/gdf/canny.gdf
  ```
  **Note:** More samples are available [here](samples#samples)

  **Note:** For `macOS` use `export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/rocm/lib`

#### Verify with mivisionx-test package

Test package will install ctest module to test MIVisionX. Follow below steps to test packge install

```shell
mkdir mivisionx-test && cd mivisionx-test
cmake /opt/rocm/share/mivisionx/test/
ctest -VV
```
### Windows

* `MIVisionX.sln` builds the libraries & executables in the folder `MIVisionX/x64`
* Use `RunVX` to test the build

  ```shell
  ./runvx.exe ADD_PATH_TO/MIVisionX/samples/gdf/skintonedetect.gdf
  ```

## Docker

MIVisionX provides developers with docker images for Ubuntu `20.04` / `22.04`. Using docker images developers can quickly prototype and build applications without having to be locked into a single system setup or lose valuable time figuring out the dependencies of the underlying software.

Docker files to build MIVisionX containers and suggested workflow are [available](docker#mivisionx-docker)

### MIVisionX docker
* [Ubuntu 20.04](https://cloud.docker.com/repository/docker/mivisionx/ubuntu-20.04)
* [Ubuntu 22.04](https://cloud.docker.com/repository/docker/mivisionx/ubuntu-22.04)

## Documentation

Run the steps below to build documentation locally.
* sphinx documentation
```Bash
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```
* Doxygen 
```Bash
doxygen .Doxyfile
```

## Technical support

Please email `mivisionx.support@amd.com` for questions, and feedback on MIVisionX.

Please submit your feature requests, and bug reports on the [GitHub issues](https://github.com/ROCm/MIVisionX/issues) page.

## Release notes

### Latest release version

[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/ROCm/MIVisionX?style=for-the-badge)](https://github.com/ROCm/MIVisionX/releases)

### Changelog

Review all notable [changes](CHANGELOG.md#changelog) with the latest release

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7` / `8`
  + RHEL - `8` / `9`
  + SLES - `15-SP4`
* ROCm: rocm-core - `5.7.0.50700-6`
* miopen-hip - `2.20.0.50700-63`
* migraphx - `2.7.0.50700-63`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* RPP - [1.5.0]
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all the above packages
* MIVisionX Setup Script - `V2.6.1`

### Known issues

* OpenCV 4.X support for some apps missing
* MIVisionX Package install requires manual prerequisites installation 

## MIVisionX dependency map

### HIP Backend

**Docker Image:** `sudo docker build -f docker/ubuntu20/{DOCKER_LEVEL_FILE_NAME}.dockerfile -t {mivisionx-level-NUMBER} .`

- ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `new component added to the level`
- ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `existing component from the previous level`

| Build Level | MIVisionX Dependencies                             | Modules                                                                   | Libraries and Executables                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Docker Tag                                                                                                                                                                                                     |
| ----------- | -------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Level_1`   | cmake <br> gcc <br> g++                            | amd_openvx  <br> utilities                                                | ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libopenvx.so` - OpenVX&trade; Lib - CPU <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvxu.so` - OpenVX&trade; immediate node Lib - CPU <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `runvx` - OpenVX&trade; Graph Executor - CPU with Display OFF                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | [![Docker Image Version (tag latest semver)](https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-1?style=flat-square)](https://hub.docker.com/repository/docker/kiritigowda/ubuntu-18.04) |
| `Level_2`   | ROCm HIP <br> +Level 1                             | amd_openvx <br> amd_openvx_extensions <br> utilities                      | ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libopenvx.so`  - OpenVX&trade; Lib - CPU/GPU <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvxu.so` - OpenVX&trade; immediate node Lib - CPU/GPU <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `runvx` - OpenVX&trade; Graph Executor - Display OFF                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | [![Docker Image Version (tag latest semver)](https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-2?style=flat-square)](https://hub.docker.com/repository/docker/kiritigowda/ubuntu-18.04) |
| `Level_3`   | OpenCV <br> FFMPEG <br> +Level 2                   | amd_openvx <br> amd_openvx_extensions <br> utilities                      | ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libopenvx.so`  - OpenVX&trade; Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvxu.so` - OpenVX&trade; immediate node Lib <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvx_amd_media.so` - OpenVX&trade; Media Extension <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvx_opencv.so` - OpenVX&trade; OpenCV InterOp Extension <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `mv_compile` - Neural Net Model Compile <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `runvx` - OpenVX&trade; Graph Executor - Display ON                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | [![Docker Image Version (tag latest semver)](https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-3?style=flat-square)](https://hub.docker.com/repository/docker/kiritigowda/ubuntu-18.04) |
| `Level_4`   | MIOpen <br> MIGraphX <br> ProtoBuf <br> +Level 3 | amd_openvx <br>  amd_openvx_extensions <br> apps <br> utilities           | ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libopenvx.so`  - OpenVX&trade; Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvxu.so` - OpenVX&trade; immediate node Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_amd_media.so` - OpenVX&trade; Media Extension <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_opencv.so` - OpenVX&trade; OpenCV InterOp Extension <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `mv_compile` - Neural Net Model Compile <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `runvx` - OpenVX&trade; Graph Executor - Display ON <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvx_nn.so` - OpenVX&trade; Neural Net Extension                                                                                                                                                                                                                                                                                                                                                                                                                                           | [![Docker Image Version (tag latest semver)](https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-4?style=flat-square)](https://hub.docker.com/repository/docker/kiritigowda/ubuntu-18.04) |
| `Level_5`   | AMD_RPP <br> rocAL deps <br> +Level 4              | amd_openvx <br> amd_openvx_extensions <br> apps <br> rocAL <br> utilities | ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libopenvx.so`  - OpenVX&trade; Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvxu.so` - OpenVX&trade; immediate node Lib <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_amd_media.so` - OpenVX&trade; Media Extension <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_opencv.so` - OpenVX&trade; OpenCV InterOp Extension <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `mv_compile` - Neural Net Model Compile <br> ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `runvx` - OpenVX&trade; Graph Executor - Display ON <br>  ![#1589F0](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png) `libvx_nn.so` - OpenVX&trade; Neural Net Extension <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `libvx_rpp.so` - OpenVX&trade; RPP Extension <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `librocal.so` - Radeon Augmentation Library <br> ![#c5f015](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png) `rocal_pybind.so` - rocAL Pybind Lib | [![Docker Image Version (tag latest semver)](https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-5?style=flat-square)](https://hub.docker.com/repository/docker/kiritigowda/ubuntu-18.04) |

**NOTE:** OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
