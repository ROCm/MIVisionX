[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![doc](https://img.shields.io/badge/doc-readthedocs-blueviolet)](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/)

<p align="center"><img width="70%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/MIVisionX.png" /></p>

> [!NOTE]
> The published documentation is available at [MIVisionX](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

AMD MIVisionX is a comprehensive computer vision and machine intelligence toolkit. It delivers a highly optimized, conformant open-source implementation of the <a href="https://www.khronos.org/openvx/" target="_blank">Khronos OpenVX&trade;</a> and OpenVX&trade; Extensions, along with a neural net model compiler & optimizer supporting <a href="https://onnx.ai/" target="_blank">ONNX</a> and <a href="https://www.khronos.org/nnef" target="_blank">Khronos NNEF&trade;</a> exchange formats. MIVisionX enables rapid prototyping and deployment of optimized computer vision and machine learning inference workloads on a wide range of hardware, including x86 CPUs, APUs, discrete GPUs, and heterogeneous servers.

#### Latest release

[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/ROCm/MIVisionX?style=for-the-badge)](https://github.com/ROCm/MIVisionX/releases)

## AMD OpenVX&trade;

<p align="center"><img width="30%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/OpenVX_logo.png" /></p>

[AMD OpenVX&trade;](amd_openvx/README.md) is a highly optimized conformant open source implementation of the <a href="https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html" target="_blank">Khronos OpenVX&trade; 1.3</a> computer vision specification. It allows for rapid prototyping as well as fast execution on a wide range of computer hardware, including small embedded x86 CPUs and large workstation discrete GPUs.

<a href="https://www.khronos.org/registry/OpenVX/specs/1.0.1/html/index.html" target="_blank">Khronos OpenVX&trade; 1.0.1</a> conformant implementation is available in [MIVisionX Lite](https://github.com/ROCm/MIVisionX/tree/openvx-1.0.1)

## AMD OpenVX&trade; Extensions

The OpenVX framework provides a mechanism for vendors to add new vision functionality. This project includes the following OpenVX [modules](amd_openvx_extensions/README.md) that extend [amd_openvx](amd_openvx/README.md), the AMD OpenVX&trade; Core Engine.

<p align="center"><img width="70%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/MIVisionX-OpenVX-Extensions.png" /></p>

* [amd_custom](amd_openvx_extensions/amd_custom/README.md): User-defined custom nodes for OpenVX graphs
* [amd_loomsl](amd_openvx_extensions/amd_loomsl/README.md): Loom stitching library for live 360-degree video applications
* [amd_media](amd_openvx_extensions/amd_media/README.md): Video and image encode/decode extension
* [amd_migraphx](amd_openvx_extensions/amd_migraphx/README.md): <a href="https://github.com/ROCmSoftwarePlatform/AMDMIGraphX#amd-migraphx" target="_blank">AMD MIGraphX</a> integration for end-to-end inference
* [amd_nn](amd_openvx_extensions/amd_nn/README.md): Neural network extension module
* [amd_opencv](amd_openvx_extensions/amd_opencv/README.md): OpenCV interop providing OpenCV functions as OpenVX kernels
* [amd_rpp](amd_openvx_extensions/amd_rpp/README.md): Interface to [ROCm Performance Primitives](https://github.com/ROCm/rpp) (RPP) for image augmentation via [rocAL](https://github.com/ROCm/rocAL)
* [amd_winml](amd_openvx_extensions/amd_winml/README.md): WinML extension to import ONNX models with pre & post processing for inference on Windows

## Applications

MIVisionX includes several [applications](apps/README.md#applications) built on top of OpenVX modules, serving as prototypes and samples for developers.

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/MIVisionX-applications.png" /></p>

## Neural network model compiler and optimizer

<p align="center"><img width="80%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/modelCompilerWorkflow.png" /></p>

[Neural net model compiler and optimizer](model_compiler/README.md#neural-net-model-compiler--optimizer) converts pre-trained neural net models to MIVisionX runtime code for optimized inference.

## Toolkit

[MIVisionX Toolkit](toolkit/README.md) is a comprehensive set of helpful tools for neural net creation, development, training, and deployment. The Toolkit provides tools to design, develop, quantize, prune, retrain, and infer your neural network work in any framework, and deploy on any AMD or 3rd party hardware.

## Utilities

* [loom_shell](utilities/loom_shell/README.md#radeon-loomsh): Interpreter for prototyping 360-degree video stitching applications
* [mv_deploy](utilities/mv_deploy/README.md): Model compiler and runtime files for neural net inference deployment
* [RunCL](utilities/runcl/README.md#amd-runcl): Command-line utility to build, execute, and debug OpenCL programs
* [RunVX](utilities/runvx/README.md#amd-runvx): Command-line utility to execute OpenVX graphs described in GDF text files

## Prerequisites

### Hardware

* **CPU**: [AMD64](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* **GPU**: [AMD Radeon&trade; Graphics](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) / [AMD Instinct&trade; Accelerators](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) [optional]
* **APU**: [AMD Radeon&trade; `Mobile`/`Embedded`](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) [optional]

> [!IMPORTANT]
> Some modules can be built for CPU only. To take advantage of advanced features and modules, we recommend using AMD GPUs or AMD APUs.

### Operating Systems

#### Linux
* Ubuntu - `22.04` / `24.04`
* RedHat - `8` / `9`
* SLES - `15-SP7`

#### Windows
* Windows `10` / `11`

#### macOS
* macOS - Ventura `13` / Sonoma `14` / Sequoia `15`

### Compiler
* AMD Clang++ Version `18.0.0` or later - installed with ROCm
> [!NOTE]
> AMD Clang++ is the preferred cxx compiler, users can change this with the `CMAKE_CXX_COMPILER` variable

### Libraries

| Package | Minimum Version |
|---------|----------------|
| CMake | `3.10` |
| HIP | - |
| OpenMP | - |
| Half | `1.12.0` |
| MIOpen | - |
| MIGraphX | - |
| RPP | `3.1.0` |
| OpenCV | `3.X` / `4.X` |
| FFmpeg | `4.4.2` |
| pkg-config | - |

```shell
sudo apt install cmake hip-dev openmp-extras-dev half miopen-hip-dev migraphx-dev rpp-dev libopencv-dev pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
```

> [!IMPORTANT]
> * Required compiler support: `C++17`, `OpenMP`, `Threads`
> * On `Ubuntu 22.04` - Additional package required: `sudo apt install libstdc++-12-dev`

> [!NOTE]
> All package installs are shown with the `apt` package manager. Use the appropriate package manager for your operating system.

## Installation instructions

### Linux

Verify you have [ROCm-supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html), then install ROCm `7.0.0` or later with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) using `--usecase=rocm`.

> [!IMPORTANT]
> Use **either** [package install](#package-install) **or** [source install](#source-install) as described below.

#### Package install

Install MIVisionX runtime, development, and test packages.
* Runtime package - `mivisionx` only provides the dynamic libraries and executables
* Development package - `mivisionx-dev`/`mivisionx-devel` provides the libraries, executables, header files, and samples
* Test package - `mivisionx-test` provides ctest to verify installation

##### Ubuntu
  ```shell
  sudo apt-get install mivisionx mivisionx-dev mivisionx-test
  ```
##### RedHat
  ```shell
  sudo yum install mivisionx mivisionx-devel mivisionx-test
  ```
##### SLES
  ```shell
  sudo zypper install mivisionx mivisionx-devel mivisionx-test
  ```

> [!IMPORTANT]
>  * Package install supports `HIP` backend. For OpenCL backend build from source.
>  * `RedHat`/`SLES` requires `OpenCV` & `FFMPEG` development packages manually installed

#### Source install

Use the `MIVisionX-setup.py` script to install all required dependencies, then build from source.

> [!NOTE]
> Install ROCm before running the setup script. This script only needs to be executed once (rerun after ROCm upgrades).

```shell
git clone https://github.com/ROCm/MIVisionX.git
cd MIVisionX
python MIVisionX-setup.py
mkdir build-hip && cd build-hip
cmake ../
make -j8
sudo make install
make test
```

Run `python MIVisionX-setup.py --help` for all setup options including `--backend`, `--opencv`, and `--rocm_path`.

* [Test option instructions](https://github.com/ROCm/MIVisionX/wiki/CTest)
* Instructions for building MIVisionX with [**OPENCL** GPU backend](https://github.com/ROCm/MIVisionX/wiki/OpenCL-Backend)

### Windows

* Windows SDK
* Visual Studio 2019 or later
* Install the latest AMD [drivers](https://www.amd.com/en/support)
* Install [OpenCL SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0)
* Install [OpenCV 3.4.0](https://github.com/opencv/opencv/releases/tag/3.4.0)
  + Set `OpenCV_DIR` environment variable to `OpenCV/build` folder
  + Add `%OpenCV_DIR%\x64\vc14\bin` or `%OpenCV_DIR%\x64\vc15\bin` to your `PATH`

#### Using Visual Studio
* Use `MIVisionX.sln` to build for x64 platform

> [!IMPORTANT]
> Some modules in MIVisionX are only supported on Linux

### macOS

macOS [build instructions](https://github.com/ROCm/MIVisionX/wiki/macOS#macos-build-instructions)

> [!IMPORTANT]
> macOS only supports MIVisionX CPU backend on `x86` processors

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

  <p align="center"><img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/samples/images/canny_image.PNG" /></p>

  ```shell
  export PATH=$PATH:/opt/rocm/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
  runvx /opt/rocm/share/mivisionx/samples/gdf/canny.gdf
  ```

> [!NOTE]
> * More samples are available [here](samples/README.md#samples)
> * For `macOS` use `export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/rocm/lib`

#### Verify with mivisionx-test package

Test package provides a ctest module to verify MIVisionX installation.

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

MIVisionX provides [Docker images](docker/README.md#mivisionx-docker) for Ubuntu `22.04` to quickly prototype and build applications.

* [Ubuntu 22.04](https://hub.docker.com/repository/docker/mivisionx/ubuntu-22.04)

## Documentation

* [Published documentation](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/)
* Build locally: `cd docs && pip3 install -r sphinx/requirements.txt && python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html`
* Doxygen: `doxygen .Doxyfile`

## Technical support

Please email `mivisionx.support@amd.com` for questions and feedback, or submit feature requests and bug reports on the [GitHub issues](https://github.com/ROCm/MIVisionX/issues) page.

## Release notes

[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/ROCm/MIVisionX?style=for-the-badge)](https://github.com/ROCm/MIVisionX/releases)

Review all notable [changes](CHANGELOG.md#changelog) with the latest release.

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  + Ubuntu - `22.04` / `24.04`
  + RedHat - `8` / `9`
  + SLES - `15-SP7`
* ROCm: `7.2.1`
* RPP - `3.1.0`
* miopen-hip - `3.4.0`
* migraphx - `2.13.0`
* OpenCV - `4.5.4`/`4.6`
* FFMPEG - `4.4.2`
* MIVisionX Setup Script - `V4.0.0`

### Known issues
* Package install on `RedHat`/`SLES` requires manual `OpenCV` and `FFMPEG` development packages installed
