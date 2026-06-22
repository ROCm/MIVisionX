[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![doc](https://img.shields.io/badge/doc-readthedocs-blueviolet)](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/)

<p align="center"><img width="70%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/MIVisionX.png" /></p>

> [!NOTE]
> The published documentation is available at [MIVisionX](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

AMD MIVisionX is a computer vision toolkit built around a highly optimized, conformant open-source implementation of the <a href="https://www.khronos.org/openvx/" target="_blank">Khronos OpenVX&trade; 1.3</a> specification. Starting with the `4.0.0` release, MIVisionX is streamlined to its core: the AMD OpenVX&trade; engine, the AMD RPP OpenVX&trade; extension, and the `RunVX` graph executor. It enables rapid prototyping and execution of optimized computer vision workloads on a wide range of hardware, including x86 CPUs, APUs, and discrete GPUs, with `CPU`, `HIP`, and `OpenCL` backends.

#### Latest release

[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/ROCm/MIVisionX?style=for-the-badge)](https://github.com/ROCm/MIVisionX/releases)

## AMD OpenVX&trade;

<p align="center"><img width="30%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/OpenVX_logo.png" /></p>

[AMD OpenVX&trade;](amd_openvx/README.md) is a highly optimized conformant open source implementation of the <a href="https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html" target="_blank">Khronos OpenVX&trade; 1.3</a> computer vision specification. It allows for rapid prototyping as well as fast execution on a wide range of computer hardware, including small embedded x86 CPUs and large workstation discrete GPUs.

<a href="https://www.khronos.org/registry/OpenVX/specs/1.0.1/html/index.html" target="_blank">Khronos OpenVX&trade; 1.0.1</a> conformant implementation is available in [MIVisionX Lite](https://github.com/ROCm/MIVisionX/tree/openvx-1.0.1)

## AMD OpenVX&trade; Extensions

The OpenVX framework provides a mechanism for vendors to add new vision functionality. This project includes the following OpenVX [module](amd_openvx_extensions/README.md) that extends [amd_openvx](amd_openvx/README.md), the AMD OpenVX&trade; Core Engine.

* [amd_rpp](amd_openvx_extensions/amd_rpp/README.md): Interface to [ROCm Performance Primitives](https://github.com/ROCm/rpp) (RPP) for image/tensor augmentation via [rocAL](https://github.com/ROCm/rocAL)

> [!NOTE]
> The AMD OpenVX&trade; core engine supports the `CPU`, `HIP`, and `OpenCL` backends. The `amd_rpp` extension supports only the `CPU` and `HIP` backends (RPP has dropped OpenCL support); when the core is built with the `OpenCL` backend, `amd_rpp` is built in CPU-only mode.

## Utilities

* [RunVX](utilities/runvx/README.md#amd-runvx): Command-line utility to execute OpenVX graphs described in GDF text files

## Applications

Sample [applications](apps/README.md#applications) built on AMD OpenVX&trade; and OpenCV (built separately against an installed MIVisionX):

* [bubble_pop](apps/bubble_pop): Creates bubbles and donuts to pop using OpenVX & OpenCV
* [optical_flow](apps/optical_flow/README.md#openvx-samples): Runs Optical Flow on a video/live stream using an OpenVX graph

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

| Package | Minimum Version | Notes |
|---------|----------------|-------|
| CMake | `3.10` | |
| HIP | - | required for the `HIP` backend |
| OpenMP | - | |
| Half | `1.12.0` | float16 support |
| RPP | `3.1.0` | required for the `amd_rpp` extension |
| OpenCV | `3.X` / `4.X` | optional, only used by `RunVX` for image/video display |
| pkg-config | - | |

```shell
sudo apt install cmake hip-dev openmp-extras-dev half rpp-dev pkg-config
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

* Windows `10` / `11`
* Windows SDK + a C++17 toolchain (Visual Studio 2019 or later)
* [CMake](https://cmake.org/download/) `3.10` or later
* Install the latest AMD [drivers](https://www.amd.com/en/support)
* Install [OpenCL SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0)

On Windows, build with CMake (the legacy Visual Studio `.sln`/`.vcxproj` files have been removed). The default Windows backend is `OpenCL`; pass `-DGPU_SUPPORT=OFF` for a CPU-only build.

```shell
git clone https://github.com/ROCm/MIVisionX.git
cd MIVisionX
cmake -B build -DGPU_SUPPORT=OFF
cmake --build build --config Release
```

Open the generated solution in the `build` folder with Visual Studio, or build directly from the command line as shown above.

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
  + Samples folder into `/opt/rocm/share/mivisionx`
  + Documents folder into `/opt/rocm/share/doc/mivisionx`

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

* The CMake build produces the libraries & executables under the `build` folder
* Use `RunVX` to test the build

  ```shell
  .\runvx.exe ADD_PATH_TO\MIVisionX\samples\gdf\skintonedetect.gdf
  ```

## Documentation

* [Published documentation](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/)
* Build locally: `cd docs && pip3 install -r sphinx/requirements.txt && python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html`
* Doxygen API docs: `cd docs/doxygen && doxygen Doxyfile`

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
* OpenCV - `4.5.4`/`4.6` (optional, RunVX display only)
* MIVisionX Setup Script - `V4.0.0`
