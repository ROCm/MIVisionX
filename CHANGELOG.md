<p align="center"><img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/MIVisionX.png" /></p>

# Changelog for MIVisionX

The full documentation for MIVisionX is available at [https://rocm.docs.amd.com/projects/MIVisionX/en/latest/doxygen/html/index.html](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/doxygen/html/index.html)

## Unreleased

### Added
* OpenVX 1.3.2 full Vision Conformance Feature Set — `vxQueryImage` now supports `VX_IMAGE_IS_UNIFORM` and `VX_IMAGE_UNIFORM_VALUE`; `vxMinMaxLoc` count scalars changed to `VX_TYPE_SIZE` per spec
* HIP GPU architecture support for gfx115x (Radeon RX 9000 / gfx1150, gfx1151, gfx1152, gfx1153)
* `MIVISIONX_HIP_CU_COUNT` environment variable to limit the number of compute units used by HIP kernels at runtime

### Fixed
* MinMaxLoc count-scalar type: `ovxKernel_MinMaxLoc` and all `agoKernel_MinMaxLoc_*` sub-kernels now declare and write count outputs as `VX_TYPE_SIZE` instead of `VX_TYPE_UINT32`, matching the OpenVX 1.3.2 spec and Khronos CTS
* Updated `tests/openvx_api_tests/vxu_api/vxu_api.cpp` and all 17 MinMaxLoc GDF files in `tests/amd_openvx_gdfs/cpu/` to use `VX_TYPE_SIZE` / `scalar:SIZE` for count scalars
* Pipelining - a GPU node reading an image a CPU node wrote in another graph read uninitialized device memory. A CPU node now records that the host copy is the newer one whether or not the data has device memory at the time, and a device buffer reserved for a queued reference starts out needing an upload, so the hand-off between two pipelined graphs on different targets uploads the data instead of skipping it
* Pipelining (OpenCL) - a node working on a queued graph parameter kept the buffer argument bound at verification time, so every execution after the first wrote the wrong reference. Queued references are now re-bound per execution, as delay slots already were, for images, arrays, matrices, LUTs, remaps, tensors, and the Canny stack
* Pipelining - `vxScheduleGraph` in `VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL` runs every complete set of enqueued references it finds, so a request could carry out work a later request was made for and that later request then reported an error for queues it found empty. Those extra executions are now credited to the requests that follow
* Tests - `tests/openvx_api_tests/pipelining_api` covers the cross-target hand-off between two pipelined graphs and the one-execution-per-enqueued-set guarantee of `QUEUE_MANUAL`

### Changed
* README and sub-library documentation updated to explicitly state OpenVX **1.3.2** conformance throughout; spec links updated to the 1.3.2 URL

## MIVisionX 4.0.0

### Changed
*  **Breaking change**: MIVisionX has been streamlined to its core components. The toolkit now ships only the AMD OpenVX engine (`amd_openvx`), the AMD RPP OpenVX extension (`amd_openvx_extensions/amd_rpp`), and the `RunVX` graph executor, with continued support for the `CPU`, `HIP`, and `OpenCL` backends.
*  MIVisionX is now built on top of the **ROCm Core SDK** and requires **ROCm 7.13 or later**. The Core SDK provides the HIP and OpenCL runtimes, the `amdclang++` compiler, the `half` library, and `RPP`.
*  Windows builds now use CMake; the legacy Visual Studio `.sln`/`.vcxproj` project files have been removed.

### Fixed
*  **HIP Canny/Harris/NonMaxSupp kernels**: out-of-bounds shared-memory reads when the top-left tile of the image falls on the image boundary. All six kernel families (`CannySobel_3x3/5x5/7x7` L1/L2NORM, `CannySuppThreshold_3x3`, `HarrisSobel_3x3/5x5/7x7`, `NonMaxSupp_3x3`) computed a negative byte offset into the source image buffer for threads assigned to the first few rows or leftmost columns, resulting in undefined behavior. Added `goffset >= 0` guards before each halo load to skip the out-of-bounds access and leave the corresponding shared-memory region zeroed.

### Removed
*  OpenVX extensions: `amd_nn`, `amd_opencv`, `amd_media`, `amd_migraphx`, `amd_loomsl`, `amd_custom`, `amd_winml`
*  Utilities: `runcl`, `loom_shell`, `mv_deploy`, `loom_io_media`
*  The `MIVisionX-setup.py` dependency installer; install the prerequisites (`half`, `rpp-dev`, and optionally OpenCV) directly with your package manager
*  The neural-net `model_compiler`, the `toolkit`, and the ML/Windows `apps` (the OpenVX + OpenCV `bubble_pop` and `optical_flow` sample apps are retained)
*  Extension-specific test suites (`neural_network_tests`, `amd_opencv_tests`, `amd_media_tests`, `amd_migraphx_tests`, `zen_dnn_tests`, `opencv_benchmark`) and the corresponding samples
*  Build options `NEURAL_NET`, `LOOM`, and `MIGRAPHX`, and the legacy `.travis.yml` CI
*  Dependencies on MIOpen, MIGraphX, rocBLAS, FFmpeg, and rocDecode

## MIVisionX 3.6.0

### Added
*  Added the `PythonFunction` extension to VX_RPP
*  Added unified `rppt_*()` function with `RppBackend` parameter for vx_rpp extensions to support runtime backend selection
*  CI - CPU (host backend) performance regression gate comparing PR vs main per-kernel and by geometric mean
*  CI - AMD OpenVX vs OpenCV per-kernel CPU benchmark summary with a +/- 5% parity band
*  Tests - Expanded CPU coverage with additional GDF graphs (Histogram, MinMaxLoc, ScaleImage) and a vision-coverage OpenVX API test

### Removed
*  Removed the batchPD extensions from VX_RPP
*  Removed separate `rppt_*_host()` and `rppt_*_gpu()` function calls for vx_rpp extensions

### Optimizations
*  OpenVX (host backend) - Optimized the S16-to-U8 range threshold kernel (`HafCpu_Threshold_U8_S16_Range`) with an AVX2 (256-bit) code path

### Changed
*  Updated vx_rpp extension for Gaussian Filter
*  **Breaking change**: This change requires updating to RPP version 3.1.0 with unified API support
*  MIGraphX extension turned off by default

## MIVisionX 3.5.0 for ROCm 7.2.0

### Changed
* AMD Clang++ - Location updated `${ROCM_PATH}/lib/llvm/bin`
* RPP required updated to RPP V2.2.1

### Resolved issues
* Memory leaks in OpenVX core, vx_nn, & vx_opencv

### Known issues
* Installation on CentOS/RedHat/SLES requires the manual installation of the `FFMPEG` & `OpenCV` dev packages.

### Upcoming changes
* VX_AMD_MEDIA - `rocDecode` and `rocJPEG` support for hardware decode

## MIVisionX 3.4.0 for ROCm 7.1.0

### Added
* VX_RPP - Update blur
* HIP - HIP_CHECK for hipLaunchKernelGGL for gated launch

### Changed
* AMD Custom V1.1.0 - OpenMP updates
* HALF - Fix half.hpp path updates

### Resolved issues
* AMD Custom - dependency linking errors resolved
* VX_RPP - Fix memory leak
* Packaging - Remove Meta Package dependency for HIP

### Known issues
* Installation on CentOS/RedHat/SLES requires the manual installation of the `FFMPEG` & `OpenCV` dev packages.

### Upcoming changes
* VX_AMD_MEDIA - rocDecode support for hardware decode

## MIVisionX 3.3.0 for ROCm 7.0.0

### Changed

* VX_RPP extension : Version 3.1.0 release
* Add support to enable/disable BatchPD code in VX_RPP extensions by checking the RPP_LEGACY_SUPPORT flag.
* Update the parameters and kernel API of Blur, Fog, Jitter, LensCorrection, Rain, Pixelate, Vignette and ResizeCrop wrt tensor kernels replacing the legacy BatchPD API calls in VX_RPP extensions.

### Known issues

* Installation on CentOS/RedHat/SLES requires the manual installation of the `FFMPEG` & `OpenCV` dev packages.

### Upcoming changes

* Optimized audio augmentations support for VX_RPP

## MIVisionX 3.2.0 for ROCm 6.4.0

### Changed

* OpenCV is now installed with the package installer on Ubuntu
* AMD Clang is now the default CXX and C compiler
* The version of OpenMP included in the ROCm LLVM project is now used instead of libomp-dev/devel

### Known issues

* Installation on CentOS/RedHat/SLES requires manually installing the FFMPEG & OpenCV dev packages.
* Hardware decode requires the ROCm graphics usecase.

### Upcoming changes

* Optimized audio augmentations support for VX_RPP


##  MIVisionX 3.1.0 for ROCm 6.3.0

### Changed

* Setup: rocdecode install disabled
* Package: rocdecode dependency removed

### Optimizations

* Setup: only core dependency packages installed

### Known issues

* MIVisionX package installation on RedHat/SLES requires the manual installation of the `OpenCV` and `FFMPEG` development package.

### Upcoming changes

* Optimized audio augmentations support for VX_RPP

## MIVisionX 3.0.0 for ROCm 6.2.0

### Added

* Support has been added for the following:
  * Advanced GPUs
  * PreEmphasis Filter augmentation in openVX extensions
  * Spectrogram augmentation in openVX extensions
  * Downmix and ToDecibels augmentations in openVX extensions
  * Resample augmentation and Operator overloading nodes in openVX extensions
  * NonSilentRegion and Slice augmentations in openVX extensions
  * Mel-Filter bank and Normalize augmentations in openVX extensions
* CentOS 7 and SLES 15 SP5 support has been added to the setup script.
* The `FindPackage` modules have been updated with `FindMIVision`.
* New tests have been added for all modules.

### Changed

* rocAL is no longer installed with MIVisionX and must be installed separately.

### Resolved issues

* MIVisionX compatibility fix: Resample and pre-emphasis filter.
* Broken image links have been fixed in the documentation.

### Known issues

* Package install requires manually installing OpenCV
* Installation CentOS/RedHat/SLES requires manually installing the `FFMPEG Dev` package
* Hardware decode requires the ROCm `graphics` usecase.

### Upcoming changes

* Optimized audio augmentations support for VX_RPP.

## MIVisionX 2.5.0

### Additions

* CTest: Tests for install verification
* Hardware support updates
* Doxygen support for API documentation

### Optimizations

* CMakeList Cleanup
* Readme

### Changes

* rocAL: PyBind Link to prebuilt library
  * PyBind11
  * RapidJSON
* Setup Updates
* RPP - Use package install
* Dockerfiles: Updates & bugfix
* CuPy - No longer installed with setup.py

### Fixes

* rocAL bug fix and updates

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7` / `8`
  * RHEL - `8` / `9`
  * SLES - `15-SP4`
* ROCm: rocm-core - `5.7.0.50700-6`
* miopen-hip - `2.20.0.50700-63`
* MIGraphX - `2.7.0.50700-63`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* RPP - [1.5.0]
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all preceding packages
* MIVisionX setup script - `V2.6.1`

### Known issues

* OpenCV 4.X support for some applications is missing
* MIVisionX package install requires manual prerequisites installation

## MIVisionX 2.4.0

### Additions

* OpenVX FP16 Support
* rocAL: CPU, HIP, & OCL backends
* AMD RPP: CPU, HIP, amd OCL backends
* MIVisionX Setup Support for RHEL
* Extended OS Support
* Docker Support for Ubuntu `22.04`
* Tests

### Optimizations

* CMakeList cleanup
* MIGraphX extension updates
* rocAL: Documentation
* CMakeList updates and cleanup

### Changes

* rocAL: Changing Python lib path
* Docker support: Ubuntu 18 support has been dropped
* RPP: Link to Version 1.0.0
* rocAL: Support updates
* Setup updates

### Fixes

* rocAL bug fix and updates
* AMD RPP bug fixes
* CMakeLists: Issues
* RPATH: Link issues

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7` / `8`
  * RHEL - `8` / `9`
  * SLES - `15-SP3`
* ROCm: rocm-core - `5.4.3.50403-121`
* miopen-hip - `2.19.0.50403-121`
* miopen-opencl - `2.18.0.50300-63`
* MIGraphX - `2.4.0.50403-121`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* RPP - [1.0.0](https://github.com/ROCm/rpp/releases/tag/1.0.0)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all preceding packages
* MIVisionX setup script - `V2.4.2`

### Known issues

* OpenCV 4.X support for some applications is missing

## MIVisionX 2.3.0

### Additions

* Zen DNN samples
* OpenCV extension tests
* rocAL SPACK support
* AMD custom extension

### Optimizations

* MIGraphX updates
* Model compiler scripts

### Changes

* CMakeList: Find HIP updates

### Fixes

* rocAL issues
* MIGraphX issues

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7` / `8`
  * SLES - `15-SP2`
* ROCm: rocm-core - `5.3.0.50300-36`
* miopen-hip - `2.18.0.50300-36`
* miopen-opencl - `2.18.0.50300-36`
* MIGraphX - `2.3.0.50300-36`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5)
* RPP - [0.97](https://github.com/ROCm/rpp/releases/tag/0.97)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all preceding packages
* MIVisionX setup script - `V2.3.7`

### Known issues

* OpenCV 4.X support for some applications is missing

## MIVisionX 2.2.0

### Additions

* Docker support for ROCm `5.1.3`

### Optimizations

* MIGraphX: Implementation and samples

### Changes

* DockerFiles: Updates to install ROCm 5.1.1 Plus

### Fixes

* Minor bugs in setup and test scripts

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  * Ubuntu - `18.04` / `20.04`
  * CentOS - `7` / `8`
  * SLES - `15-SP2`
* ROCm: rocm-core - `5.1.3.50103-66`
* miopen-hip - `2.16.0.50101-48`
* miopen-opencl - `2.16.0.50101-48`
* MIGraphX - `2.1.0.50101-48`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5)
* RPP - [0.93](https://github.com/ROCm/rpp/releases/tag/0.93)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all preceding packages
* MIVisionX setup script - `V2.3.2`

### Known issues

* OpenCV 4.X support for some applications is missing

## MIVisionX 2.1.0

### Additions

* New Tests: `AMD_MEDIA`

### Optimizations

* Readme updates
* HIP buffer transfer: Eliminate `cupy` usage

### Changes

* **Backend**: Default backend has been set to `HIP`
* DockerFiles: Updates to install ROCm 5.0 Plus
* RPP: Upgraded to V0.93

### Fixes

* Minor bugs and warnings
* `AMD_MEDIA`: Bug Fixes

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  * Ubuntu - `18.04` / `20.04`
  * CentOS - `7` / `8`
  * SLES - `15-SP2`
* ROCm: rocm-core - `5.0.0.50000-49`
* rocm-cmake - [rocm-5.1.1](https://github.com/RadeonOpenCompute/rocm-cmake/releases/tag/rocm-5.1.1)
* MIOpenGEMM - [1.1.5](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5)
* MIOpen - [2.14.0](https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/2.14.0)
* Protobuf - [V3.12.0](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.0)
* OpenCV - [4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5)
* RPP - [0.93](https://github.com/ROCm/rpp/releases/tag/0.93)
* FFMPEG - [n4.0.4](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.0.4)
* Dependencies for all preceding packages
* MIVisionX Setup Script - `V2.1.1`

### Known issues

* `CMakeList.txt` warnings: With ROCm CMake version MIVisionX will have CMake warnings

## MIVisionX 2.0.1

### Additions

* Support for CMake 3.22.X
* Support for OpenCV 4.X.X
* Support for mv_compile with the HIP GPU backend
* Support for tensor_compare node (`less`, `greater`, `less_than`, `greater_than`, and `equal` ONNX
  operators)

### Optimizations

* Code cleanup
* Readme updates

### Changes

* License updates

### Fixes

* Minor bugs and warnings
* Inference server application: OpenCL backend
* `vxCreateThreshold` fix: applications and sample

### Tested configurations

* Windows 10
* Linux distribution
  * Ubuntu - `18.04` / `20.04`
  * CentOS - `7` / `8`
  * SLES - `15-SP2`
* ROCm: rocm-dev - `4.5.2.40502-164`
* rocm-cmake - [rocm-4.2.0](https://github.com/RadeonOpenCompute/rocm-cmake/releases/tag/rocm-4.2.0)
* MIOpenGEMM - [1.1.5](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5)
* MIOpen - [2.14.0](https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/2.14.0)
* Protobuf - [V3.12.0](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.0)
* OpenCV - [3.4.0](https://github.com/opencv/opencv/releases/tag/3.4.0)
* RPP - [0.92](https://github.com/ROCm/rpp/releases/tag/0.92)
* FFMPEG - [n4.0.4](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.0.4)
* Dependencies for all preceding packages
* MIVisionX Setup Script - `V2.0.0`

### Known issues

* Package install requires **OpenCV** `v3.4.X` to execute `AMD OpenCV extensions`

## MIVisionX 2.0.0

### Additions

* OpenVX 1.3: vision feature set
* Conformance test script
* HIP backend support for OpenVX and OpenVX extensions

### Optimizations

* Improved rocAL performance
* Improved the performance of OpenVX OpenCL backend functions

### Changes

* Docker build files

### Fixes

* `MIVisionX-setup.py` install on Linux
* Fixed out-of-bounds read for OpenVX OpenCL kernels
* OpenVX: Optical flow segfault fix

### Tested Configurations

* Windows 10
* Linux distribution
  * Ubuntu - `18.04` / `20.04`
  * CentOS - `7` / `8`
  * SLES - `15-SP2`
* ROCm: rocm-dev - `4.5.2.40502-164`
* rocm-cmake - [rocm-4.2.0](https://github.com/RadeonOpenCompute/rocm-cmake/releases/tag/rocm-4.2.0)
* MIOpenGEMM - [1.1.5](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5)
* MIOpen - [2.14.0](https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/2.14.0)
* Protobuf - [V3.12.0](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.0)
* OpenCV - [3.4.0](https://github.com/opencv/opencv/releases/tag/3.4.0)
* RPP - [0.92](https://github.com/ROCm/rpp/releases/tag/0.92)
* FFMPEG - [n4.0.4](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.0.4)
* Dependencies for all preceding packages
* MIVisionX Setup Script - `V2.0.0`

### Known issues

* Package install requires **OpenCV** `v3.4.X` to run `AMD OpenCV extensions`
