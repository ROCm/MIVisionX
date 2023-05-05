<p align="center"><img width="60%" src="https://raw.githubusercontent.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/master/docs/data/MIVisionX.png" /></p>

# Changelog

## Online Documentation

[MIVisionX Documentation](https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/)

## MIVisionX 2.4.0

### Added

* OpenVX FP16 Support
* rocAL - CPU, HIP, & OCL backends
* AMD RPP - CPU, HIP, & OCL backends
* MIVisionX Setup Support for RHEL
* Extended OS Support
* Docker Support for Ubuntu `22.04`
* Tests

### Optimizations

* CMakeList Cleanup
* MIGraphX Extension Updates
* rocAL - Documentation
* CMakeList Updates & Cleanup

### Changed

* rocAL - Changing Python Lib Path
* Docker Support - Ubuntu 18 Support Dropped
* RPP - Link to Version 1.0.0
* rocAL - support updates
* Setup Updates

### Fixed

* rocAL bug fix and updates
* AMD RPP - bug fixes
* CMakeLists - Issues
* RPATH - Link Issues

### Tested Configurations

* Windows `10` / `11`
* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7` / `8`
  + RHEL - `8` / `9`
  + SLES - `15-SP3`
* ROCm: rocm-core - `5.4.3.50403-121`
* miopen-hip - `2.19.0.50403-121`
* miopen-opencl - `2.18.0.50300-63`
* migraphx - `2.4.0.50403-121`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* RPP - [1.0.0](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/releases/tag/1.0.0)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all the above packages
* MIVisionX Setup Script - `V2.4.2`

### Known issues

* OpenCV 4.X support for some apps missing

## MIVisionX 2.3.0

### Added

* Zen DNN Samples
* OpenCV Extension tests
* rocAL SPACK Support
* AMD Custom Extension

### Optimizations

* MIGraphX Updates
* Model Compiler Scripts

### Changed

* CMakeList - Find HIP updates

### Fixed

* rocAL issues
* MIGraphX issues

### Tested Configurations

* Windows `10` / `11`
* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7` / `8`
  + SLES - `15-SP2`
* ROCm: rocm-core - `5.3.0.50300-36`
* miopen-hip - `2.18.0.50300-36`
* miopen-opencl - `2.18.0.50300-36`
* migraphx - `2.3.0.50300-36`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5)
* RPP - [0.97](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/releases/tag/0.97)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all the above packages
* MIVisionX Setup Script - `V2.3.7`

### Known issues

* OpenCV 4.X support for some apps missing

## MIVisionX 2.2.0

### Added

* Docker Support for ROCm `5.1.3`

### Optimizations

* MIGraphX - Implementation & Samples

### Changed

* DockerFiles - Updates to install ROCm 5.1.1 Plus

### Fixed

* Minor bugs in setup & test scripts

### Tested Configurations

* Windows `10` / `11`
* Linux distribution
  + Ubuntu - `18.04` / `20.04`
  + CentOS - `7` / `8`
  + SLES - `15-SP2`
* ROCm: rocm-core - `5.1.3.50103-66`
* miopen-hip - `2.16.0.50101-48`
* miopen-opencl - `2.16.0.50101-48`
* migraphx - `2.1.0.50101-48`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5)
* RPP - [0.93](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/releases/tag/0.93)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all the above packages
* MIVisionX Setup Script - `V2.3.2`

### Known issues

* OpenCV 4.X support for some apps missing

## MIVisionX 2.1.0

### Added

* New Tests - AMD_MEDIA

### Optimizations

* Readme Updates
* HIP Buffer Transfer - Eliminate cupy usage

### Changed

* **Backend** - Default Backend set to `HIP`
* DockerFiles - Updates to install ROCm 5.0 Plus
* RPP - Upgraded to V0.93

### Fixed

* Minor bugs and warnings
* AMD_MEDIA - Bug Fixes

### Tested Configurations

* Windows `10` / `11`
* Linux distribution
  + Ubuntu - `18.04` / `20.04`
  + CentOS - `7` / `8`
  + SLES - `15-SP2`
* ROCm: rocm-core - `5.0.0.50000-49`
* rocm-cmake - [rocm-5.1.1](https://github.com/RadeonOpenCompute/rocm-cmake/releases/tag/rocm-5.1.1)
* MIOpenGEMM - [1.1.5](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5)
* MIOpen - [2.14.0](https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/2.14.0)
* Protobuf - [V3.12.0](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.0)
* OpenCV - [4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5)
* RPP - [0.93](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/releases/tag/0.93)
* FFMPEG - [n4.0.4](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.0.4)
* Dependencies for all the above packages
* MIVisionX Setup Script - `V2.1.1`

### Known issues

* `CMakeList.txt Warnings` - With ROCm CMake Version MIVisionX will have cmake warnings

## MIVisionX 2.0.1

### Added

* Support for cmake 3.22.X
* Support for OpenCV 4.X.X
* Support for mv_compile with the HIP GPU backend
* Support for tensor_compare node (less/greater/less_than/greater_than/equal onnx operators)

### Optimizations

* Code Cleanup
* Readme Updates

### Changed

* License Updates

### Fixed

* Minor bugs and warnings
* Inference server application - OpenCL Backend
* vxCreateThreshold Fix - Apps & Sample

### Tested Configurations

* Windows 10
* Linux distribution
  + Ubuntu - `18.04` / `20.04`
  + CentOS - `7` / `8`
  + SLES - `15-SP2`
* ROCm: rocm-dev - `4.5.2.40502-164`
* rocm-cmake - [rocm-4.2.0](https://github.com/RadeonOpenCompute/rocm-cmake/releases/tag/rocm-4.2.0)
* MIOpenGEMM - [1.1.5](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5)
* MIOpen - [2.14.0](https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/2.14.0)
* Protobuf - [V3.12.0](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.0)
* OpenCV - [3.4.0](https://github.com/opencv/opencv/releases/tag/3.4.0)
* RPP - [0.92](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/releases/tag/0.92)
* FFMPEG - [n4.0.4](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.0.4)
* Dependencies for all the above packages
* MIVisionX Setup Script - `V2.0.0`

### Known issues

* Package install requires **OpenCV** `v3.4.X` to execute `AMD OpenCV extensions`

## MIVisionX 2.0.0

### Added

- Added OpenVX 1.3 - Vision Feature Set
- Added Conformance Test Script
- HIP Backend Support for OpenVX and OpenVX Extensions

### Optimizations

- Improved performance of rocAL
- Improved performance of OpenVX OpenCL Backend Functions

### Changed

- Docker Build Files

### Fixed

- MIVisionX-setup.py install on Linux
- Fixed out-of-bounds read for OpenVX OpenCL Kernels
- OpenVX - optical flow segfault fix

### Tested Configurations

* Windows 10
* Linux distribution
  + Ubuntu - `18.04` / `20.04`
  + CentOS - `7` / `8`
  + SLES - `15-SP2`
* ROCm: rocm-dev - `4.5.2.40502-164`
* rocm-cmake - [rocm-4.2.0](https://github.com/RadeonOpenCompute/rocm-cmake/releases/tag/rocm-4.2.0)
* MIOpenGEMM - [1.1.5](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5)
* MIOpen - [2.14.0](https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/2.14.0)
* Protobuf - [V3.12.0](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.0)
* OpenCV - [3.4.0](https://github.com/opencv/opencv/releases/tag/3.4.0)
* RPP - [0.92](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/releases/tag/0.92)
* FFMPEG - [n4.0.4](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.0.4)
* Dependencies for all the above packages
* MIVisionX Setup Script - `V2.0.0`

### Known issues

* Package install requires **OpenCV** `v3.4.X` to execute `AMD OpenCV extensions`
