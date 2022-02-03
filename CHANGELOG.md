<p align="center"><img width="60%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/MIVisionX.png" /></p>

# Changelog

## Online Documentation

[MIVisionX Documentation](https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/)

## MIVisionX 2.0.1 (unreleased)

### Added

* Support for cmake 3.22.X
* Support for OpenCV 4.X.X
* Support for mv_compile with the HIP GPU backend
* Support for tensor_compare node (less/greater/less_than/greater_than/equal onnx operators)

### Optimizations

* Code Cleanup

### Changed

* License Updates

### Fixed

* Minor bugs and warnings

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
