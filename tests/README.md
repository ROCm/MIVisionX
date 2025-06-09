# MIVisionX Test Suite

This folder contains the MIVisionX test suite, designed to thoroughly verify the installation and functionality of the MIVisionX™ toolkit. It covers various modules, extensions, and core OpenVX™ implementations, ensuring a robust and reliable computer vision development experience.

## Overview

The MIVisionX test suite is a comprehensive collection of tests that validate the proper installation, functionality, and performance of the MIVisionX libraries and modules. It's an essential tool for developers and users to ensure the integrity of their MIVisionX setup across different backends (CPU, OpenCL™, HIP).

## Test Categories

The tests are organized into logical categories to facilitate easier understanding and execution.

### Core OpenVX Tests

These tests focus on the fundamental OpenVX specification and AMD's implementation.

* **[OpenVX API Tests](openvx_api_tests)**: Verifies the functionality of the OpenVX C/C++ API.
* **[OpenVX Conformance Tests](openvx_conformance_tests)**: Runs the official OpenVX 1.3 Conformance tests for the Vision Feature Set, targeting both CPU and GPU (OpenCL & HIP Backend) implementations.
* **[OpenVX Node Tests](openvx_node_tests)**: Exercises various AMD OpenVX functionalities across HOST, OpenCL, and HIP backends using `RunVX`.
* **[Vision Tests](vision_tests)**: Conducts tests on OpenVX vision functions for both verification and performance assessment.
* **AMD OpenVX Tests**: Validates AMD's specific implementation of OpenVX and its vendor extensions.

### AMD Extension Tests

This section covers tests for AMD-specific OpenVX extensions, enhancing MIVisionX capabilities.

* **AMD Media Tests (`vx_amd_media`)**: Tests the OpenVX AMD media extension module, which includes:
    * `com.amd.amd_media.decode` node for video/JPEG decoding.
    * `com.amd.amd_media.encode` node for video encoding.
* **AMD MIGraphX Tests (`vx_amd_migraphx`)**: Verifies the `com.amd.amd_migraphx_node`, which allows importing the [AMD MIGraphX library](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX#amd-migraphx) into an OpenVX graph for efficient inference.
* **AMD OpenCV Tests (`vx_opencv`)**: Assesses the OpenVX module that provides a mechanism to access OpenCV functionality as OpenVX kernels, with tests implemented using GDFs.
* **VX_RPP Tests (`vx_rpp`)**: Tests the AMD VX RPP extension, an OpenVX module that provides an interface to access RPP (ROCm Performance Primitives) functionality as OpenVX kernels. These tests utilize GDFs.

### Neural Network Tests

These tests focus on MIVisionX's capabilities for neural network inference.

* **[Neural Network Tests](neural_network_tests)**: Verifies the `Caffe`, `ONNX`, and `NNEF` model flow with OpenVX for accurate verification and performance analysis.
* **[Zen DNN Unit Tests](zen_dnn_tests)**: Provides unit, verification, and performance tests for Zen DNN integration.

### Utility Tests

These tests provide general system checks and quick verification.

* **[Library Tests](library_tests)**: Verifies the installation and checks if all MIVisionX libraries are properly built and installed.
* **[Smoke Tests](smoke_tests)**: A quick MIVisionX test suite for rapid validation of core functionality.

## Getting Started

To run these tests, you typically need a complete MIVisionX installation. Refer to the main MIVisionX documentation for detailed build and installation instructions.

## Contributing

We welcome contributions to the MIVisionX test suite! If you find a bug, have an idea for a new test, or want to improve existing ones.

## License

This test suite is released under the MIT License

## Trademarks

* OpenVX™ and the OpenVX logo are trademarks of the Khronos Group Inc.
* MIVisionX™ is a trademark of Advanced Micro Devices, Inc.
* OpenCL™ is a trademark of Apple Inc. used under license by Khronos.
* All other trademarks are the property of their respective owners.