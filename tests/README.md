# MIVisionX Test Suite

This folder contains the MIVisionX test suite, designed to verify the installation and functionality of the MIVisionX™ toolkit. It covers the core OpenVX™ implementation and the AMD RPP extension across the `CPU`, `OpenCL™`, and `HIP` backends.

## Overview

The MIVisionX test suite validates the proper installation, functionality, and performance of the MIVisionX libraries. It's an essential tool for developers and users to ensure the integrity of their MIVisionX setup across the supported backends.

## Test Categories

### Core OpenVX Tests

These tests focus on the fundamental OpenVX specification and AMD's implementation.

* **[OpenVX Conformance Tests](openvx_conformance_tests)**: Runs the official OpenVX 1.3 Conformance tests for the Vision Feature Set, targeting both CPU and GPU (OpenCL & HIP backend) implementations.
* **[AMD OpenVX GDF Tests](amd_openvx_gdfs)**: Exercises AMD OpenVX functionality across CPU and GPU backends using `RunVX`.
* **[Vision Tests](vision_tests)**: Conducts tests on OpenVX vision functions for both verification and performance assessment.

### AMD Extension Tests

* **[VX_RPP Tests](vx_rpp_tests)** (`vx_rpp`): Tests the AMD RPP extension, an OpenVX module that provides an interface to access RPP (ROCm Performance Primitives) functionality as OpenVX kernels. These tests utilize GDFs.

## Getting Started

To run these tests, you typically need a complete MIVisionX installation. Refer to the main MIVisionX documentation for detailed build and installation instructions. After installing the `mivisionx-test` package (or building from source), run:

```shell
mkdir mivisionx-test && cd mivisionx-test
cmake /opt/rocm/share/mivisionx/test/
ctest -VV
```

