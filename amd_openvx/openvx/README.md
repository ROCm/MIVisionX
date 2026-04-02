# AMD OpenVX&trade; Library

AMD OpenVX&trade; library is a highly optimized open-source implementation of the [Khronos OpenVX 1.3](https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html) computer vision specification. This directory contains the core library source code.

## Libraries

This module builds two shared libraries:

| Library | Description |
|---------|-------------|
| `libopenvx.so` | Core OpenVX implementation with graph optimizer and vision functions |
| `libvxu.so` | VXU immediate function library for calling OpenVX operators directly without graph construction |

## Backend Support

The library supports three compute backends:

| Backend | Platform | Description |
|---------|----------|-------------|
| CPU | Linux, Windows, macOS | Default backend using SSE4.2 optimized x86 kernels |
| HIP | Linux | AMD GPU acceleration via ROCm HIP |
| OpenCL | Linux, Windows | GPU acceleration via OpenCL |

## Directory Structure

| Directory | Contents |
|-----------|----------|
| `ago/` | Core engine -- graph optimizer (DRAMA), CPU/GPU kernel implementations, and OpenCL/HIP utilities |
| `api/` | OpenVX API and VXU immediate function entry points |
| `hipvx/` | HIP GPU kernel implementations (arithmetic, color, filter, geometric, logical, statistical, vision) |
| `include/VX/` | Khronos OpenVX standard header files |
| `include/vx_ext_amd.h` | AMD vendor extension header |

## API Reference

* [OpenVX Standard Headers](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/doxygen/html/files.html) (`include/VX/`)
* [AMD Vendor Extensions](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/doxygen/html/vx__ext__amd_8h.html) (`include/vx_ext_amd.h`)

**NOTE:** OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
