<p align="center"><img width="30%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/OpenVX_logo.png" /></p>

# AMD OpenVX&trade;

AMD OpenVX&trade; is a highly optimized conformant open-source implementation of the [Khronos OpenVX&trade; 1.3](https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html) computer vision specification. It allows for rapid prototyping as well as fast execution on a wide range of computer hardware, including small embedded `AMD64` CPUs and large workstation discrete GPUs.

[OpenVX&trade;](https://www.khronos.org/openvx/) is an open, royalty-free standard for cross-platform acceleration of computer vision applications. It provides a set of optimized primitives for low-level image processing, computer vision, and neural net operators with performance portability across CPUs, GPUs, and special-function hardware.

## Features

* Highly optimized for both x86 CPU and OpenCL/HIP GPU backends
* Supports hardware from low-power embedded APUs to workstation discrete GPUs
* Supports Windows, Linux, and macOS
* Graph optimizer that analyzes the entire processing pipeline to remove/replace/merge functions for improved performance and minimized bandwidth
* Scripting support with [RunVX](../utilities/runvx/README.md) for rapid prototyping without recompilation

## OpenVX 1.3 Vision Conformance

AMD OpenVX implements the full [Vision Conformance Feature Set](https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html), which includes:

* **Base Feature Set**: Core [framework objects](https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html#sec_framework_objects) (`vx_context`, `vx_graph`, `vx_kernel`, `vx_node`, `vx_parameter`, `vx_reference`, `vx_meta_format`, `vx_delay`) for constructing and executing OpenVX graphs.

* **Vision Data Objects**: [Data objects](https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html#sec_data_objects) including `vx_image`, `vx_array`, `vx_convolution`, `vx_distribution`, `vx_lut`, `vx_matrix`, `vx_pyramid`, `vx_remap`, `vx_scalar`, `vx_threshold`, and `vx_object_array`.

* **Vision Functions**: 36 [vision processing functions](https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html#group_vision_functions) including edge detection (Canny, Sobel), feature detection (Harris, FAST corners), filtering (Gaussian, Median, Box), geometric transforms (Warp, Remap, Scale), color conversion, histogram, optical flow, and more.

* **VXU Immediate Functions**: The [VXU library](https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html#group_vxu) provides all OpenVX operators as directly callable C functions without requiring graph construction, useful for porting existing vision applications.

[Khronos OpenVX&trade; 1.0.1](https://www.khronos.org/registry/OpenVX/specs/1.0.1/html/index.html) conformant implementation is available in [MIVisionX Lite](https://github.com/ROCm/MIVisionX/tree/openvx-1.0.1).

## OpenVX Extensions

AMD OpenVX can be extended with additional modules. See [amd_openvx_extensions](../amd_openvx_extensions/README.md) for the available OpenVX extension module: `amd_rpp`, which provides RPP image/tensor augmentation as OpenVX kernels.

## Prerequisites

* **CPU**: [AMD64](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* **GPU**: [AMD Radeon&trade; Graphics](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) [optional]
  + Windows: Install the latest [drivers](https://www.amd.com/en/support) and [OpenCL SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases)
  + Linux: Install the [ROCm Core SDK](https://rocm.docs.amd.com/en/latest/install/rocm.html) (ROCm `7.13` or later)
* **APU**: [AMD Radeon&trade; `Mobile`/`Embedded`](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) [optional]

## Build Instructions

AMD OpenVX is built as part of the [MIVisionX](../README.md) project.

* Refer to [openvx/include/VX](openvx/include/VX) for Khronos OpenVX standard header files
* Refer to [openvx/include/vx_ext_amd.h](openvx/include/vx_ext_amd.h) for AMD vendor extensions

### Build using CMake

AMD OpenVX is built as part of the top-level MIVisionX project. Run CMake from the **MIVisionX repo root**, not from within `amd_openvx/`:

```shell
# From the MIVisionX repo root
mkdir build && cd build
cmake ..          # HIP backend (default on Linux)
make -j$(nproc)
```

On Windows, build with CMake as well (the legacy Visual Studio `.sln`/`.vcxproj` files have been removed). The default Windows backend is `OpenCL`; pass `-DGPU_SUPPORT=OFF` for a CPU-only build. Optionally install [OpenCV](https://github.com/opencv/opencv/releases) (set `OpenCV_DIR` to the `OpenCV/build` folder) to enable RunVX camera capture and image display.

> [!NOTE]
> AMD GPU HIP backend is not supported on Windows.

OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
