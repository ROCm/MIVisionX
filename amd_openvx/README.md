<p align="center"><img width="30%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/OpenVX_logo.png" /></p>

# AMD OpenVX&trade;

AMD OpenVX&trade; is a highly optimized conformant open-source implementation of the [Khronos OpenVX&trade; 1.3.2](https://registry.khronos.org/OpenVX/specs/1.3.2/html/OpenVX_Specification_1_3_2.html) computer vision specification. It allows for rapid prototyping as well as fast execution on a wide range of computer hardware, including small embedded `AMD64` CPUs and large workstation discrete GPUs.

## Features

* Highly optimized for both x86 CPU and OpenCL/HIP GPU backends
* Supports hardware from low-power embedded APUs to workstation discrete GPUs
* Supports Windows, Linux, and macOS
* Graph optimizer that analyzes the entire processing pipeline to remove/replace/merge functions for improved performance and minimized bandwidth
* Scripting support with [RunVX](../utilities/runvx/README.md) — execute OpenVX graphs from GDF text files without writing or recompiling C code

## OpenVX 1.3.2 Vision Conformance

AMD OpenVX implements the full [Vision Conformance Feature Set](https://registry.khronos.org/OpenVX/specs/1.3.2/html/OpenVX_Specification_1_3_2.html), which includes:

* **Base Feature Set**: Core [framework objects](https://registry.khronos.org/OpenVX/specs/1.3.2/html/OpenVX_Specification_1_3_2.html#sec_framework_objects) (`vx_context`, `vx_graph`, `vx_kernel`, `vx_node`, `vx_parameter`, `vx_reference`, `vx_meta_format`, `vx_delay`) for constructing and executing OpenVX graphs.

* **Vision Data Objects**: [Data objects](https://registry.khronos.org/OpenVX/specs/1.3.2/html/OpenVX_Specification_1_3_2.html#sec_data_objects) including `vx_image`, `vx_array`, `vx_convolution`, `vx_distribution`, `vx_lut`, `vx_matrix`, `vx_pyramid`, `vx_remap`, `vx_scalar`, `vx_threshold`, and `vx_object_array`.

* **Vision Functions**: 36 [vision processing functions](https://registry.khronos.org/OpenVX/specs/1.3.2/html/OpenVX_Specification_1_3_2.html#group_vision_functions) including edge detection (Canny, Sobel), feature detection (Harris, FAST corners), filtering (Gaussian, Median, Box), geometric transforms (Warp, Remap, Scale), color conversion, histogram, optical flow, and more.

* **VXU Immediate Functions**: The [VXU library](https://registry.khronos.org/OpenVX/specs/1.3.2/html/OpenVX_Specification_1_3_2.html#group_vxu) provides all OpenVX operators as directly callable C functions without requiring graph construction, useful for porting existing vision applications.

[Khronos OpenVX&trade; 1.0.1](https://registry.khronos.org/OpenVX/specs/1.0.1/html/index.html) conformant implementation is available in [MIVisionX Lite](https://github.com/ROCm/MIVisionX/tree/openvx-1.0.1).

## OpenVX Extensions

AMD OpenVX can be extended with additional modules. See [amd_openvx_extensions](../amd_openvx_extensions/README.md) for the available OpenVX extension module: `amd_rpp`, which provides RPP image/tensor augmentation as OpenVX kernels.

## Prerequisites

* **CPU**: [AMD64](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html)
* **GPU**: [AMD Radeon&trade; Graphics](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html) [optional]
  + Windows: Install the latest [drivers](https://www.amd.com/en/support) and [OpenCL SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases)
  + Linux: Install the [ROCm Core SDK](https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html) (ROCm `7.13` or later)
* **APU**: [AMD Radeon&trade; `Mobile`/`Embedded`](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html) [optional]

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

On Windows, the default backend is `OpenCL`; pass `-DGPU_SUPPORT=OFF` for a CPU-only build. Optionally install [OpenCV](https://github.com/opencv/opencv/releases) (set `OpenCV_DIR` to the `OpenCV/build` folder) to enable RunVX camera capture and image display.

> [!NOTE]
> AMD GPU HIP backend is not supported on Windows.

## Profiling with rocprof / ROCTX

AMD OpenVX can emit [ROCTX](https://rocm.docs.amd.com/projects/roctracer/en/latest/) markers so graph execution can be correlated with GPU kernel activity in [`rocprofv3`](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) traces. This is useful for understanding per-node GPU time, host&harr;device transfers, and the effect of `MIVISIONX_HIP_CU_COUNT`.

The instrumentation is **off by default** and adds no dependency unless explicitly enabled.

### Enable at build time

```shell
# From the MIVisionX repo root
cmake -B build -S . -DMIVISIONX_ENABLE_ROCPROF=ON
cmake --build build -j$(nproc)
```

When `MIVISIONX_ENABLE_ROCPROF=ON`, CMake links `libroctx64` from `${ROCM_PATH}`. If the library is not found the build still succeeds with tracing compiled out (a warning is printed).

### Enable at run time

Even when compiled in, no markers are emitted unless `MIVISIONX_ROCPROF` is set to `1`/`true`/`yes` (honored on both Linux and Windows):

```shell
export MIVISIONX_ROCPROF=1
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
rocprofv3 --marker-trace --kernel-trace --memory-copy-trace --memory-allocation-trace \
    --output-format csv pftrace --output-directory ./trace \
    -- runvx -frames:10 -affinity:GPU graph.gdf
```

### What you get in the trace

| Range / event | Meaning |
|---------------|---------|
| `MIVisionX: agoExecuteGraph` | one full graph execution |
| `MIVisionX: pipelined execution` | one pipelined/streaming graph iteration (wraps `agoExecuteGraph`) |
| `MIVisionX: level N` | all nodes at hierarchical level N |
| `<kernel name>` (e.g. `com.amd.openvx.ColorConvert_RGB_RGBX`) | a single node's launch/execute |
| `MIVisionX: copy-in <kernel> paramN` | host&rarr;device transfer of an input, attributed to the node/param that needed it |
| `MIVisionX: copy-out <kernel>` | device&rarr;host transfer of a node's output |
| `MIVisionX: GPU sync/wait` | where the graph blocks on GPU completion (`hipStreamSynchronize`/`clFinish`) &mdash; the real GPU kernel time lands here, since per-node ranges only cover the asynchronous launch |
| kernel dispatches, memory copies, allocations | captured by `rocprofv3` directly (`--kernel-trace`, `--memory-copy-trace`, `--memory-allocation-trace`) |

> [!NOTE]
> GPU kernel launches are asynchronous. The per-node ranges (`<kernel name>`) measure the time to **enqueue** the kernel, not to run it. Actual device execution is bounded by the `MIVisionX: GPU sync/wait` range and shown precisely by the kernel-dispatch trace.

Open the resulting `*.pftrace` at [ui.perfetto.dev](https://ui.perfetto.dev) to see the ROCTX ranges, kernel dispatches, and memory transfers on a single timeline.

### Continuous integration

The `ROCprof GDF profiling` workflow (`.github/workflows/rocprof-gdf.yml`) builds with `MIVISIONX_ENABLE_ROCPROF=ON` and profiles a small set of GDFs under `rocprofv3`. Because it uses GPU runners, it is gated so it does not run on unrelated pull requests:

* **Push** to `master`/`main`/`develop` &mdash; runs only when a tracing-relevant file changes (`amd_openvx/openvx/ago/**`, `amd_openvx/openvx/CMakeLists.txt`, the profiling script, or the workflow).
* **Pull request** &mdash; runs automatically when one of those files changes, or on demand for any PR by adding the **`run-rocprof`** label.

OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
