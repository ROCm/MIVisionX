# AMD RPP Extension (`vx_rpp`)

The AMD VX RPP extension (`vx_rpp`) is an OpenVX module that exposes [ROCm Performance Primitives (RPP)](https://github.com/ROCm/rpp) image and tensor augmentation functions as OpenVX kernels. It is the backend used by [rocAL](https://github.com/ROCm/rocAL) for data augmentation pipelines.

Load the extension at runtime with:

```c
vxLoadKernels(context, "vx_rpp");
```

> [!NOTE]
> `vx_rpp` supports the `CPU` and `HIP` backends. When MIVisionX is built with the `OpenCL` backend, `vx_rpp` is built in CPU-only mode (RPP has dropped OpenCL support).

## Available kernels

| Kernel name | Description |
|-------------|-------------|
| `org.rpp.Blend` | Alpha-blend two images |
| `org.rpp.Blur` | Box blur |
| `org.rpp.Brightness` | Brightness adjustment |
| `org.rpp.ColorTemperature` | Color temperature shift |
| `org.rpp.Contrast` | Contrast adjustment |
| `org.rpp.Exposure` | Exposure adjustment |
| `org.rpp.Fisheye` | Fisheye lens distortion |
| `org.rpp.Flip` | Horizontal/vertical flip |
| `org.rpp.Fog` | Fog overlay effect |
| `org.rpp.GammaCorrection` | Gamma correction |
| `org.rpp.Resize` | Image resize |
| `org.rpp.Jitter` | Random jitter |
| `org.rpp.LensCorrection` | Lens distortion correction |
| `org.rpp.Pixelate` | Pixelation effect |
| `org.rpp.Rain` | Rain overlay effect |
| `org.rpp.ResizeCrop` | Resize and crop |
| `org.rpp.Rotate` | Image rotation |
| `org.rpp.NoiseSnp` | Salt-and-pepper noise |
| `org.rpp.Snow` | Snow overlay effect |
| `org.rpp.Vignette` | Vignette effect |
| `org.rpp.WarpAffine` | Affine warp |

For the full OpenVX API for each kernel, see [include/vx_ext_rpp.h](include/vx_ext_rpp.h).

## Prerequisites

* AMD OpenVX&trade; library (`libopenvx.so`) — built as part of MIVisionX
* [RPP](https://github.com/ROCm/rpp) `3.1.0` or later — provided by the ROCm Core SDK
* CMake `3.10` or later

## Build

`amd_rpp` is built automatically as part of the top-level MIVisionX CMake build. To build it standalone:

```shell
mkdir build && cd build
cmake ../
make -j$(nproc)
```

OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
