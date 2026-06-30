# AMD OpenVX&trade; Extensions

The OpenVX framework provides a mechanism for vendors to add new vision functions. MIVisionX ships the following extension module on top of the [AMD OpenVX&trade;](../amd_openvx/README.md#amd-openvx) core engine:

* [amd_rpp](amd_rpp/README.md): Exposes [ROCm Performance Primitives (RPP)](https://github.com/ROCm/rpp) image and tensor augmentation functions as OpenVX kernels. Used by [rocAL](https://github.com/ROCm/rocAL) for data augmentation pipelines.

> [!NOTE]
> The AMD OpenVX&trade; core engine supports the `CPU`, `HIP`, and `OpenCL` backends. `amd_rpp` supports the `CPU` and `HIP` backends only (RPP has dropped OpenCL support); when the core is built with the `OpenCL` backend, `amd_rpp` is built in CPU-only mode.

OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
