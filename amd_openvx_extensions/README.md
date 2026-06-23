# AMD OpenVX&trade; Extensions

The OpenVX framework provides a mechanism to add new vision functions to OpenVX by 3rd party vendors. This project provides the following OpenVX module that extends [AMD OpenVX&trade;](../amd_openvx/README.md#amd-openvx) (amd_openvx), which contains the AMD OpenVX&trade; Core Engine.

* [amd_rpp](amd_rpp/README.md): OpenVX extension providing an interface to the ROCm Performance Primitives ([RPP](https://github.com/ROCm/rpp)) functions. This extension is used to enable [rocAL](https://github.com/ROCm/rocAL) to perform image and tensor augmentation.

**NOTE:**
* The AMD OpenVX&trade; core engine supports the `CPU`, `HIP`, and `OpenCL` backends. The `amd_rpp` extension supports the `CPU` and `HIP` backends only (RPP has dropped OpenCL support); when the core is built with the `OpenCL` backend, `amd_rpp` is built in CPU-only mode.
* OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
