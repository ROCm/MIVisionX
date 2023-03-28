## AMD OpenVX&trade; Extensions

The OpenVX framework provides a mechanism to add new vision functionality to OpenVX by vendors. This project has below mentioned OpenVX [modules](amd_openvx_extensions#amd-openvx-extensions-amd_openvx_extensions) and utilities to extend [amd_openvx](amd_openvx#amd-openvx-amd_openvx), which contains the AMD OpenVX&trade; Core Engine.

![MIVisionX OpenVX Extensions](images/MIVisionX-OpenVX-Extensions.png)

* [amd_loomsl](amd_openvx_extensions/amd_loomsl): AMD Radeon Loom stitching library for live 360 degree video applications
* [amd_media](amd_openvx_extensions/amd_media): `vx_amd_media` is an OpenVX AMD media extension module for encode and decode
* [amd_migraphx](amd_openvx_extensions/amd_migraphx): amd_migraphx extension integrates the <a href="https://github.com/ROCmSoftwarePlatform/AMDMIGraphX#amd-migraphx" target="_blank"> AMD's MIGraphx </a> into an OpenVX graph. This extension allows developers to combine the vision funcions in OpenVX with the MIGraphX and build an end-to-end application for inference.
* [amd_nn](amd_openvx_extensions/amd_nn#openvx-neural-network-extension-library-vx_nn): OpenVX neural network module
* [amd_opencv](amd_openvx_extensions/amd_opencv#amd-module-for-opencv-interop-from-openvx-vx_opencv): OpenVX module that implements a mechanism to access OpenCV functionality as OpenVX kernels
* [amd_rpp](amd_openvx_extensions/amd_rpp): OpenVX extension providing an interface to some of the [RPP](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp)'s (ROCm Performance Primitives) functions. This extension is used to enable [rocAL](rocAL/README.md) to perform image augmentation.
* [amd_winml](amd_openvx_extensions/amd_winml#amd-winml-extension): WinML extension will allow developers to import a pre-trained ONNX model into an OpenVX graph and add hundreds of different pre & post processing `vision` / `generic` / `user-defined` functions, available in OpenVX and OpenCV interop, to the input and output of the neural net model. This will allow developers to build an end to end application for inference.
