.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _amd-openvx:

******************************************
AMD OpenVX
******************************************

AMD OpenVX is an open-source implementation of the |openvx|_ computer vision specification. 

AMD OpenVX can be found in the `MIVisionX GitHub repository <https://github.com/ROCm/MIVisionX/blob/develop/amd_openvx>`_.

`RunVX <https://github.com/ROCm/MIVisionX/tree/master/utilities/runvx>`_ provides a means for rapid prototyping without re-compiling.

In addition to implementing Khronos OPenVX functions and data type, `AMD OpenVX extends OpenVX <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx_extensions>`_ with the following modules and libraries:

| `amd_custom <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx_extensions/amd_custom>`_: Custom node extension module. 
| `amd_loomsl <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx_extensions/amd_loomsl>`_: Radeon LOOM stitching library for live 360-degree video applications.
| `amd_media <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx_extensions/amd_media>`_: Media extension module for video and JPG encoding and decoding.
| `amd_migraphx <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx_extensions/amd_migraphx>`_: Imports the `MIGraphx <https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/index.html>`_ library into an OpenVX graph.
| `amd_nn <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx_extensions/amd_nn>`_: Neural network module built on top of the `MIOpen <https://rocm.docs.amd.com/projects/MIOpen/en/latest/index.html>`_ library.
| `amd_opencv <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx_extensions/amd_opencv>`_: Used to access `OpenCV <https://opencv.org/>`_ as OpenVX kernels.
| `amd_rpp <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx_extensions/amd_rpp>`_: Used to access `ROCm Performance Primitives (RPP) <https://rocm.docs.amd.com/projects/rpp/en/latest/index.html>`_ as OpenVX kernels
| `amd_winml <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx_extensions/amd_media>`_: Used to access `Windows Machine Learning (WinML) <https://github.com/microsoft/Windows-Machine-Learning>`_ as OpenVX kernels.



.. |trade| raw:: html

    &trade;

.. |openvx| replace:: Khronos OpenVX\ |trade| Version 1.3 
.. _openvx: https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html
