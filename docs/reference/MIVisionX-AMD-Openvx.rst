.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _amd-openvx:

******************************************
AMD OpenVX
******************************************

AMD OpenVX is an open-source implementation of the |openvx|_ computer vision specification. 

AMD OpenVX can be found in the `MIVisionX GitHub repository <https://github.com/ROCm/MIVisionX/blob/develop/amd_openvx>`_.

`RunVX <https://github.com/ROCm/MIVisionX/tree/develop/utilities/runvx>`_ provides a means for rapid prototyping without re-compiling.

The AMD OpenVX core engine supports the ``CPU``, ``HIP``, and ``OpenCL`` backends.

MIVisionX ships one `OpenVX extension module <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx_extensions>`_:

* `amd_rpp <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx_extensions/amd_rpp>`_: Exposes `ROCm Performance Primitives (RPP) <https://rocm.docs.amd.com/projects/rpp/en/latest/index.html>`_ image and tensor augmentation functions as OpenVX kernels. Supports the ``CPU`` and ``HIP`` backends only.



.. |trade| raw:: html

    &trade;

.. |openvx| replace:: Khronos OpenVX\ |trade| Version 1.3 
.. _openvx: https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html
