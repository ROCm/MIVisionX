.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _amd-openvx-install:

******************************************
AMD OpenVX installation
******************************************

AMD OpenVX is built as part of the top-level MIVisionX CMake project. Build MIVisionX to produce ``libopenvx.so`` and ``libvxu.so``.

Prerequisites
=============

* **CPU**: AMD64
* **GPU**: AMD Radeon Graphics [optional]

  * Windows: install the latest AMD `drivers <https://www.amd.com/en/support>`_ and the `OpenCL SDK <https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases>`_
  * Linux: install the `ROCm Core SDK <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_ (ROCm ``7.13`` or later)

Refer to `openvx/include/VX <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx/openvx/include>`_ for the Khronos OpenVX standard headers and to `openvx/include/vx_ext_amd.h <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx/openvx/include/vx_ext_amd.h>`_ for AMD vendor extensions.

.. note::

    The AMD GPU ``HIP`` backend is not supported on Windows. On Windows the default backend is ``OpenCL``.

Build
=====

Build from the **MIVisionX repository root**. Optionally install `OpenCV <https://github.com/opencv/opencv/releases>`_ to enable ``RunVX`` camera capture and image display (set ``OpenCV_DIR`` to the ``OpenCV/build`` folder).

.. code-block:: shell

    git clone https://github.com/ROCm/MIVisionX.git
    cd MIVisionX
    cmake -B build               # HIP backend (default on Linux); add -DBACKEND=OCL for OpenCL, -DGPU_SUPPORT=OFF for CPU
    cmake --build build --config Release
    sudo cmake --install build
