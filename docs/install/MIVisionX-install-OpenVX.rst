.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _amd-openvx-install:

******************************************
AMD OpenVX installation
******************************************

Pre-requisites
==============

* **CPU**: AMD64
* **GPU**: AMD Radeon Graphics [optional]

  + Windows: install the latest drivers and OpenCL SDK `download <https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases>`_
  + Linux: install the `ROCm Core SDK <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_ (ROCm ``7.13`` or later)


Build Instructions
==================

Build this project to generate AMD OpenVX library 

* Refer to `openvx/include/VX <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx/openvx/include>`_ for Khronos OpenVX standard header files.
* Refer to `openvx/include/vx_ext_amd.h <https://github.com/ROCm/MIVisionX/tree/develop/amd_openvx/openvx/include/vx_ext_amd.h>`_ for vendor extensions in AMD OpenVX library

.. note::
  AMD GPU ``HIP`` backend is not supported on Windows. On Windows the default backend is ``OpenCL``.

Build using CMake
------------------

* Install CMake 3.10 or later
* Optionally install `OpenCV <https://github.com/opencv/opencv/releases>`_ to enable the ``RunVX`` tool to support camera capture and image display (set ``OpenCV_DIR`` to the ``OpenCV/build`` folder)
* Use CMake to configure and generate the build files, then build (pass ``-DGPU_SUPPORT=OFF`` for a CPU-only build)

.. code-block:: shell

    cmake -B build
    cmake --build build --config Release