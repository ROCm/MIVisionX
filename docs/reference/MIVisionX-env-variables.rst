.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, environment variable, environment

.. _env-variables:

******************************************
MIVisionX environment variables
******************************************

The following environment variables control the runtime behavior of MIVisionX.

Core OpenVX configuration
=========================

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - Environment variable
      - Values

    * - | ``AGO_DEFAULT_TARGET``
        | Sets the default execution target for OpenVX kernels.
      - | ``"GPU"``: execute kernels on the GPU
        | ``"CPU"``: execute kernels on the CPU
        | Unset: use the library default

    * - | ``AGO_BUFFER_MERGE_FLAGS``
        | Controls buffer-merging optimization.
      - | Integer bitmask
        | ``0``: disable buffer merging
        | Higher values enable more aggressive merging

    * - | ``AGO_THREAD_CONFIG``
        | Configures the number of CPU threads used for graph execution.
      - | ``0``: use the default thread count
        | Positive integer: use that many threads

    * - | ``VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS``
        | Sets graph optimizer flags for AMD extensions.
      - | Integer bitmask
        | ``0``: disable optimizations
        | Positive values enable specific optimizations

GPU and device configuration
============================

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - Environment variable
      - Values

    * - | ``GPU_ENABLE_WGP_MODE``
        | Controls Workgroup Processor (WGP) mode on RDNA GPUs that support both CU and WGP modes (GPU major version ≥ 10).
      - | ``0``: use CU mode
        | ``1`` (or any non-zero): use WGP mode (default)

OpenCL configuration
====================

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - Environment variable
      - Values

    * - | ``AGO_OPENCL_PLATFORM``
        | Overrides the default OpenCL platform selection.
      - | String — name of the OpenCL platform to use

    * - | ``AGO_OPENCL_VERSION_CHECK``
        | Controls OpenCL version checking.
      - | String — set to disable or relax version validation

    * - | ``AGO_OPENCL_BUILD_OPTIONS``
        | Appends options to the OpenCL kernel compiler invocation.
      - | String — valid OpenCL compiler flags

    * - | ``AGO_OPENCL_DEVICE_INFO``
        | Controls OpenCL device information reporting (useful for debugging).
      - | String — controls the verbosity of device info output

Debugging and profiling
========================

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - Environment variable
      - Values

    * - | ``AGO_DUMP_GPU``
        | Dumps compiled GPU kernel source to disk for inspection.
      - | Set to any non-empty string to enable
