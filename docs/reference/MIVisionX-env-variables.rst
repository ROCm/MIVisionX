.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, environment variable, environment

.. _env-variables:

******************************************
MIVisionX environment variables
******************************************

This section describes the most important MIVisionX environment variables,
which are grouped by functionality.

Core OpenVX Configuration
=========================

The core OpenVX configuration environment variables for MIVisionX are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``AGO_DEFAULT_TARGET``
        | Sets the default execution target for OpenVX kernels.
      - | "GPU": Execute kernels on GPU
        | "CPU": Execute kernels on CPU
        | Unset: Use library default target

    * - | ``AGO_BUFFER_MERGE_FLAGS``
        | Controls buffer merging optimization flags.
      - | Integer bitmask value
        | Higher values: More aggressive merging
        | 0: Disable buffer merging

    * - | ``AGO_THREAD_CONFIG``
        | Configures thread usage for CPU execution.
      - | Integer value (likely number of threads)
        | 0: Use default threading
        | Positive integer: Specific thread count

    * - | ``VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS``
        | Sets OpenVX graph optimizer flags for AMD extensions.
      - | Integer bitmask value
        | 0: Disable optimizations
        | Positive values: Enable specific optimizations

GPU and Device Configuration
============================

The GPU and device configuration environment variables for MIVisionX are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``GPU_ENABLE_WGP_MODE``
        | Controls Workgroup Processor (WGP) mode on RDNA GPUs that support both CU and WGP modes.
      - | 0: Disable WGP mode (use CU mode)
        | 1 or any non-zero: Enable WGP mode (default)
        | Only applies to GPUs with major version >= 10

OpenCL Configuration
====================

The OpenCL configuration environment variables for MIVisionX are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``AGO_OPENCL_PLATFORM``
        | Overrides the default OpenCL platform selection.
      - | String specifying OpenCL platform name
        | Used to select specific OpenCL implementation

    * - | ``AGO_OPENCL_VERSION_CHECK``
        | Controls OpenCL version checking behavior.
      - | String value controlling version validation
        | May disable or modify version requirements

    * - | ``AGO_OPENCL_BUILD_OPTIONS``
        | Specifies additional OpenCL kernel build options.
      - | String containing OpenCL compiler flags
        | Passed to OpenCL kernel compilation

    * - | ``AGO_OPENCL_DEVICE_INFO``
        | Controls OpenCL device information reporting.
      - | String value controlling device info output
        | Used for debugging device capabilities

Debugging and Profiling
========================

The debugging and profiling environment variables for MIVisionX are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``AGO_DUMP_GPU``
        | Enables GPU kernel dumping for debugging purposes.
      - | String value enabling GPU kernel dump
        | Used for analyzing GPU kernel behavior
