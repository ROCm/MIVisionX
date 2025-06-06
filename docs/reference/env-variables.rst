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
    :widths: 49,51

    * - **Environment variable**
      - **Value**

    * - | ``AGO_DEFAULT_TARGET``
        | Sets the default execution target for OpenVX kernels.
      - | "CPU": Execute kernels on CPU
        | "GPU": Execute kernels on GPU

    * - | ``AGO_BUFFER_MERGE_FLAGS``
        | Controls virtual data buffer merging and marks if CPU access is not needed for virtual buffers.
      - | Integer bitmask value
        | Controls GPU buffer allocation for virtual data
        | Optimizes memory usage for GPU-only operations

    * - | ``AGO_THREAD_CONFIG``
        | Initializes thread configuration through environment variable.
      - | Integer value specifying thread configuration
        | Controls threading behavior for CPU execution

    * - | ``VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS``
        | Sets OpenVX graph optimizer flags to control specific optimizations.
      - | ``0x00000000``: Default options (all optimizations enabled)
        | ``0x00000001``: Disable drama divide
        | ``0x00000002``: Don't remove unnecessary copy operations
        | ``0x00000004``: Don't remove nodes with unused outputs
        | ``0x00000008``: Don't perform node merge
        | ``0x00000010``: Don't convert 8-bit images to 1-bit images
        | ``0x00000020``: Don't merge supernodes
        | Values can be combined with bitwise OR

    * - | ``AGO_DUMP_GRAPH``
        | Enables graph dumping for debugging purposes.
      - | ``0`` or unset: Disable graph dump
        | ``1``: Enable graph dump


GPU and Device Configuration
============================

The GPU and device configuration environment variables for MIVisionX are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 49,51

    * - **Environment variable**
      - **Value**

    * - | ``GPU_ENABLE_WGP_MODE``
        | Controls Workgroup Processor (WGP) mode on RDNA GPUs that support both CU and WGP modes.
      - | ``0``: Disable WGP mode (use CU mode)
        | ``1``: Enable WGP mode (default)
        | Only applies to GPUs with major version >= 10

OpenCL Configuration
====================

The OpenCL configuration environment variables for MIVisionX are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 49,51

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
    :widths: 49,51

    * - **Environment variable**
      - **Value**

    * - | ``AGO_DUMP_GPU``
        | Dumps OpenCL kernels to files with specified path prefix.
      - | String path prefix for dump files
        | Output files: $(AGO_DUMP_GPU)-<group>.cl
        | Used for analyzing generated OpenCL kernels

    * - | ``ENABLE_DEBUG_DUMP_CL_BUFFERS``
        | Enables dumping of OpenCL buffers for debugging.
      - | ``0`` or unset: Disable buffer dumping
        | ``1``: Enable OpenCL buffer dumping

    * - | ``ENABLE_DEBUG_MESSAGES``
        | Controls general debug message output.
      - | ``0``: Disable debug messages (default)
        | ``1``: Enable debug messages

    * - | ``ENABLE_LOCAL_DEBUG_MESSAGES``
        | Provides detailed debug information on OpenCL workflow.
      - | ``0`` or unset: Disable detailed debugging
        | ``1``: Enable detailed OpenCL debug information

    * - | ``SHOW_DEBUG_HIERARCHICAL_LEVELS``
        | Shows debug hierarchical node levels in the OpenVX graph.
      - | ``0``: Disable hierarchical level debugging (default)
        | ``1``: Enable hierarchical level debugging

Stitching Configuration
=======================

The stitching configuration environment variables for MIVisionX are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 49,51

    * - **Environment variable**
      - **Value**

    * - | ``DRAW_SEAM``
        | Enables drawing of seam lines for verification in stitching operations.
      - | ``0``: Disable seam drawing (default)
        | ``1``: Enable seam drawing visualization

    * - | ``SEAM_ADJUST``
        | Controls seam adjustment behavior in stitching.
      - | ``0``: Disable seam adjustment (default)
        | ``1``: Enable dynamic seam adjustment

    * - | ``PRINT_COST``
        | Enables printing of cost calculations during seam finding.
      - | ``0``: Disable cost printing (default)
        | ``1``: Print cost matrix values for debugging

    * - | ``SEAM_FIND_TARGET``
        | Sets the target method for seam finding algorithms.
      - | Integer value (algorithm selector)
        | Different values select different seam finding methods

    * - | ``VIEW_SCENE_CHANGE``
        | Controls view scene change detection behavior.
      - | ``0``: Disable scene change detection
        | ``1``: Enable scene change detection
        | Higher values: More sensitive detection

    * - | ``SEAM_THRESHOLD``
        | Sets threshold value for seam detection algorithms.
      - | Integer value (0-255 typical range)
        | Lower values: More sensitive seam detection
        | Higher values: Less sensitive seam detection

    * - | ``SCENE_DURATION``
        | Sets duration for scene analysis in stitching.
      - | Integer value (likely in frames or milliseconds)
        | Controls temporal window for scene analysis

    * - | ``SEAM_FREQUENCY``
        | Controls frequency of seam finding operations.
      - | Integer value (likely in frames)
        | ``1``: Find seams every frame
        | Higher values: Find seams less frequently

    * - | ``SEAM_FIND_MODE``
        | Sets the operational mode for seam finding algorithms.
      - | Integer value (mode selector)
        | Different values select different seam finding strategies

    * - | ``LOOM_SEAM_FIND_DISABLE``
        | Disables seam finding in LOOM stitching operations.
      - | ``0`` or unset: Enable seam finding
        | ``1``: Disable seam finding


Chroma Key Configuration
========================

The chroma key configuration environment variables for MIVisionX are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 49,51

    * - **Environment variable**
      - **Value**

    * - | ``CHROMAKEY_MASK``
        | Controls chroma key masking behavior.
      - | Integer value specifying chroma key mask settings
        | Used for green screen/chroma key operations

    * - | ``CHROMAKEY_MERGE``
        | Controls chroma key merging operations.
      - | Integer value specifying chroma key merge settings
        | Controls how chroma key layers are combined

File I/O and Auxiliary Operations
=================================

The file I/O and auxiliary operation environment variables for MIVisionX are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 49,51

    * - **Environment variable**
      - **Value**

    * - | ``LOOMIO_AUX_DUMP``
        | Specifies file path for auxiliary LOOM I/O data dumping.
      - | String path to dump file
        | Enables debugging of LOOM I/O operations

Model Deployment
=================

The model deployment environment variables for MIVisionX are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 49,51

    * - **Environment variable**
      - **Value**

    * - | ``MIVISIONX_MODEL_COMPILER_PATH``
        | Overrides the default path to the MIVisionX model compiler.
      - | String path to model compiler executable
        | Default: "/opt/rocm/libexec/mivisionx/model_compiler"

Neural Network Configuration
=============================

The neural network configuration environment variables for MIVisionX are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``NN_MIOPEN_CBR_MODE``
        | Controls MIOpen Convolution-BatchNorm-ReLU fusion mode for neural networks.
      - ``1``
      - | Integer value specifying CBR mode
        | ``0``: Disable operator fusion optimization
        | ``1``: Enable operator fusion optimization

