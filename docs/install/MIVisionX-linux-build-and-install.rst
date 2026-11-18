.. meta::
  :description: MIVisionX building and installing on Linux
  :keywords: MIVisionX, ROCm, installation, Linux, source, build

*************************************************************
Building and installing MIVisionX on Linux from source code
*************************************************************

MIVisionX ``4.0`` and later are built on top of the **ROCm Core SDK** and require **ROCm 7.13 or later** for GPU builds. Before building MIVisionX, install the ROCm Core SDK and the remaining :doc:`prerequisites <./MIVisionX-prerequisites>`. The Core SDK provides the HIP and OpenCL runtimes, the ``amdclang++`` compiler, the ``half`` library, and ``RPP``.

Clone the MIVisionX repository:

.. code:: shell

    git clone https://github.com/ROCm/MIVisionX.git
    cd MIVisionX

HIP backend (default)
=====================

.. code:: shell

    mkdir build-hip && cd build-hip
    cmake ../
    make -j$(nproc)
    sudo make install

OpenCL backend
==============

.. code:: shell

    mkdir build-ocl && cd build-ocl
    cmake -DBACKEND=OCL ../
    make -j$(nproc)
    sudo make install

.. note::

    If both the HIP and OpenCL backends will be installed on the same system, use ``-DCMAKE_INSTALL_PREFIX`` to install each into a separate directory so they do not overwrite each other. For example: ``cmake -DCMAKE_INSTALL_PREFIX=/opt/mivisionx-ocl/ -DBACKEND=OCL ../``

OpenCL ICD loader setup
-----------------------

On current ROCm releases the OpenCL driver (``libamdocl64.so``) is installed under ``/opt/rocm/lib/opencl/`` but is **not** registered with the system ICD loader (nothing is placed in ``/etc/OpenCL/vendors/``). An OpenCL build then finds no platform and fails at graph verification:

.. code:: text

    ERROR: clGetPlatformIDs(0,0,*) => -1001 (failed)
    ERROR: vxVerifyGraph(graph) failed (-1:VX_FAILURE)

``clinfo`` likewise reports ``Number of platforms  0``.

Until the ROCm packaging registers the vendor ICD by default, point the loader at the AMD driver with a user-local vendor file (no ``sudo`` required). The ``.icd`` file must contain the **absolute** path to the driver:

.. code:: shell

    mkdir -p $HOME/opencl-vendors
    echo '/opt/rocm/lib/opencl/libamdocl64.so' > $HOME/opencl-vendors/amdocl64.icd
    export OCL_ICD_VENDORS=$HOME/opencl-vendors

.. note::

    Use the absolute path to ``libamdocl64.so``; a relative path is not resolved by the ICD loader. If ROCm is installed under a versioned prefix (for example ``/opt/rocm/core-7.13``), adjust the path to match (``/opt/rocm/core-7.13/lib/opencl/libamdocl64.so``).

Confirm the platform is now visible with ``clinfo`` (``Number of platforms  1``, ``AMD Accelerated Parallel Processing``) before running the OpenCL tests.

After installation, MIVisionX files are found under ``/opt/rocm/`` unless ``-DCMAKE_INSTALL_PREFIX`` was specified.

To run tests after building:

.. code:: shell

    make test

CPU-only build
==============

MIVisionX can be built without ROCm or an AMD GPU. Only a C++17 compiler and CMake 3.10 or later are required. Pass ``-DGPU_SUPPORT=OFF`` to CMake:

.. code:: shell

    mkdir build-cpu && cd build-cpu
    cmake -DGPU_SUPPORT=OFF ../
    make -j$(nproc)
    sudo make install
