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
