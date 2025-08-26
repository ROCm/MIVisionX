.. meta::
  :description: MIVisionX building and installing on Linux
  :keywords: MIVisionX, ROCm, installation, Linux, source, build

*************************************************************
Building and installing MIVisionX on Linux from source code
*************************************************************

Before building and installing MIVisionX, ensure that ROCm has been installed with the `AMDGPU installer <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.1/install/install-methods/amdgpu-installer-index.html>`_ and the ``rocm`` usecase:

.. code:: shell

    sudo amdgpu-install --usecase=rocm

The MIVisionX source code is available from `https://github.com/ROCm/MIVisionX <https://github.com/ROCm/MIVisionX>`_. Use the version of MIVisionX that corresponds to the installed version of ROCm.

MIVisionX on Linux supports both the HIP and OpenCL backends. 

MIVisionX is installed in the ROCm installation directory by default. If MIVisionX for both HIP and OpenCL backends will be installed on the system, each version must be installed in its own custom directory and not in the default directory. 

You can choose to use the |setup|_ Python script to install most :doc:`prerequisites <./MIVisionX-prerequisites>`:

.. code:: shell

    python3 MIVisionX-setup.py

To build and install MIVisionX for the HIP backend, create the ``build_hip`` directory under the ``MIVisionX`` root directory. Change directory to ``build_hip``:

.. code:: shell
 
    mkdir build-hip
    cd build-hip

Use ``cmake`` to generate a makefile: 

.. code:: shell
  
    cmake ../

If MIVisionX will be built for both the HIP and OpenCL backends, use the ``-DCMAKE_INSTALL_PREFIX`` CMake directive to set the installation directory. For example:

.. code:: shell

    cmake -DCMAKE_INSTALL_PREFIX=/opt/hip_backend/


Run ``make`` and ``make install`` :

.. code:: shell

    make 
    make install

To build and install MIVisionX for the OpenCL backend, run ``cmake`` with ``-DBACKEND=OPENCL``:

.. code:: shell

  mkdir build-ocl
  cd build-ocl
  cmake -DBACKEND=OPENCL ../
  make
  sudo make install

If MIVisionX is being built for both the HIP and OpenCL backends, use ``-DCMAKE_INSTALL_PREFIX`` to set the installation directory for the OpenCL backend as well.

After installation, the MIVisionX files will be found under ``/opt/rocm/`` unless ``-DCMAKE_INSTALL_PREFIX`` was specified. If ``-DCMAKE_INSTALL_PREFIX`` was specified, the MIVisionX files will be installed under the specified directory.

To make and run tests, use ``make test``.

.. |setup| replace:: ``MIVisionX-setup.py``
.. _setup: https://github.com/ROCm/MIVisionX/blob/develop/MIVisionX-setup.py
