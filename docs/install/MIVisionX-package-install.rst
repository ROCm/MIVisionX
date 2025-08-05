.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, package, installation

********************************************************
Install MIVisionX using the Linux package installer
********************************************************

Three MIVisionX packages are available on Linux:

| ``mivisionx``: The MIVisionX runtime package. This is the basic rocAL package that only provides dynamic libraries. It must always be installed.
| ``mivisionx-dev``: The MIVisionX development package. This package installs a full suite of libraries, header files, and samples. This package needs to be installed to use samples.
| ``mivisionx-test``: A test package that provides a CTest to verify the installation. 

All the required prerequisites are installed when the package installation method is used.

.. note::
  
    | The package installation only supports the HIP backend. :doc:`Build and install from source <./MIVisionX-linux-build-and-install>` to use the OpenCL backend. 
    |
    | The FFmpeg and OpenCV dev packages must be installed manually on RHEL and SLES.


Basic installation
========================================

Use the following commands to install only the MIVisionX runtime package:

.. tab-set::
 
  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt install mivisionx

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install mivisionx

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install mivisionx


Complete installation
========================================

Use the following commands to install ``mivisionx``, ``mivisionx-dev``, and ``mivisionx-test``:

.. tab-set::

  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt-get install mivision mivisionx-dev mivisionx-test

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install mivision mivisionx-dev mivisionx-test

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install mivision mivisionx-dev mivisionx-test


Th test package will install the ``ctest`` module to test MIVisionX. Use the following steps to test package install:

.. code-block:: shell

    mkdir mivisionx-test
    cd mivisionx-test
    cmake /opt/rocm/share/mivisionx/test/
    ctest -VV



