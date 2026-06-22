.. meta::
  :description: MIVisionX prerequisites
  :keywords: MIVisionX, ROCm, installation, prerequisites

******************************************
MIVisionX prerequisites
******************************************

MIVisionX can be used with or without ROCm.

MIVisionX on ROCm requires ROCm running on an `GPUs based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_ installed with the AMDGPU installer and the ``rocm`` usecase:

.. code:: shell

    sudo amdgpu-install --usecase=rocm

MIVisionX has been tested on the following Linux environments:
  
* Ubuntu 22.04 or 24.04
* RHEL 8 or 9
* SLES 15 SP7

See `Supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_ for the complete list of ROCm supported Linux environments.

MIVisionX can also be installed on the following operating systems:

* Microsoft Windows 10 or 11
* macOS 13 Ventura and later

Building MIVisionX from source on Linux requires CMake Version 3.10 or later, AMD Clang++ Version 18.0.0 or later, and the following compiler support:

* C++17
* OpenMP
* Threads

When building MIVisionX from source on Linux, install the prerequisites with your package manager. On Ubuntu:

.. code-block:: shell

    sudo apt install cmake hip-dev openmp-extras-dev half rpp-dev pkg-config

Use the appropriate package manager (``yum``/``dnf`` or ``zypper``) and equivalent ``-devel`` package names on RHEL and SLES.

The following prerequisites are required and are also installed by the Linux package installer:

* `RPP <https://rocm.docs.amd.com/projects/rpp/en/latest/>`_ version 3.1.0 or later (required for the ``amd_rpp`` extension; supports the ``CPU`` and ``HIP`` backends)
* `The half-precision floating-point library <https://half.sourceforge.net>`_ version 1.12.0 or later

The following prerequisite is optional:

* `OpenCV <https://docs.opencv.org/4.6.0/index.html>`_ version 3.x or 4.x, only used by ``RunVX`` for image and video display

.. note::

    On Ubuntu 22.04, ``libstdc++-12-dev`` must also be installed manually.
