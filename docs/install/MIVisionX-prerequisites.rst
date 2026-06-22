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

When building MIVisionX from source on Linux, the |setup|_ Python script can be used to install prerequisites:

.. code-block:: shell

  MIVisionX-setup.py [-h]   [--directory DIRECTORY; default: ~/]
                            [--opencv OpenCV_VERSION; default: 4.6.0]
                            [--developer {ON|OFF}; default:OFF]
                            [--reinstall {ON|OFF}; default:OFF]
                            [--backend {HIP|OCL|CPU}]
                            [--rocm_path ROCM_PATH; default: /opt/rocm]

| ``directory``: The user home directory.
| ``opencv``: The OpenCV version to install (optional, only used by RunVX for image/video display).
| ``developer``: Use the developer options.
| ``reinstall``: Remove the previous dependency installations and install new dependencies.
| ``backend``: Specifies the backend to use.
| ``rocm_path``: The ROCm installation path.

.. note::

    libstdc++-12-dev isn't installed by the setup script and must be installed manually on Ubuntu 22.04 only.


The following prerequisites are required and are installed with both the Linux package installer and the setup script:

* `RPP <https://rocm.docs.amd.com/projects/rpp/en/latest/>`_ version 3.1.0 or later (required for the ``amd_rpp`` extension; supports the ``CPU`` and ``HIP`` backends)
* `The half-precision floating-point library <https://half.sourceforge.net>`_ version 1.12.0 or later
* `Python3 <https://www.python.org/>`_

The following prerequisite is optional:

* `OpenCV <https://docs.opencv.org/4.6.0/index.html>`_ version 3.x or 4.x, only used by ``RunVX`` for image and video display


.. |setup| replace:: ``MIVisionX-setup.py``

.. _setup: https://github.com/ROCm/MIVisionX/blob/develop/MIVisionX-setup.py
