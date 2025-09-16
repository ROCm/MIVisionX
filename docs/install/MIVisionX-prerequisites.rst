.. meta::
  :description: MIVisionX prerequisites
  :keywords: MIVisionX, ROCm, installation, prerequisites

******************************************
MIVisionX prerequisites
******************************************

MIVisionX can be used with or without ROCm.

MIVisionX on ROCm requires ROCm running on an `accelerators based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_ installed with the `AMDGPU installer <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.1/install/install-methods/amdgpu-installer-index.html>`_ and the ``rocm`` usecase:

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
                            [--ffmpeg {ON|OFF}; default: ON]
                            [--amd_rpp {ON|OFF}; default: ON]
                            [--neural_net {ON|OFF}; default: ON]
                            [--inference {ON|OFF}; default: ON]
                            [--developer {ON|OFF}; default:OFF]
                            [--reinstall {ON|OFF}; default:OFF]
                            [--backend {HIP|OCL|CPU}]
                            [--rocm_path ROCM_PATH; default: /opt/rocm]

| ``directory``: The user home directory.
| ``opencv``: The OpenCV version to install.
| ``ffmpeg``: Install the required FFMpeg libraries.
| ``amd_rpp``: Install the packages needed to install and use RPP.
| ``neural_net``: Install the packages needed to install and use neural net.
| ``inference``: Install the packages needed to install and use neural net inference.
| ``developer``: Use the developer options.
| ``reinstall``: Remove the previous dependency installations and install new dependencies.
| ``backend``: Specifies the backend to use.
| ``rocm_path``: The ROCm installation path.

.. note::

    libstdc++-12-dev isn't installed by the setup script and must be installed manually on Ubuntu 22.04 only.


The following prerequisites are required and are installed with both the Linux package installer and the setup script:

* `MIOpen <https://rocm.docs.amd.com/projects/MIOpen/en/latest/>`_
* `MIGraphX <https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/>`_
* `RPP <https://rocm.docs.amd.com/projects/rpp/en/latest/>`_
* `The half-precision floating-point library <https://half.sourceforge.net>`_ version 1.12.0 or later
* `Google Protobuf <https://developers.google.com/protocol-buffers>`_ version 3.12.4 or later
* `LMBD Library <http://www.lmdb.tech/doc/>`_
* `TurboJPEG <https://libjpeg-turbo.org/>`_
* `PyBind11 <https://github.com/pybind/pybind11/releases/tag/v2.11.1>`_ version 2.11.1
* `RapidJSON <https://github.com/Tencent/rapidjson>`_
* `OpenCV <https://docs.opencv.org/4.6.0/index.html>`_ version 4.6
* `Python3 <https://www.python.org/>`_
* libavcodec-dev, libavformat-dev, libavutil-dev, libswscale-dev version 4.4.2 or later


.. note::

    libavcodec-dev, libavformat-dev, libavutil-dev, and libswscale-dev are the only `FFmpeg <https://www.ffmpeg.org>`_ libraries required by MIVisionX. They're installed by default with the setup script and by the package installers.


.. |setup| replace:: ``MIVisionX-setup.py``

.. _setup: https://github.com/ROCm/MIVisionX/blob/develop/MIVisionX-setup.py

