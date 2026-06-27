.. meta::
  :description: MIVisionX prerequisites
  :keywords: MIVisionX, ROCm, installation, prerequisites

******************************************
MIVisionX prerequisites
******************************************

MIVisionX ``4.0`` and later are built on top of the **ROCm Core SDK** and require **ROCm 7.13 or later** for GPU (``HIP``/``OpenCL``) builds. A CPU-only build (``-DGPU_SUPPORT=OFF``) requires neither ROCm nor an AMD GPU.

.. note::

    * **ROCm 7.2.x and below**: install MIVisionX from the prebuilt packages (``mivisionx``, ``mivisionx-dev``, ``mivisionx-test``). See :doc:`./MIVisionX-package-install`.
    * **ROCm 7.13 and later**: build MIVisionX from source on top of the ROCm Core SDK. See :doc:`./MIVisionX-linux-build-and-install`.
    * **CPU-only**: build from source with ``-DGPU_SUPPORT=OFF`` (no ROCm or GPU required).

Install the ROCm Core SDK by following `Install AMD ROCm <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_ for your GPU and operating system. The Core SDK provides the HIP and OpenCL runtimes, the ``amdclang++`` compiler, OpenMP, the ``half`` (float16) library, and ``RPP`` (used by the ``amd_rpp`` extension).

The following example registers the ROCm repository and installs the Core SDK on Ubuntu 24.04 for a Radeon ``gfx1100`` GPU. Adjust the distribution, GPU architecture (``gfx110x``, ``gfx120x``, ``gfx94x``, ...), and ROCm version to match your system:

.. code-block:: shell

    sudo mkdir --parents --mode=0755 /etc/apt/keyrings
    wget https://repo.amd.com/rocm/packages/gpg/rocm.gpg -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
    sudo tee /etc/apt/sources.list.d/rocm.list << EOF
    deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages/ubuntu2404 stable main
    EOF
    sudo apt update
    sudo apt install amdrocm-core-sdk7.13-gfx110x

The ROCm Core SDK installs into ``/opt/rocm``.

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

Beyond the ROCm Core SDK, building MIVisionX from source only needs CMake from your distribution. On Ubuntu:

.. code-block:: shell

    sudo apt install cmake

Use the appropriate package manager (``yum``/``dnf`` or ``zypper``) on RHEL and SLES.

The ROCm Core SDK provides the required runtime and development dependencies:

* HIP and OpenCL runtimes and the ``amdclang++`` compiler
* `The half-precision floating-point library <https://half.sourceforge.net>`_ version 1.12.0 or later
* `RPP <https://rocm.docs.amd.com/projects/rpp/en/latest/>`_ version 3.1.0 or later (required for the ``amd_rpp`` extension; supports the ``CPU`` and ``HIP`` backends)

The following prerequisite is optional:

* `OpenCV <https://docs.opencv.org/4.6.0/index.html>`_ version 3.x or 4.x, only used by ``RunVX`` for image and video display

.. note::

    * ``RPP`` is included with the ROCm Core SDK. On ROCm releases where it is packaged separately, install it from the ROCm repository.
    * On Ubuntu 22.04, ``libstdc++-12-dev`` must also be installed manually.
