.. meta::
  :description: MIVisionX Windows installation
  :keywords: MIVisionX, ROCm, installation, Windows, Microsoft


******************************************
Install MIVisionX on Windows
******************************************

.. note::

    The HIP backend is not supported on Windows. The default Windows backend is ``OpenCL``.

Prerequisites
=============

* `Windows SDK <https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/>`_ and a C++17 toolchain (`Visual Studio 2019 or later <https://visualstudio.microsoft.com/>`_)
* `CMake 3.10 or later <https://cmake.org/download/>`_
* `AMD drivers <https://www.amd.com/en/support>`_
* `OpenCL SDK <https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0>`_ (for the OpenCL backend)
* `OpenCV <https://github.com/opencv/opencv/releases>`_ (optional — only used by ``RunVX`` for image and video display)

If OpenCV is installed, set the ``OpenCV_DIR`` environment variable to the ``OpenCV/build`` folder and add ``%OpenCV_DIR%\x64\vc14\bin`` or ``%OpenCV_DIR%\x64\vc15\bin`` to ``PATH``.

Build
=====

The legacy Visual Studio ``.sln``/``.vcxproj`` project files have been removed. Build with CMake:

.. code-block:: shell

    git clone https://github.com/ROCm/MIVisionX.git
    cd MIVisionX

    # OpenCL backend (default — requires the OpenCL SDK)
    cmake -B build
    cmake --build build --config Release

    # CPU-only (no GPU or OpenCL SDK required)
    cmake -B build -DGPU_SUPPORT=OFF
    cmake --build build --config Release

You can open the CMake-generated solution in the ``build`` folder with Visual Studio, or build entirely from the command line as shown above.

Verify the build
================

Use ``RunVX`` to run a sample graph:

.. code-block:: shell

    .\build\bin\Release\runvx.exe ADD_PATH_TO\MIVisionX\samples\gdf\skintonedetect.gdf
