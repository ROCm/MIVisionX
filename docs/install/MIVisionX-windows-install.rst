.. meta::
  :description: MIVisionX Windows installation
  :keywords: MIVisionX, ROCm, installation, Windows, Microsoft


******************************************
Install MIVisionX on Windows
******************************************

.. note::

    The HIP backend is not supported on Windows.

To install MIVisionX on Windows, you will need:

* `Windows SDK <https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/>`_
* A C++17 toolchain (`Visual Studio 2019 or later <https://visualstudio.microsoft.com/>`_)
* `CMake 3.10 or later <https://cmake.org/download/>`_
* `AMD drivers <https://www.amd.com/en/support>`_
* `OpenCL SDK <https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0>`_
* `OpenCV <https://github.com/opencv/opencv/releases>`_ (optional, only used by ``RunVX`` for image/video display)

If OpenCV is installed, set the ``OpenCV_DIR`` environment variable to point to the ``OpenCV/build`` folder and add ``%OpenCV_DIR%\x64\vc14\bin`` or ``%OpenCV_DIR%\x64\vc15\bin`` to your ``$PATH``.

Build with CMake. The legacy Visual Studio ``.sln``/``.vcxproj`` project files have been removed. The default Windows backend is ``OpenCL``; pass ``-DGPU_SUPPORT=OFF`` for a CPU-only build.

.. code-block:: shell

    git clone https://github.com/ROCm/MIVisionX.git
    cd MIVisionX
    cmake -B build -DGPU_SUPPORT=OFF
    cmake --build build --config Release

Use ``RunVX`` to test the build

.. code-block:: shell

    .\runvx.exe MIVisionX\samples\gdf\skintonedetect.gdf

