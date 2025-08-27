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
* `Visual Studio 2019 or later <https://visualstudio.microsoft.com/>`_
* `AMD drivers <https://www.amd.com/en/support>`_
* `OpenCL SDK <https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0>`_
* `OpenCV 4.6.0 <https://github.com/opencv/opencv/releases/tag/4.6.0>`_

Set the ``OpenCV_DIR`` environment variable to point to the ``OpenCV/build`` folder.

Add ``%OpenCV_DIR%\x64\vc14\bin`` and ``%OpenCV_DIR%\x64\vc15\bin`` to your ``$PATH``.

Build ``MIVisionX.sln`` in Visual Studio for the x64 platform. The MIVisionX libraries and executables will be saved to ``MIVisionX/x64`` folder.

Use ``RunVX`` to test the build

.. code-block:: shell

    ./runvx.exe MIVisionX/samples/gdf/skintonedetect.gdf

