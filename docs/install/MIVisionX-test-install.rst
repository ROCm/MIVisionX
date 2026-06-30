.. meta::
  :description: MIVisionX test installation
  :keywords: MIVisionX, ROCm, test, installation, OpenVX

******************************************
Testing the MIVisionX installation
******************************************

The MIVisionX test suite is available in the `MIVisionX GitHub repository <https://github.com/ROCm/MIVisionX/tree/develop/tests>`_ and covers core OpenVX (API, conformance, GDF, and vision tests) and the AMD RPP extension.

Using ctest
===========

After a source build, run the full test suite from the build directory:

.. code:: shell

    cd build-hip   # or build-ocl / build-cpu
    make test

After installing the ``mivisionx-test`` package, verify the installation using:

.. code:: shell

    mkdir mivisionx-test && cd mivisionx-test
    cmake /opt/rocm/share/mivisionx/test/
    ctest -VV

Verifying with a sample
========================

Use ``RunVX`` to run the Canny edge detection sample as a quick smoke test:

.. code:: shell

    export PATH=$PATH:/opt/rocm/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
    runvx /opt/rocm/share/mivisionx/samples/gdf/canny.gdf

OpenVX conformance tests
========================

To run the full Khronos OpenVX 1.3 conformance test suite against MIVisionX:

.. code:: shell

    python tests/openvx_conformance_tests/runConformanceTests.py --backend_type HOST

See `tests/openvx_conformance_tests/README.md <https://github.com/ROCm/MIVisionX/blob/develop/tests/openvx_conformance_tests/README.md>`_ for all available options including ``--backend_type HIP``, ``--backend_type OCL``, and ``--jobs``.
