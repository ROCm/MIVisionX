.. meta::
  :description: MIVisionX model compiler
  :keywords: MIVisionX, ROCm, model compiler, how to

.. _model-compiler-howto:

******************************************************
Using the MIVisionX model compiler and optimizer
******************************************************

The MIVisionX model compiler is used to convert Caffe, ONNX, and NNEF neural network models to an OpenVX-compatible library that can be used by any application that uses OpenVX.

The steps to convert the models is:

1. Convert the pre-trained models to Neural Net Intermediate Representation (NNIR), AMD's internal open format.
2. Optimize the NNIR.
3. Use the NNIR to generate the OPenVX C code.
4. Compile the C code into the ``libannmodule.so`` shared library.

Model compiler examples and samples are available in the `MIVisionX GitHub repository <https://github.com/ROCm/MIVisionX/tree/develop/samples/model_compiler_samples>`_.

You will need to :doc:`install the model compiler <../install/MIVisionX-model-compiler-install>` first before using it.

Convert the pre-trained model to NNIR
=======================================

Use the appropriate Python script to convert a pre-trained model to NNIR.

To convert a pre-trained Caffe model to NNIR, use ``caffe_to_nnir.py``:

.. code:: shell
    
    python3 caffe_to_nnir.py CAFFE_MODEL \ 
                NNIR_OUTPUT_DIR \ 
                --input-dims DIMENSIONS \
                [--verbose {0|1}; default: 0] \
                [--node_type_append {0|1}; default: 0]

| ``CAFFE_MODEL``: the input Caffe model.
| ``NNIR_OUTPUT_DIR``: the directory that the NNIR model will be written to.
| ``--input-dims``: the dimensions of the model. DIMENSIONS must be in the ``n,c,h,w`` format.
| ``--verbose``: the verbosity of the output. Set it to ``1`` for verbose output.
| ``--node_type_append``: appends the node type name to output names when set to ``1``.

To convert a pre-trained ONNX model to NNIR, use ``onnx_to_nnir.py``:

.. code:: shell
    
    python3 onnx_to_nnir.py ONNX_MODEL \ 
                NNIR_OUTPUT_DIR \
                [--input-dims DIMENSIONS] \
                [--node_type_append {0|1}; default: 0]

| ``ONNX_MODEL``: the input ONNX model.
| ``NNIR_OUTPUT_DIR``: the directory that the NNIR model will be written to.
| ``--input-dims``: the dimensions of the model. DIMENSIONS must be in the ``n,c,h,w`` format.
| ``--node_type_append``: appends the node type name to output names when set to ``1``.

To convert a pre-trained NNEF model to NNIR, use ``nnef_to_nnir.py``:

.. code:: shell
    
    python3 nnef_to_nnir.py NNEF_INPUT_DIR \
                NNIR_OUTPUT_DIR \
                [--node_type_append {0|1}; default: 0]

| ``NNEF_INPUT_DIR``: the NNEF model directory.
| ``NNIR_OUTPUT_DIR``: the directory that the NNIR model will be written to.
| ``--node_type_append``: appends the node type name to output names when set to ``1``.

Apply optimizations
======================

Optimizations are applied to the NNIR model using the ``nnir_update.py`` Python script.

.. code:: shell

    python3 nnir_update.py [--batch-size N] [--fuse-ops 1] \
                [--convert-fp16 1] [--slice-groups 1] \
                NNIR_MODEL_DIR OPTIMIZED_NNIR_MODEL_DIR

| ``--batch-size``: updates the batch size to N.
| ``--fuse-ops``: fuses operations when set to 1.
| ``--convert-fp16``: quantizes the model to fp16 when set to 1.
| ``--slice-groups``: uses slice and concat operations to work around groups when set to 1.
| ``NNIR_MODEL_DIR``: the input NNIR model directory.
| ``OPTIMIZED_NNIR_MODEL_DIR``: the directory where the optimized NNIR is written to.

Convert NNIR to OpenVX C code
==============================

A script is used to convert the NNIR model to OpenVX C code. The C code is then compiled into an OpenVX model.

Use the ``nnir_to_openvx.py`` Python script to convert the NNIR model to OpenVX. 

.. code:: shell
    
    python3 nnir_to_openvx.py \
            [--argmax {UINT8|UINT16|RGB_LUT_FILE|RGBA_LUT_FILE}] \
            [--help] NNIR_INPUT_DIR OUTPUT_DIR

| ``NNIR_INPUT_DIR``: the NNIR input directory.
| ``OUTPUT_DIR``: the output directory.
| ``--argmax``: adds an argmax to the end of the OpenVX model. 
| ``--help``: prints the help.

``--argmax`` can take one of the following:

| ``UINT8``: adds an 8 bit argmax.
| ``UINT6``: adds a 16 bit argmax.
| RGB color mapping look-up table (LUT). The LUT file name must be of the form ``PREFIXrgb.txt``. For example, ``MyLUTrgb.txt``.
| RGBA color mapping look-up table (LUT). The LUT file name must be of the form ``PREFIXrgba.txt``. For example, ``MyLUTrgba.txt``.

The RGB and RGBA LUT must a text file with one 8 bit RGB or RGBA value per label. For example for RGB:

.. code::

    R0 G0 B0
    R1 G1 B1
    R2 G2 B2

For example for RGBA:

.. code::

    R0 G0 B0 A0
    R1 G1 B1 A1
    R2 G2 B2 A2

Compile the code into the ``libannmodule.so`` library
=======================================================

After running ``nnir_to_openvx.py``, change directory to the output directory. Create a build directory. For example:

.. code:: shell

    python3 nnir_to_openvx.py nnirInputFolderFused openvxCodeFolder
    cd openvxCodeFolder
    mkdir build

Use cmake to generate a makefile, then compile the OpenVX code:

.. code:: shell

    cmake ..
    make

This will create ``libannmodule.so`` and the ``anntest`` application for testing inference.