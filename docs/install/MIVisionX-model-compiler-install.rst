.. meta::
  :description: MIVisionX model compiler installation and configurations
  :keywords: MIVisionX, ROCm, model compiler, installation

.. _model-compiler-install:

*********************************************************
MIVisionX model compiler installation and configuration
*********************************************************

The MIVisionX model compiler is used to convert pre-trained Caffe, ONNX, and NNEF neural network models to MIVisionX runtime code that can be used by any application that uses OpenVX. For more information on how to use the model compiler, see :ref:`model-compiler-howto`.

Use the following commands to install the MIVisionX model compiler:

.. tab-set::
 
  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt-get -y install python3 python3-pip protobuf-compiler libprotoc-dev


  .. tab-item:: RHEL/CentOS

    .. code:: shell

      sudo yum -y python3-devel python3-pip protobuf python3-protobuf

        
Install additional packages:

.. code:: shell
    
    sudo pip3 install future==1.0.0 pytz==2022.1 numpy==1.23.0


The model compiler files will be saved in ``/opt/rocm/libexec/mivisionx/model_compiler/python/``.

To use the model compiler with Caffe models, install ``google`` and ``protobuff``:

.. code:: shell

    sudo pip3 install google==3.0.0 protobuf==3.12.4

ONNX Models are available at `ONNX Model Zoo <https://github.com/onnx/models>`_

To use the model compiler with NNEF models, you will need to build the NNEF Python module using the `nnef-parser <https://github.com/KhronosGroup/NNEF-Tools>`_:

  .. code:: shell

    git clone -b nnef-v1.0.0 https://github.com/KhronosGroup/NNEF-Tools.git
    cd NNEF-Tools/parser/cpp
    mkdir -p build && cd build
    cmake ../
    make
    sudo make install
    cd ../../../python
    sudo python3 setup.py install

NNEF models are available at `NNEF Model Zoo <https://github.com/KhronosGroup/NNEF-Tools/tree/master/models#nnef-model-zoo>`_

