
.. meta::
  :description: MIVisionX macOS installation
  :keywords: MIVisionX, ROCm, installation, macOS, Apple


*************************************************************
Building and installing MIVisionX on macOS from source code
*************************************************************

.. note::

    macOS only supports the MIVisionX CPU backend

MIVisionX on macOS is built from the source code. The MIVisionX source code is available from `https://github.com/ROCm/MIVisionX <https://github.com/ROCm/MIVisionX>`_. 

Building MIVisionX on macOS requires Homebrew, OpenCV, and OpenSSL:

.. code:: shell

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install cmake opencv openssl



Create the ``build`` directory under the ``MIVisionX`` root directory. Change directory to ``build``:

Use ``cmake`` to generate a makefile: 

.. code:: shell

    cmake ../

Run ``make`` and ``make install`` :

.. code:: shell

    make 
    make install

To make and run tests, use ``make test``. 