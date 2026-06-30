.. meta::
  :description: MIVisionX macOS installation
  :keywords: MIVisionX, ROCm, installation, macOS, Apple


*************************************************************
Building and installing MIVisionX on macOS from source code
*************************************************************

.. note::

    macOS supports the MIVisionX CPU backend only.

Prerequisites
=============

Install Homebrew, then use it to install the required dependencies:

.. code:: shell

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install cmake opencv openssl

Build
=====

Clone the MIVisionX repository and build with CMake:

.. code:: shell

    git clone https://github.com/ROCm/MIVisionX.git
    cd MIVisionX
    mkdir build && cd build
    cmake -DGPU_SUPPORT=OFF ../
    make -j$(nproc)
    sudo make install

To run the test suite after building:

.. code:: shell

    make test
