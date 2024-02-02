# rocAL Python Binding

rocAL Python Binding allows you to call functions and pass data from Python to rocAL C/C++ libraries,
letting you take advantage of the rocAL functionality in both languages.

rocal_pybind.so is a wrapper library that bridge python and C/C++, so that a rocAL functionality
written primarily in C/C++ language can be used effectively in Python.

## Prerequisites

* [rocAL C/C++ Library](../rocAL/README.md#prerequisites)
* CMake Version `3.5` or higher
* Python Version `3`
* PIP3
* PIP3 Packages - `numpy`, `opencv-python`, `torch`
* [CuPy for rocm](https://github.com/ROCmSoftwarePlatform/cupy)

## rocal_pybind install

rocAL_pybind installs during [MIVisionX build](https://github.com/ROCm/MIVisionX#build--install-mivisionx)

### Prerequisites install to run test scripts

* Install PIP3
  * Ubuntu 20/22

    ```shell
    sudo apt install python3-pip
    ```

* Install pip packages

````shell
pip3 install numpy opencv-python torch
````

* Install `CuPy` for `ROCm` - `https://github.com/ROCmSoftwarePlatform/cupy`

#### Run Test Scripts

* Test scripts and instructions to run them can be found [here](./examples/README.md)
