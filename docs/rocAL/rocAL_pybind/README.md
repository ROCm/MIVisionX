# rocAL Python Binding

rocAL Python Binding allows you to call functions and pass data from Python to rocAL C/C++ libraries,
letting you take advantage of the rocAL functionality in both languages.

rocal_pybind.so is a wrapper library that bridge python and C/C++, so that a rocAL functionality
written primarily in C/C++ language can be used effectively in Python.

## Prerequisites
* [rocAL C/C++ Library](../rocAL#prerequisites)
* CMake Version 3.10 or higher
* Python 3
* PIP3 - `sudo apt install python3-pip`

## Install
Install rocAL_pybind using the run.sh script
```
./run.sh
```

#### Prerequisites

* Install pip packages
````
pip3 install numpy opencv-python torch
````

#### Run Test Scripts
* Test scripts and instructions to run them can be found [here](examples/)
