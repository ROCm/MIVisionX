# RALI Python Binding

RALI Python Binding allows you to call functions and pass data from Python to RALI C/C++ libraries, 
letting you take advantage of the RALI functionality in both languages. 

rali_pybind.so is a wrapper library that bridge python and C/C++, so that a RALI functionality 
written primarily in C/C++ language can be used effectively in Python.

## Prerequisites
* [RALI C/C++ Library](../rali#prerequisites)
* CMake Version 3.10 or higher
* Python 3.6 
* PIP3 - `sudo apt install python3-pip`

## Install 
Install rali_pybind using the run.sh script
```
sudo ./run.sh
```

## Run Samples

### Run `test.py`

#### Prerequisites

* Install pip packages
````
pip3 install numpy opencv-python torch
````

* Export `RPP` & `RALI` library into PATH
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/rpp/lib/
```
#### Run Test Script

```
python3 example/test.py
```
