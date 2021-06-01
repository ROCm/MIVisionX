# rocAL Python Binding

rocAL Python Binding allows you to call functions and pass data from Python to rocAL C/C++ libraries, 
letting you take advantage of the rocAL functionality in both languages. 

rali_pybind.so is a wrapper library that bridge python and C/C++, so that a rocAL functionality 
written primarily in C/C++ language can be used effectively in Python.

## Prerequisites
* [rocAL C/C++ Library](../rocAL#prerequisites)
* CMake Version 3.10 or higher
* Python 3.6 
* PIP3 - `sudo apt install python3-pip`

## Install 
Install rali_pybind using the run.sh script (for all except conda environment)
```
sudo ./run.sh
```
NOTE: If using conda environment, use:
```
python3.6 setup.py build
python3.6 setup.py install
```

## Run Samples

### Run `test.py`

#### Prerequisites

* Install pip packages
````
pip3 install numpy opencv-python torch
````

* Export `RPP` & `rocAL` library into PATH
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/rpp/lib/
```
#### Run Test Script

test.py [image_folder] [cpu:0/gpu:1] [batch_size]

```
python3 example/test.py ../../data/images/AMD-tinyDataSet/ 0 4
```
