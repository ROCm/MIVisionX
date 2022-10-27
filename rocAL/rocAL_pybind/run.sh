# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#!/bin/bash

PYTHON_VERSION=`python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";` ##Gets the Python3 version
DEFAULT_PYTHON=$(which python$PYTHON_VERSION) ## Gets the default python
CONDA="conda"
if [ -n "$CONDA_DEFAULT_ENV" ]  || [ -n "$VIRTUAL_ENV" ] || [[ "$DEFAULT_PYTHON" == *"$CONDA"* ]]; then ## Checks if it is in any env then removes packages accordingly
    PYTHON_LIB_PATH=`python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"`
    INSTALL_FILE_PATH_ROCAL="/site-packages/amd_rocal-1.1.0.dist-info/"
    ROCAL_PYTHON_LIB_PATH=$PYTHON_LIB_PATH$INSTALL_FILE_PATH_ROCAL
    sudo rm -r "$ROCAL_PYTHON_LIB_PATH"
else
    sudo rm -r "/usr/local/lib/python$PYTHON_VERSION/dist-packages/amd_rocal-1.1.0.dist-info/"
fi
sudo rm -r ./amd_rocal.egg-info/
sudo rm -r ./build
sudo rm -r ./dist

blue=`tput setaf 4`
reset=`tput sgr0`

if [[ $# -eq 1 ]]; then
  # Either --backend_hip / --backend_ocl can be passed by the user
  echo "${blue}Running setup.py with $1 ${reset}"

  if [[ "$1" == "--backend_ocl" ]] || [[ "$1" == "--backend_hip" ]]; then
    sudo "$DEFAULT_PYTHON" setup.py bdist_wheel $1
    WHEEL_DIR="./dist"
    WHEEL_NAME=""
    for WHEEL_NAME in "$WHEEL_DIR"/*
    do
        echo "Going to install $WHEEL_NAME"
    done
    pip$PYTHON_VERSION install $WHEEL_NAME
  else
    echo
    echo "The run.sh bash script runs the setup.py with OCL / HIP backends"
    echo
    echo "Syntax : ./run.sh --backend_ocl / --backend_hip"
    echo
    exit
  fi

else
  # Default Backend: --backend_hip
  echo "${blue}Running setup.py with --backend_hip ${reset}"
  sudo "$DEFAULT_PYTHON" setup.py bdist_wheel --backend_hip
  WHEEL_DIR="./dist"
  WHEEL_NAME=""
  for WHEEL_NAME in "$WHEEL_DIR"/*
  do
    echo "Going to install $WHEEL_NAME"
  done
  pip$PYTHON_VERSION install $WHEEL_NAME
fi
