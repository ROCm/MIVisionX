#!/bin/bash

PYTHON_VERSION=`python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";` ##Gets the Python3 version
DEFAULT_PYTHON=$(which python$PYTHON_VERSION) ## Gets the default python
CONDA="conda"
if [ -n "$CONDA_DEFAULT_ENV" ]  || [ -n "$VIRTUAL_ENV" ] || [[ "$DEFAULT_PYTHON" == *"$CONDA"* ]]; then ## Checks if it is in any env then removes packages accordingly
    PYTHON_LIB_PATH=${DEFAULT_PYTHON/bin/lib}
    EGG_FILE_PATH="/site-packages/amd_rali-1.1.0-py$PYTHON_VERSION-linux-x86_64.egg"
    ROCAL_PYTHON_LIB_PATH=$PYTHON_LIB_PATH$EGG_FILE_PATH
    sudo rm -r "$ROCAL_PYTHON_LIB_PATH"
else
  sudo rm -r "/usr/local/lib/python$PYTHON_VERSION/dist-packages/amd_rali-1.1.0-py$PYTHON_VERSION-linux-x86_64.egg"
fi
sudo rm -r ./amd_rali.egg-info/
sudo rm -r ./build
sudo rm -r ./dist

blue=`tput setaf 4`
reset=`tput sgr0`

if [[ $# -eq 1 ]]; then
  # Either --backend_hip / --backend_ocl can be passed by the user
  echo "${blue}Running setup.py with $1 ${reset}"

  if [[ "$1" == "--backend_ocl" ]] || [[ "$1" == "--backend_hip" ]]; then
    sudo "$DEFAULT_PYTHON" setup.py build $1
    sudo "$DEFAULT_PYTHON" setup.py install $1
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
  sudo "$DEFAULT_PYTHON" setup.py build --backend_hip
  sudo "$DEFAULT_PYTHON" setup.py install --backend_hip
fi
