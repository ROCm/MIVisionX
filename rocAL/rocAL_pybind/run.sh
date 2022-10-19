#!/bin/bash

PYTHON_VERSION=`python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";` ##Gets the Python3 version
DEFAULT_PYTHON=$(which python$PYTHON_VERSION) ## Gets the default python

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
    python$PYTHON_VERSION -m pip uninstall $WHEEL_NAME
    python$PYTHON_VERSION -m pip install $WHEEL_NAME
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
  python$PYTHON_VERSION -m pip uninstall $WHEEL_NAME
  python$PYTHON_VERSION -m pip install $WHEEL_NAME
fi