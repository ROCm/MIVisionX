#!/bin/bash

DEFAULT_PYTHON=$(which python3.6) ## Gets the default python
CONDA="conda"
if [ -n "$CONDA_DEFAULT_ENV" ]  || [ -n "$VIRTUAL_ENV" ] || [[ "$DEFAULT_PYTHON" == *"$CONDA"* ]]; then ## Checks if it is in any env then removes packages accordingly
    PYTHON_LIB_PATH=${DEFAULT_PYTHON/bin/lib}
    EGG_FILE_PATH="/site-packages/amd_rali-1.1.0-py3.6-linux-x86_64.egg"
    ROCAL_PYTHON_LIB_PATH=$PYTHON_LIB_PATH$EGG_FILE_PATH
    sudo rm -r "$ROCAL_PYTHON_LIB_PATH"
else
  sudo rm -r /usr/local/lib/python3.6/dist-packages/amd_rali-1.1.0-py3.6-linux-x86_64.egg
fi
sudo rm -r ./amd_rali.egg-info/
sudo rm -r ./build
sudo rm -r ./dist
sudo "$DEFAULT_PYTHON" setup.py build
sudo "$DEFAULT_PYTHON" setup.py install