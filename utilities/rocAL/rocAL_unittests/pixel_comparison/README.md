# Rocal Unit Test

This python script can be used to verify the correctness of outputs of the API offered by rocAL.

## Pre-requisites
- Python
- Ubuntu Linux, version 16.04 or later
- rocAL library (Part of the MIVisionX toolkit)
- OpenCV 3.4+
- Radeon Performance Primitives (RPP)

## Steps to follow

1. Run rocal unit test, dump the outputs and comparing output

Input data : [MIVisionX-data](https://github.com/fiona-gladwin/MIVisionX-data/tree/bb507be1d877a9d899be24540105a1db40a4e1ee)
export ROCAL_DATA_PATH

```
./testAllScripts.sh <device_type 0/1/2> <rgb 0/1/2>
```