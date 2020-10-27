# HIP OPENVX UNITTESTS
## For HOST/HIP one shot testing all functionalities with default inputs:

```
cd samples/hip_samples/hip-openvx
./testAllScript.sh
```

## For HOST/HIP testing all functionalities with manual inputs:

```
cd samples/hip_samples/hip-openvx
mkdir build
cd build
cmake ..
make
./hipvx_sample <case number (1:99)> <width> <height> <gpu=1/cpu=0> <image1 constant pixel value (optional)> <image2 constant pixel value (optional)>
```

To enable/disable input/output parameters printing, edit main.cpp
```
// ------------------------------------------------------------
// Enable/Disable INPUT/OUTPUT parameters printing
#define PRINT_INPUT
#define PRINT_OUTPUT
// ------------------------------------------------------------
```