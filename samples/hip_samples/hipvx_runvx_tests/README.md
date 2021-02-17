### Kernel testing script

The runvxTestAllScript.sh bash script runs runvx for all AMD OpenVX functionalities in OCL/HIP backends.
It can optionally generate dumps:
- .bin dumps for input/output images for different sizes.
- OpenCL kernel code dumps for all kernels.

Syntax: `./runvxTestAllScript.sh <W> <H> <D> <K>` where:
```
- W     Width of image in pixels
- H     Height of image pixels
- D     Bin dump toggle (1=True, 0=False)
- K     OpenCL kernel dump toggle (1=True, 0=False)
```

Example:
```
./runvxTestAllScript.sh 20 20 1 0
```