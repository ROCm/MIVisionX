### Kernel testing script

The runvxTestAllScript.sh bash script runs runvx for all AMD OpenVX functionalities in OCL/HIP backends.
- It can optionally run the test for all kernels / single kernel.
- It takes a user given width, height to run tests.
- It can optionally generate OpenCL kernel code dumps for all kernels / single kernel.
- It can optionally generate .bin dumps for input/output pixel values, compare OCLvsHIP outputs and report inconsistencies.

Syntax: `./runvxTestAllScript.sh <W> <H> <D> <K> <B> <N>` where:
```
- W     Width of image in pixels
- H     Height of image pixels
- D     Bin dump toggle (1 = True, 0 = False)
- K     OpenCL kernel dump toggle (1 = True, 0 = False)
- B     Clean build toggle (1 = Clean build-make-install OCL/HIP backends and then run test, 0 = Make-install OCL/HIP backends and then run test)
- N     Name of a single kernel to run (from the list of kernels in this script)
```

Examples:
```
./runvxTestAllScript.sh 16 16 1 1 1
./runvxTestAllScript.sh 16 16 1 0 0
./runvxTestAllScript.sh 16 16 0 0 0
./runvxTestAllScript.sh 16 16 1 0 0 Add_U8_U8U8_Wrap
./runvxTestAllScript.sh 16 16 1 0 0 ChannelExtract_U8U8U8U8_U32
```