# C/C++ Samples for OpenVX

MIVisionX C/C++ sample applications demonstrating the OpenVX API.

## Canny Edge Detector

Runs Canny edge detection on an image file or live camera using an OpenVX graph.

```shell
cd c_samples/canny/
cmake .
make

# On an image file
./cannyDetect --image <path/to/image>

# On a live camera
./cannyDetect --live
```
