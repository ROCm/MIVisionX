# Samples

MIVisionX samples using OpenVX and OpenVX extensions. In the samples below we will learn how to run computer vision, inference, and a combination of computer vision & inference efficiently on target hardware.

* [C/C++ Samples for OpenVX and OpenVX Extensions](#cc-samples-for-openvx-and-openvx-extensions)
* [GDF - Graph Description Format Samples](#gdf---graph-description-format)

## GDF - Graph Description Format

MIVisionX samples using [RunVX](../utilities/runvx#amd-runvx)

**Note:** 
* To run the samples we need to put MIVisionX executables and libraries into the system path

````
export PATH=$PATH:/opt/rocm/mivisionx_lite/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx_lite/lib
````
* To get help on RunVX, use `-h` option

````
runvx -h
````

### skintonedetect.gdf

<p align="center"><img width="90%" src="images/skinToneDetect_image.PNG" /></p>

usage:

````
runvx gdf/skintonedetect.gdf
````
### canny.gdf

<p align="center"><img width="90%" src="images/canny_image.PNG" /></p>

usage:

````
runvx gdf/canny.gdf
````
### skintonedetect-LIVE.gdf
Using a live camera

usage:

````
runvx -frames:live gdf/skintonedetect-LIVE.gdf
````
### canny-LIVE.gdf
Using a live camera

usage:

````
runvx -frames:live gdf/canny-LIVE.gdf
````

## C/C++ Samples for OpenVX 

MIVisionX Lite samples in C/C++

### Canny
usage:

````
cd c_samples/canny/
cmake .
make
./cannyDetect --image <imageName> 
./cannyDetect --live
````
