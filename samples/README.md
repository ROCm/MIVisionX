# Samples

MIVisionX samples using OpenVX and OpenVX extension libraries

## GDF - Graph Description Format

MIVisionX samples using [RunVX](../utilities/runvx#amd-runvx)

**Note:** 
* To run the samples we need to put MIVisionX executables and libraries into the system path

````
export PATH=$PATH:/opt/rocm/mivisionx/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib
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
Using live camera

usage:

````
runvx -frames:live gdf/skintonedetect-LIVE.gdf
````
### canny-LIVE.gdf
Using live camera

usage:

````
runvx -frames:live gdf/canny-LIVE.gdf
````
### OpenCV_orb-LIVE.gdf
Using live camera

usage:

````
runvx -frames:live gdf/OpenCV_orb-LIVE.gdf
````

## C / C++ Samples

MIVisionX samples IN C / C++

### Canny

usage:

````
cmake .
make
./cannyDetect --image <imageName> 
./cannyDetect --live
````
### Orb Detect
usage:

````
cmake .
make
./orbDetect
````
## Loom 360 Stitch - Radeon Loom 360 Stitch Samples

MIVisionX samples using [LoomShell](../utilities/loom_shell#radeon-loomshell)

[![Loom Stitch](../docs/images/loom-4.png)](https://youtu.be/E8pPU04iZjw)

**Note:** 
* To run the samples we need to put MIVisionX executables and libraries into the system path

````
export PATH=$PATH:/opt/rocm/mivisionx/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib
````
* To get help on loom_shell, use `-help` option

````
loom_shell -help
````
### Sample - 1

usage:

* Get Data for the stitch
````
cd loom_360_stitch/sample-1/
python loomStitch-sample1-get-data.py
````
* Run Loom Shell Script to generate the 360 Image
````
loom_shell loomStitch-sample1.txt
````

### Sample - 2

usage:

* Get Data for the stitch
````
cd loom_360_stitch/sample-2/
python loomStitch-sample2-get-data.py
````
* Run Loom Shell Script to generate the 360 Image
````
loom_shell loomStitch-sample2.txt
````

### Sample - 3

usage:

* Get Data for the stitch
````
cd loom_360_stitch/sample-3/
python loomStitch-sample3-get-data.py
````
* Run Loom Shell Script to generate the 360 Image
````
loom_shell loomStitch-sample3.txt
````
