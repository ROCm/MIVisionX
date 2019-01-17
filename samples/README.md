# Samples

MIVisionX samples using OpenVX and OpenVX extension libraries

## GDF - Graph Description Format

MIVisionX samples using runvx with GDF

### skintonedetect.gdf

usage:

````
runvx skintonedetect.gdf
````
### canny.gdf

usage:

````
runvx canny.gdf
````
### skintonedetect-LIVE.gdf
Using live camera

usage:

````
runvx -frames:live skintonedetect-LIVE.gdf
````
### canny-LIVE.gdf
Using live camera

usage:

````
runvx -frames:live canny-LIVE.gdf
````
### OpenCV_orb-LIVE.gdf
Using live camera

usage:

````
runvx -frames:live OpenCV_orb-LIVE.gdf
````

## C_Samples - Sample codes in C language

MIVisionX samples using cmake and c code

### canny.cpp

usage:

````
cd c_samples/canny/
cmake .
make

Usage : ./cannyDetect <imageName>
````
