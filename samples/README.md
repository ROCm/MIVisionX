# Samples

MIVisionX samples using AMD OpenVX&trade; and the RunVX graph executor. The samples below show how to run computer vision graphs efficiently on target hardware.

* [GDF - Graph Description Format Samples](#gdf---graph-description-format)
* [C/C++ Samples for OpenVX](#cc-samples-for-openvx)

## GDF - Graph Description Format

MIVisionX samples using [RunVX](../utilities/runvx/README.md#amd-runvx)

**Note:**

* To run the samples we need to put MIVisionX executables and libraries into the system path

```
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
```

* To get help on RunVX, use `-h` option

```
runvx -h
```

### skintonedetect.gdf

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/skinToneDetect_image.PNG" /></p>

usage:

```
runvx gdf/skintonedetect.gdf
```

### canny.gdf

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/canny_image.PNG" /></p>

usage:

```
runvx gdf/canny.gdf
```

### skintonedetect-LIVE.gdf

Using a live camera

usage:

```
runvx -frames:live gdf/skintonedetect-LIVE.gdf
```

### canny-LIVE.gdf

Using a live camera

usage:

```
runvx -frames:live gdf/canny-LIVE.gdf
```

## C/C++ Samples for OpenVX

MIVisionX samples in C/C++

### Canny

usage:

```
cd c_samples/canny/
cmake .
make
./cannyDetect --image <imageName>
./cannyDetect --live
```
