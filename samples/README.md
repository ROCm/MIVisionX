# Samples

MIVisionX samples demonstrating AMD OpenVX&trade; and the RunVX graph executor.

* [GDF samples](#gdf---graph-description-format) — OpenVX graphs run with RunVX
* [C/C++ samples](#cc-samples-for-openvx) — OpenVX API usage in C/C++

## Setup

Add MIVisionX to your environment before running any sample:

```shell
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
```

## GDF - Graph Description Format

MIVisionX GDF samples using [RunVX](../utilities/runvx/README.md#amd-runvx). Run from the `samples/gdf/` directory.

### skintonedetect.gdf

Detects skin-tone pixels in an image using color thresholding.

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/skinToneDetect_image.PNG" /></p>

```shell
runvx gdf/skintonedetect.gdf
```

### canny.gdf

Runs Canny edge detection on a sample image.

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/canny_image.PNG" /></p>

```shell
runvx gdf/canny.gdf
```

### Live camera variants

```shell
runvx -frames:live gdf/skintonedetect-LIVE.gdf
runvx -frames:live gdf/canny-LIVE.gdf
```

## C/C++ Samples for OpenVX

### Canny Edge Detector

```shell
cd c_samples/canny/
cmake .
make

./cannyDetect --image <path/to/image>
./cannyDetect --live
```
