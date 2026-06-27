# GDF - Graph Description Format Samples

MIVisionX GDF samples using [RunVX](../../utilities/runvx/README.md#amd-runvx). Before running, add MIVisionX to your environment:

```shell
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
```

Use `runvx -h` for the full command reference.

## skintonedetect.gdf

Detects skin-tone pixels using color thresholding.

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/skinToneDetect_image.PNG" /></p>

```shell
runvx skintonedetect.gdf
```

## canny.gdf

Runs Canny edge detection on a sample image.

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/canny_image.PNG" /></p>

```shell
runvx canny.gdf
```

## Live camera variants

```shell
runvx -frames:live skintonedetect-LIVE.gdf
runvx -frames:live canny-LIVE.gdf
```
