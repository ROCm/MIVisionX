#!/bin/bash

############# Edit GDF path and file names #############
GDF_PATH="../../../vision_tests/gdfs"
GDF_FILE_LIST="01_absDiff.gdf 02_accumulate.gdf 05_add.gdf 06_and.gdf 07_box.gdf 09_iyuv_channelCombine.gdf
                09_nv12_channelCombine.gdf 09_rgb_channelCombine.gdf 09_rgbx_channelCombine.gdf 10_iyuv_channelExtract.gdf
                10_rgb_channelExtract.gdf 10_rgbx_channelExtract.gdf 11_colorConvert.gdf 11_iyuv_colorConvert.gdf 11_nv12_colorConvert.gdf
                11_nv21_colorConvert.gdf 11_rgb_colorConvert.gdf 11_rgbx_colorConvert.gdf 11_uyvy_colorConvert.gdf 12_convertDepth.gdf 13_convolve.gdf
                14_dilate.gdf 16_erode.gdf 18_gaussian.gdf 23_magnitude.gdf 25_median.gdf 27_multiply.gdf 28_not.gdf 30_or.gdf 31_phase.gdf 
                34_scaleImage.gdf 35_sobel.gdf 36_subtract.gdf 37_tableLookup.gdf 38_threshold.gdf 39_warpaffine.gdf 40_warpperspective.gdf 41_xor.gdf"
AFFINITY_LIST="CPU GPU"
############# Edit GDF path and file names #############

for AFFINITY in $AFFINITY_LIST;
do
    printf "\n\n---------------------------------------------"
    printf "\nRunning GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF_FILE in $GDF_FILE_LIST;
    do
        
        printf "\n$GDF_FILE..."
        runvx -affinity:$AFFINITY $GDF_PATH/$GDF_FILE
    done
done