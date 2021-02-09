#!/bin/bash

############# Edit GDF path and kernel names here #############

GDF_PATH="kernelGDFs/"

KERNEL_GROUP_LIST="ARITHMETIC LOGICAL COLOR STATISTICAL FILTER GEOMETRIC VISION"

GDF_ARITHMETIC_LIST="AbsDiff_U8_U8U8
AbsDiff_S16_S16S16_Sat
Add_U8_U8U8_Wrap
Add_U8_U8U8_Sat
Add_S16_U8U8_Wrap
Add_S16_S16U8_Wrap
Add_S16_S16U8_Sat
Add_S16_S16S16_Wrap
Add_S16_S16S16_Sat
Sub_U8_U8U8_Wrap
Sub_U8_U8U8_Sat
Sub_S16_U8U8_Wrap
Sub_S16_S16U8_Wrap
Sub_S16_S16U8_Sat
Sub_S16_S16S16_Wrap
Sub_S16_S16S16_Sat
Mul_U8_U8U8_Wrap_Trunc
Mul_U8_U8U8_Wrap_Round
Mul_U8_U8U8_Sat_Trunc
Mul_U8_U8U8_Sat_Round
Mul_S16_U8U8_Wrap_Trunc
Mul_S16_U8U8_Wrap_Round
Mul_S16_U8U8_Sat_Trunc
Mul_S16_U8U8_Sat_Round
Mul_S16_S16U8_Wrap_Trunc
Mul_S16_S16U8_Wrap_Round
Mul_S16_S16U8_Sat_Trunc
Mul_S16_S16U8_Sat_Round
Mul_S16_S16S16_Wrap_Trunc
Mul_S16_S16S16_Wrap_Round
Mul_S16_S16S16_Sat_Trunc
Mul_S16_S16S16_Sat_Round
Magnitude_S16_S16S16
Phase_U8_S16S16
WeightedAverage_U8_U8U8"

GDF_LOGICAL_LIST="And_U8_U8U8
And_U8_U8U1
And_U8_U1U8
And_U8_U1U1
And_U1_U8U8
And_U1_U8U1
And_U1_U1U8
And_U1_U1U1
Or_U8_U8U8
Or_U8_U8U1
Or_U8_U1U8
Or_U8_U1U1
Or_U1_U8U8
Or_U1_U8U1
Or_U1_U1U8
Or_U1_U1U1
Xor_U8_U8U8
Xor_U8_U8U1
Xor_U8_U1U8
Xor_U8_U1U1
Xor_U1_U8U8
Xor_U1_U8U1
Xor_U1_U1U8
Xor_U1_U1U1
Not_U8_U8
Not_U1_U8
Not_U8_U1
Not_U1_U1"

GDF_COLOR_LIST="ColorDepth_U8_S16_Wrap
ColorDepth_U8_S16_Sat
ColorDepth_S16_U8
ChannelExtract_U8_U16_Pos0
ChannelExtract_U8_U24_Pos0
ChannelExtract_U8_U24_Pos1
ChannelExtract_U8_U24_Pos2
ChannelExtract_U8_U32_Pos0
ChannelExtract_U8_U32_Pos1
ChannelExtract_U8_U32_Pos2
ChannelExtract_U8_U32_Pos3
ChannelCombine_U32_U8U8U8_UYVY
ChannelCombine_U32_U8U8U8_YUYV
ChannelCombine_U24_U8U8U8_RGB
ChannelCombine_U32_U8U8U8U8_RGBX
ColorConvert_RGB_RGBX
ColorConvert_RGB_UYVY
ColorConvert_RGB_YUYV
ColorConvert_RGB_IYUV
ColorConvert_RGB_NV12
ColorConvert_RGB_NV21
ColorConvert_RGBX_RGB
ColorConvert_RGBX_UYVY
ColorConvert_RGBX_YUYV
ColorConvert_RGBX_IYUV
ColorConvert_RGB_NV12
ColorConvert_RGB_NV21
ColorConvert_IYUV_RGB
ColorConvert_IYUV_RGBX
FormatConvert_IYUV_UYVY
FormatConvert_IYUV_YUYV
ColorConvert_NV12_RGB
ColorConvert_NV12_RGBX
FormatConvert_NV12_UYVY
FormatConvert_NV12_YUYV
ColorConvert_YUV4_RGB
ColorConvert_YUV4_RGBX
"

GDF_STATISTICAL_LIST="Threshold_U8_U8_Binary
Threshold_U8_U8_Range
"

GDF_FILTER_LIST="Box_U8_U8_3x3
Dilate_U8_U8_3x3
Dilate_U1_U8_3x3
Dilate_U8_U1_3x3
Dilate_U1_U1_3x3
Erode_U8_U8_3x3
Erode_U1_U8_3x3
Erode_U8_U1_3x3
Erode_U1_U1_3x3
Median_U8_U8_3x3
Gaussian_U8_U8_3x3
HalfGaussian_U8_U8_3x3
HalfGaussian_U8_U8_5x5
Convolve_U8_U8_3x3
Convolve_S16_U8_3x3
Sobel_S16S16_U8_3x3_GXY
Sobel_S16_U8_3x3_GX
Sobel_S16_U8_3x3_GY
"

GDF_GEOMETRIC_LIST="ScaleImage_U8_U8_Nearest
ScaleImage_U8_U8_bilinear
ScaleImage_U8_U8_bilinear_replicate
ScaleImage_U8_U8_bilinear_constant
ScaleImage_U8_U8_u8_area
WarpAffine_U8_U8_Nearest
WarpAffine_U8_U8_Nearest_constant
WarpAffine_U8_U8_Nearestbilinear
WarpAffine_U8_U8_Nearest_bilinear_constant
WarpPerspective_U8_U8_Nearest
WarpPerspective_U8_U8_Nearest
WarpPerspective_U8_U8_Nearest_bilinear
WarpPerspective_U8_U8_Nearest_constant
"

GDF_VISION_LIST="Lut_U8_U8
"

AFFINITY_LIST="GPU" # Or it can be AFFINITY_LIST="CPU GPU"

############# Edit GDF path and kernel names here #############







############# Need not edit #############

rm -rvf oclDumps

for AFFINITY in $AFFINITY_LIST;
do
    printf "\n\n---------------------------------------------"
    printf "\nRunning ARITHMETIC GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_ARITHMETIC_LIST;
    do
        printf "\nRunning $GDF...\n"
        unset AMD_OCL_BUILD_OPTIONS_APPEND
        export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        # touch $GDF_PATH/$GDF.gdf
        runvx -frames:1 -affinity:$AFFINITY -dump-profile $GDF_PATH/$GDF.gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning LOGICAL GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_LOGICAL_LIST;
    do
        printf "\nRunning $GDF...\n"
        unset AMD_OCL_BUILD_OPTIONS_APPEND
        export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        # touch $GDF_PATH/$GDF.gdf
        runvx -frames:1 -affinity:$AFFINITY -dump-profile $GDF_PATH/$GDF.gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning COLOR GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_COLOR_LIST;
    do
        printf "\nRunning $GDF...\n"
        unset AMD_OCL_BUILD_OPTIONS_APPEND
        export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        # touch $GDF_PATH/$GDF.gdf
        runvx -frames:1 -affinity:$AFFINITY -dump-profile $GDF_PATH/$GDF.gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning STATISTICAL GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_STATISTICAL_LIST;
    do
        printf "\nRunning $GDF...\n"
        unset AMD_OCL_BUILD_OPTIONS_APPEND
        export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        # touch $GDF_PATH/$GDF.gdf
        runvx -frames:1 -affinity:$AFFINITY -dump-profile $GDF_PATH/$GDF.gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning FILTER GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_FILTER_LIST;
    do
        printf "\nRunning $GDF...\n"
        unset AMD_OCL_BUILD_OPTIONS_APPEND
        export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        # touch $GDF_PATH/$GDF.gdf
        runvx -frames:1 -affinity:$AFFINITY -dump-profile $GDF_PATH/$GDF.gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning GEOMETRIC GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_GEOMETRIC_LIST;
    do
        printf "\nRunning $GDF...\n"
        unset AMD_OCL_BUILD_OPTIONS_APPEND
        export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        # touch $GDF_PATH/$GDF.gdf
        runvx -frames:1 -affinity:$AFFINITY -dump-profile $GDF_PATH/$GDF.gdf
    done

    printf "\n\n---------------------------------------------"
    printf "\nRunning VISION GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF in $GDF_VISION_LIST;
    do
        printf "\nRunning $GDF...\n"
        unset AMD_OCL_BUILD_OPTIONS_APPEND
        export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        # touch $GDF_PATH/$GDF.gdf
        runvx -frames:1 -affinity:$AFFINITY -dump-profile $GDF_PATH/$GDF.gdf
    done
done

mkdir oclDumps
mv $GDF_PATH/agoKernel_* oclDumps

############# Need not edit #############