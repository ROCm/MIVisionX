#!/bin/bash

############# Help #############
# The runvxTestAllScript.sh bash script runs runvx for all AMD OpenVX functionalities in OCL/HIP backends.
# It can optionally generate dumps:
#     - .bin dumps for input/output images for different sizes.
#     - OpenCL kernel code dumps for all kernels.
# It can compare diff between OpenCL and HIP bin dumps and flag any inconsistencies in outputs.
############# Help #############





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
Sub_S16_U8S16_Wrap
Sub_S16_U8S16_Sat
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
ChannelExtract_U8_U16_Pos1
ChannelExtract_U8_U24_Pos0
ChannelExtract_U8_U24_Pos1
ChannelExtract_U8_U24_Pos2
ChannelExtract_U8_U32_Pos0
ChannelExtract_U8_U32_Pos1
ChannelExtract_U8_U32_Pos2
ChannelExtract_U8_U32_Pos3
ChannelExtract_U8U8U8_U24
ChannelExtract_U8U8U8_U32
ChannelExtract_U8U8U8U8_U32
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

# Input parameters

if [ "$#" -ne 4 ]; then
    echo
    echo "The runvxTestAllScript.sh bash script runs runvx for all AMD OpenVX functionalities in OCL/HIP backends."
    echo "It can optionally generate dumps:"
    echo "    - .bin dumps for input/output images for different sizes."
    echo "    - OpenCL kernel code dumps for all kernels."
    echo
    echo "Syntax: ./runvxTestAllScript.sh <W> <H> <D> <K>"
    echo "W     Width of image in pixels"
    echo "H     Height of image pixels"
    echo "D     Bin dump toggle (1=True, 0=False)"
    echo "K     OpenCL kernel dump toggle (1=True, 0=False)"
    exit 1
fi

if [ "$3" -ne 0 ]; then
    if [ "$3" -ne 1 ]; then
        echo "The bin dump toggle should be 0 or 1!"
        exit 1
    fi
fi

if [ "$4" -ne 0 ]; then
    if [ "$4" -ne 1 ]; then
        echo "The OpenCL kernel dump toggle should be 0 or 1!"
        exit 1
    fi
fi

WIDTH="$1"
HEIGHT="$2"
DUMP="$3"
KERNEL_DUMP="$4"

HALF_WIDTH=$(expr $WIDTH / 2)
STRING_I1="data input_1"
STRING_I2="data input_2"
STRING_I3="data input_3"
STRING_I4="data input_4"
STRING_O1="data output_1"
STRING_O2="data output_2"
STRING_O3="data output_3"
STRING_O4="data output_4"
cwd=$(pwd)





# generator function to auto-generate gdfs for different image sizes, with/without binary dump

generator() {
    CATEGORY=$1
    if [ "$DUMP" -eq 0 ]; then
        cp $GDF_PATH/$CATEGORY/$GDF.gdf $GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf
    else
        while IFS= read -r LINE || [[ -n "$LINE" ]]; do

            if [[ $LINE == $STRING_I1* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"input_1".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_I2* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"input_2".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_I3* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"input_3".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_I4* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"input_4".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"

            elif [[ $LINE == $STRING_O1* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"output_1".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_O2* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"output_2".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_O3* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"output_3".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            elif [[ $LINE == $STRING_O4* ]]; then
                echo "$LINE":WRITE,agoKernel_"$GDF"_"output_4".bin >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"

            else
                echo "$LINE" >> "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
            fi
        done < "$GDF_PATH/$CATEGORY/$GDF.gdf"
    fi
    sed -i "s/1920,1080/$WIDTH,$HEIGHT/" "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
    sed -i "s/960,1080/$HALF_WIDTH,$HEIGHT/" "$GENERATED_GDF_PATH/$CATEGORY/$GDF.gdf"
}





# case_tester function to test each case in each functionality group

case_tester() {
    for AFFINITY in "$AFFINITY_LIST";
    do
        printf "\n\n---------------------------------------------"
        printf "\nRunning ARITHMETIC GDF cases on runvx for $AFFINITY..."
        printf "\n---------------------------------------------\n"
        for GDF in $GDF_ARITHMETIC_LIST;
        do
            printf "\nRunning $GDF...\n"
            unset AMD_OCL_BUILD_OPTIONS_APPEND
            if [ "$KERNEL_DUMP" -eq 1 ]; then
                export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
            fi
            generator "arithmetic"
            runvx -frames:1 -affinity:$AFFINITY -dump-profile $GENERATED_GDF_PATH/arithmetic/$GDF.gdf
        done

        # printf "\n\n---------------------------------------------"
        # printf "\nRunning LOGICAL GDF cases on runvx for $AFFINITY..."
        # printf "\n---------------------------------------------\n"
        # for GDF in $GDF_LOGICAL_LIST;
        # do
        #     printf "\nRunning $GDF...\n"
        #     unset AMD_OCL_BUILD_OPTIONS_APPEND
        #     if [ "$KERNEL_DUMP" -eq 1 ]; then
        #         export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        #     fi
        #     generator "logical"
        #     runvx -frames:1 -affinity:$AFFINITY -dump-profile $GENERATED_GDF_PATH/logical/$GDF.gdf
        # done

        # printf "\n\n---------------------------------------------"
        # printf "\nRunning COLOR GDF cases on runvx for $AFFINITY..."
        # printf "\n---------------------------------------------\n"
        # for GDF in $GDF_COLOR_LIST;
        # do
        #     printf "\nRunning $GDF...\n"
        #     unset AMD_OCL_BUILD_OPTIONS_APPEND
        #     if [ "$KERNEL_DUMP" -eq 1 ]; then
        #         export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        #     fi
        #     generator "color"
        #     runvx -frames:1 -affinity:$AFFINITY -dump-profile $GENERATED_GDF_PATH/color/$GDF.gdf
        # done

        # printf "\n\n---------------------------------------------"
        # printf "\nRunning STATISTICAL GDF cases on runvx for $AFFINITY..."
        # printf "\n---------------------------------------------\n"
        # for GDF in $GDF_STATISTICAL_LIST;
        # do
        #     printf "\nRunning $GDF...\n"
        #     unset AMD_OCL_BUILD_OPTIONS_APPEND
        #     if [ "$KERNEL_DUMP" -eq 1 ]; then
        #         export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        #     fi
        #     generator "statistical"
        #     runvx -frames:1 -affinity:$AFFINITY -dump-profile $GENERATED_GDF_PATH/statistical/$GDF.gdf
        # done

        # printf "\n\n---------------------------------------------"
        # printf "\nRunning FILTER GDF cases on runvx for $AFFINITY..."
        # printf "\n---------------------------------------------\n"
        # for GDF in $GDF_FILTER_LIST;
        # do
        #     printf "\nRunning $GDF...\n"
        #     unset AMD_OCL_BUILD_OPTIONS_APPEND
        #     if [ "$KERNEL_DUMP" -eq 1 ]; then
        #         export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        #     fi
        #     generator "filter"
        #     runvx -frames:1 -affinity:$AFFINITY -dump-profile $GENERATED_GDF_PATH/filter/$GDF.gdf
        # done

        # printf "\n\n---------------------------------------------"
        # printf "\nRunning GEOMETRIC GDF cases on runvx for $AFFINITY..."
        # printf "\n---------------------------------------------\n"
        # for GDF in $GDF_GEOMETRIC_LIST;
        # do
        #     printf "\nRunning $GDF...\n"
        #     unset AMD_OCL_BUILD_OPTIONS_APPEND
        #     if [ "$KERNEL_DUMP" -eq 1 ]; then
        #         export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        #     fi
        #     generator "geometric"
        #     runvx -frames:1 -affinity:$AFFINITY -dump-profile $GENERATED_GDF_PATH/geometric/$GDF.gdf
        # done

        # printf "\n\n---------------------------------------------"
        # printf "\nRunning VISION GDF cases on runvx for $AFFINITY..."
        # printf "\n---------------------------------------------\n"
        # for GDF in $GDF_VISION_LIST;
        # do
        #     printf "\nRunning $GDF...\n"
        #     unset AMD_OCL_BUILD_OPTIONS_APPEND
        #     if [ "$KERNEL_DUMP" -eq 1 ]; then
        #         export AMD_OCL_BUILD_OPTIONS_APPEND=-save-temps-all=./agoKernel_$GDF
        #     fi
        #     generator "vision"
        #     runvx -frames:1 -affinity:$AFFINITY -dump-profile $GENERATED_GDF_PATH/vision/$GDF.gdf
        # done
    done
}





# Running all kernels for OCL backend

rm -rvf dumpsOCL
rm -rvf generatedGDFsOCL
mkdir generatedGDFsOCL
mkdir generatedGDFsOCL/arithmetic generatedGDFsOCL/logical generatedGDFsOCL/color generatedGDFsOCL/filter generatedGDFsOCL/geometric generatedGDFsOCL/statistical generatedGDFsOCL/vision
GENERATED_GDF_PATH="generatedGDFsOCL/"

# Running OCL - MCW-Dev/MIVISION:hip-porting - not working currently
# cd ../../../
# rm -rvf build_ocl
# mkdir build_ocl
# cd build_ocl
# cmake ..
# sudo make -j20 install
# cd ../samples/hip_samples/hipvx_runvx_tests

# Running OCL - GPUOpen-ProfessionalCompute-Libraries/MIVisionX:master - working - to be removed after MCW-Dev/MIVISION:hip-porting OCL is fixed
cd ../../../../../MIVisionX/    # change path manually to your clone of GPUOpen-ProfessionalCompute-Libraries/MIVisionX
rm -rvf build_ocl
mkdir build_ocl
cd build_ocl
cmake ..
sudo make -j20 install
cd $cwd

case_tester

mkdir dumpsOCL
mkdir dumpsOCL/arithmetic dumpsOCL/logical dumpsOCL/color dumpsOCL/filter dumpsOCL/geometric dumpsOCL/statistical dumpsOCL/vision
mv "$GENERATED_GDF_PATH"/arithmetic/agoKernel_* dumpsOCL/arithmetic
mv "$GENERATED_GDF_PATH"/logical/agoKernel_* dumpsOCL/logical
mv "$GENERATED_GDF_PATH"/color/agoKernel_* dumpsOCL/color
mv "$GENERATED_GDF_PATH"/filter/agoKernel_* dumpsOCL/filter
mv "$GENERATED_GDF_PATH"/geometric/agoKernel_* dumpsOCL/geometric
mv "$GENERATED_GDF_PATH"/statistical/agoKernel_* dumpsOCL/statistical
mv "$GENERATED_GDF_PATH"/vision/agoKernel_* dumpsOCL/vision





# Running all kernels for HIP backend

rm -rvf dumpsHIP
rm -rvf generatedGDFsHIP
mkdir generatedGDFsHIP
mkdir generatedGDFsHIP/arithmetic generatedGDFsHIP/logical generatedGDFsHIP/color generatedGDFsHIP/filter generatedGDFsHIP/geometric generatedGDFsHIP/statistical generatedGDFsHIP/vision
GENERATED_GDF_PATH="generatedGDFsHIP/"

cd ../../../
rm -rvf build_hip
mkdir build_hip
cd build_hip
cmake -DBACKEND=HIP ..
sudo make -j20 install
cd ../samples/hip_samples/hipvx_runvx_tests

case_tester

mkdir dumpsHIP
mkdir dumpsHIP/arithmetic dumpsHIP/logical dumpsHIP/color dumpsHIP/filter dumpsHIP/geometric dumpsHIP/statistical dumpsHIP/vision
mv "$GENERATED_GDF_PATH"/arithmetic/agoKernel_* dumpsHIP/arithmetic
mv "$GENERATED_GDF_PATH"/logical/agoKernel_* dumpsHIP/logical
mv "$GENERATED_GDF_PATH"/color/agoKernel_* dumpsHIP/color
mv "$GENERATED_GDF_PATH"/filter/agoKernel_* dumpsHIP/filter
mv "$GENERATED_GDF_PATH"/geometric/agoKernel_* dumpsHIP/geometric
mv "$GENERATED_GDF_PATH"/statistical/agoKernel_* dumpsHIP/statistical
mv "$GENERATED_GDF_PATH"/vision/agoKernel_* dumpsHIP/vision





# OCLvsHIP output match check - by running a diff check on all OCLvsHIP bin dumps and flag any inconsistencies in outputs

if [ "$DUMP" -eq 1 ]; then
    OUTPUT_BIN_LIST="arithmetic/agoKernel_AbsDiff_U8_U8U8_output_1.bin
    arithmetic/agoKernel_AbsDiff_S16_S16S16_Sat_output_1.bin
    arithmetic/agoKernel_Add_U8_U8U8_Wrap_output_1.bin
    arithmetic/agoKernel_Add_U8_U8U8_Sat_output_1.bin
    arithmetic/agoKernel_Add_S16_U8U8_Wrap_output_1.bin
    arithmetic/agoKernel_Add_S16_S16U8_Wrap_output_1.bin
    arithmetic/agoKernel_Add_S16_S16U8_Sat_output_1.bin
    arithmetic/agoKernel_Add_S16_S16S16_Wrap_output_1.bin
    arithmetic/agoKernel_Add_S16_S16S16_Sat_output_1.bin
    arithmetic/agoKernel_Sub_U8_U8U8_Wrap_output_1.bin
    arithmetic/agoKernel_Sub_U8_U8U8_Sat_output_1.bin
    arithmetic/agoKernel_Sub_S16_U8U8_Wrap_output_1.bin
    arithmetic/agoKernel_Sub_S16_S16U8_Wrap_output_1.bin
    arithmetic/agoKernel_Sub_S16_S16U8_Sat_output_1.bin
    arithmetic/agoKernel_Sub_S16_U8S16_Wrap_output_1.bin
    arithmetic/agoKernel_Sub_S16_U8S16_Sat_output_1.bin
    arithmetic/agoKernel_Sub_S16_S16S16_Wrap_output_1.bin
    arithmetic/agoKernel_Sub_S16_S16S16_Sat_output_1.bin
    arithmetic/agoKernel_Mul_U8_U8U8_Wrap_Trunc_output_1.bin
    arithmetic/agoKernel_Mul_U8_U8U8_Wrap_Round_output_1.bin
    arithmetic/agoKernel_Mul_U8_U8U8_Sat_Trunc_output_1.bin
    arithmetic/agoKernel_Mul_U8_U8U8_Sat_Round_output_1.bin
    arithmetic/agoKernel_Mul_S16_U8U8_Wrap_Trunc_output_1.bin
    arithmetic/agoKernel_Mul_S16_U8U8_Wrap_Round_output_1.bin
    arithmetic/agoKernel_Mul_S16_U8U8_Sat_Trunc_output_1.bin
    arithmetic/agoKernel_Mul_S16_U8U8_Sat_Round_output_1.bin
    arithmetic/agoKernel_Mul_S16_S16U8_Wrap_Trunc_output_1.bin
    arithmetic/agoKernel_Mul_S16_S16U8_Wrap_Round_output_1.bin
    arithmetic/agoKernel_Mul_S16_S16U8_Sat_Trunc_output_1.bin
    arithmetic/agoKernel_Mul_S16_S16U8_Sat_Round_output_1.bin
    arithmetic/agoKernel_Mul_S16_S16S16_Wrap_Trunc_output_1.bin
    arithmetic/agoKernel_Mul_S16_S16S16_Wrap_Round_output_1.bin
    arithmetic/agoKernel_Mul_S16_S16S16_Sat_Trunc_output_1.bin
    arithmetic/agoKernel_Mul_S16_S16S16_Sat_Round_output_1.bin
    arithmetic/agoKernel_Magnitude_S16_S16S16_output_1.bin
    arithmetic/agoKernel_Phase_U8_S16S16_output_1.bin
    arithmetic/agoKernel_WeightedAverage_U8_U8U8_output_1.bin"
    # logical/agoKernel_And_U8_U8U8_output_1.bin
    # logical/agoKernel_And_U8_U8U1_output_1.bin
    # logical/agoKernel_And_U8_U1U8_output_1.bin
    # logical/agoKernel_And_U8_U1U1_output_1.bin
    # logical/agoKernel_And_U1_U8U8_output_1.bin
    # logical/agoKernel_And_U1_U8U1_output_1.bin
    # logical/agoKernel_And_U1_U1U8_output_1.bin
    # logical/agoKernel_And_U1_U1U1_output_1.bin
    # logical/agoKernel_Or_U8_U8U8_output_1.bin
    # logical/agoKernel_Or_U8_U8U1_output_1.bin
    # logical/agoKernel_Or_U8_U1U8_output_1.bin
    # logical/agoKernel_Or_U8_U1U1_output_1.bin
    # logical/agoKernel_Or_U1_U8U8_output_1.bin
    # logical/agoKernel_Or_U1_U8U1_output_1.bin
    # logical/agoKernel_Or_U1_U1U8_output_1.bin
    # logical/agoKernel_Or_U1_U1U1_output_1.bin
    # logical/agoKernel_Xor_U8_U8U8_output_1.bin
    # logical/agoKernel_Xor_U8_U8U1_output_1.bin
    # logical/agoKernel_Xor_U8_U1U8_output_1.bin
    # logical/agoKernel_Xor_U8_U1U1_output_1.bin
    # logical/agoKernel_Xor_U1_U8U8_output_1.bin
    # logical/agoKernel_Xor_U1_U8U1_output_1.bin
    # logical/agoKernel_Xor_U1_U1U8_output_1.bin
    # logical/agoKernel_Xor_U1_U1U1_output_1.bin
    # logical/agoKernel_Not_U8_U8_output_1.bin
    # logical/agoKernel_Not_U1_U8_output_1.bin
    # logical/agoKernel_Not_U8_U1_output_1.bin
    # logical/agoKernel_Not_U1_U1_output_1.bin
    # color/agoKernel_ColorDepth_U8_S16_Wrap_output_1.bin
    # color/agoKernel_ColorDepth_U8_S16_Sat_output_1.bin
    # color/agoKernel_ColorDepth_S16_U8_output_1.bin
    # color/agoKernel_ChannelExtract_U8_U16_Pos0_output_1.bin
    # color/agoKernel_ChannelExtract_U8_U16_Pos1_output_1.bin
    # color/agoKernel_ChannelExtract_U8_U24_Pos0_output_1.bin
    # color/agoKernel_ChannelExtract_U8_U24_Pos1_output_1.bin
    # color/agoKernel_ChannelExtract_U8_U24_Pos2_output_1.bin
    # color/agoKernel_ChannelExtract_U8_U32_Pos0_output_1.bin
    # color/agoKernel_ChannelExtract_U8_U32_Pos1_output_1.bin
    # color/agoKernel_ChannelExtract_U8_U32_Pos2_output_1.bin
    # color/agoKernel_ChannelExtract_U8_U32_Pos3_output_1.bin
    # color/agoKernel_ChannelExtract_U8U8U8_U24_output_1.bin
    # color/agoKernel_ChannelExtract_U8U8U8_U32_output_1.bin
    # color/agoKernel_ChannelExtract_U8U8U8U8_U32_output_1.bin
    # color/agoKernel_ChannelCombine_U32_U8U8U8_UYVY_output_1.bin
    # color/agoKernel_ChannelCombine_U32_U8U8U8_YUYV_output_1.bin
    # color/agoKernel_ChannelCombine_U24_U8U8U8_RGB_output_1.bin
    # color/agoKernel_ChannelCombine_U32_U8U8U8U8_RGBX_output_1.bin
    # color/agoKernel_ColorConvert_RGB_RGBX_output_1.bin
    # color/agoKernel_ColorConvert_RGB_UYVY_output_1.bin
    # color/agoKernel_ColorConvert_RGB_YUYV_output_1.bin
    # color/agoKernel_ColorConvert_RGB_IYUV_output_1.bin
    # color/agoKernel_ColorConvert_RGB_NV12_output_1.bin
    # color/agoKernel_ColorConvert_RGB_NV21_output_1.bin
    # color/agoKernel_ColorConvert_RGBX_RGB_output_1.bin
    # color/agoKernel_ColorConvert_RGBX_UYVY_output_1.bin
    # color/agoKernel_ColorConvert_RGBX_YUYV_output_1.bin
    # color/agoKernel_ColorConvert_RGBX_IYUV_output_1.bin
    # color/agoKernel_ColorConvert_RGB_NV12_output_1.bin
    # color/agoKernel_ColorConvert_RGB_NV21_output_1.bin
    # color/agoKernel_ColorConvert_IYUV_RGB_output_1.bin
    # color/agoKernel_ColorConvert_IYUV_RGBX_output_1.bin
    # color/agoKernel_FormatConvert_IYUV_UYVY_output_1.bin
    # color/agoKernel_FormatConvert_IYUV_YUYV_output_1.bin
    # color/agoKernel_ColorConvert_NV12_RGB_output_1.bin
    # color/agoKernel_ColorConvert_NV12_RGBX_output_1.bin
    # color/agoKernel_FormatConvert_NV12_UYVY_output_1.bin
    # color/agoKernel_FormatConvert_NV12_YUYV_output_1.bin
    # color/agoKernel_ColorConvert_YUV4_RGB_output_1.bin
    # color/agoKernel_ColorConvert_YUV4_RGBX_output_1.bin
    # statistical/agoKernel_Threshold_U8_U8_Binary_output_1.bin
    # statistical/agoKernel_Threshold_U8_U8_Range_output_1.bin
    # filter/agoKernel_Box_U8_U8_3x3_output_1.bin
    # filter/agoKernel_Dilate_U8_U8_3x3_output_1.bin
    # filter/agoKernel_Dilate_U1_U8_3x3_output_1.bin
    # filter/agoKernel_Dilate_U8_U1_3x3_output_1.bin
    # filter/agoKernel_Dilate_U1_U1_3x3_output_1.bin
    # filter/agoKernel_Erode_U8_U8_3x3_output_1.bin
    # filter/agoKernel_Erode_U1_U8_3x3_output_1.bin
    # filter/agoKernel_Erode_U8_U1_3x3_output_1.bin
    # filter/agoKernel_Erode_U1_U1_3x3_output_1.bin
    # filter/agoKernel_Median_U8_U8_3x3_output_1.bin
    # filter/agoKernel_Gaussian_U8_U8_3x3_output_1.bin
    # filter/agoKernel_HalfGaussian_U8_U8_3x3_output_1.bin
    # filter/agoKernel_HalfGaussian_U8_U8_5x5_output_1.bin
    # filter/agoKernel_Convolve_U8_U8_3x3_output_1.bin
    # filter/agoKernel_Convolve_S16_U8_3x3_output_1.bin
    # filter/agoKernel_Sobel_S16S16_U8_3x3_GXY_output_1.bin
    # filter/agoKernel_Sobel_S16_U8_3x3_GX_output_1.bin
    # filter/agoKernel_Sobel_S16_U8_3x3_GY_output_1.bin
    # geometric/agoKernel_ScaleImage_U8_U8_Nearest_output_1.bin
    # geometric/agoKernel_ScaleImage_U8_U8_bilinear_output_1.bin
    # geometric/agoKernel_ScaleImage_U8_U8_bilinear_replicate_output_1.bin
    # geometric/agoKernel_ScaleImage_U8_U8_bilinear_constant_output_1.bin
    # geometric/agoKernel_ScaleImage_U8_U8_u8_area_output_1.bin
    # geometric/agoKernel_WarpAffine_U8_U8_Nearest_output_1.bin
    # geometric/agoKernel_WarpAffine_U8_U8_Nearest_constant_output_1.bin
    # geometric/agoKernel_WarpAffine_U8_U8_Nearestbilinear_output_1.bin
    # geometric/agoKernel_WarpAffine_U8_U8_Nearest_bilinear_constant_output_1.bin
    # geometric/agoKernel_WarpPerspective_U8_U8_Nearest_output_1.bin
    # geometric/agoKernel_WarpPerspective_U8_U8_Nearest_output_1.bin
    # geometric/agoKernel_WarpPerspective_U8_U8_Nearest_bilinear_output_1.bin
    # geometric/agoKernel_WarpPerspective_U8_U8_Nearest_constant_output_1.bin
    # "

    UNMATCHED_OUTPUT_LIST=""

    for OUTPUT_BIN in $OUTPUT_BIN_LIST;
    do
        printf "\n"
        echo "Checking $OUTPUT_BIN..."
        DIFF=$(diff "dumpsHIP/$OUTPUT_BIN" "dumpsOCL/$OUTPUT_BIN")
        if [ "$DIFF" != "" ]
        then
            echo "Outputs don't match!"
            UNMATCHED_OUTPUT_LIST="$UNMATCHED_OUTPUT_LIST $OUTPUT_BIN"
        fi
    done

    printf "\n\n"
    echo "Kernels for which OCL and HIP don't match:"
    for UNMATCHED_OUTPUT in $UNMATCHED_OUTPUT_LIST;
    do
        echo "$UNMATCHED_OUTPUT"
    done

    printf "\nFinished running funcitonalities. Finished the OCLvsHIP output matching test."
else
    printf "\nFinished running funcitonalities. Bin dump generation is required for the OCLvsHIP output matching test."
fi

############# Need not edit #############