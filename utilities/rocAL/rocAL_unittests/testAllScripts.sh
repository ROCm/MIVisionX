#!/bin/bash
if [$ROCAL_DATA_PATH == ""]
then 
    echo "Need to export ROCAL_DATA_PATH"
    exit
fi

if [ -z "$1" ]
  then
    echo "No device argument supplied."
    echo "Usage : ./testAllScript.sh 0(for HOST)/1(for HIP backend)"
    exit
fi

if [ -d build ];then 
    rm -rf ./build/*
else
    mkdir build
fi
cd build
cmake ..
make -j$nproc

image_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/train_10images_2017/
coco_classification_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/train_10images_2017/
coco_detection_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/train_10images_2017/
tf_classification_path=${ROCAL_DATA_PATH}/rocal_data/tf/classification/
tf_detection_path=${ROCAL_DATA_PATH}/rocal_data/tf/detection/
caffe_path=${ROCAL_DATA_PATH}/rocal_data/caffe/classification/
caffe2_path=${ROCAL_DATA_PATH}/rocal_data/caffe2/classification/
mxnet_path=${ROCAL_DATA_PATH}/rocal_data/mxnet/
display=0
device=$1
width=640 
height=480
device_name="host"

if [ $device -eq 1 ]
then 
    device_name="hip"
    echo "Running HIP Backend"
else
    echo "Running on HOST Backend"
fi

output_path="../OUTPUT_FOLDER_$(date +%Y-%m-%d_%H-%M-%S)_"$device_name"/"
mkdir $output_path

 #FileSource Reader
./rocAL_unittests 0 $image_path ${output_path}LensCorrection_rgb_${device_name} $width $height 45 $device 1 0 $display
./rocAL_unittests 0 $image_path ${output_path}Exposure_rgb_${device_name} $width $height 46 $device 1 0 $display
./rocAL_unittests 0 $image_path ${output_path}Flip_rgb_${device_name} $width $height 47 $device 1 0 $display

#coco detection
./rocAL_unittests 2 $coco_detection_path ${output_path}Gamma_rgb_${device_name} $width $height 33 $device 1 0 $display
./rocAL_unittests 2 $coco_detection_path ${output_path}Contrast_rgb_${device_name} $width $height 34 $device 1 0 $display
./rocAL_unittests 2 $coco_detection_path ${output_path}Vignette_rgb_${device_name} $width $height 38 0 1 $device $display

#tf classification
./rocAL_unittests 4 $tf_classification_path ${output_path}Blend_rgb_${device_name} $width $height 36 $device 1 0 $display
./rocAL_unittests 4 $tf_classification_path ${output_path}WarpAffine_rgb_${device_name} $width $height 37 $device 1 0 $display
./rocAL_unittests 4 $tf_classification_path ${output_path}Blur_rgb_${device_name} $width $height 35 $device 1 0 $display

#tf detection
./rocAL_unittests 5 $tf_detection_path ${output_path}SNPNoise_rgb_${device_name} $width $height 40 $device 1 0 $display
./rocAL_unittests 5 $tf_detection_path ${output_path}ColorTemp_rgb_${device_name} $width $height 43 $device 1 0 $display
./rocAL_unittests 5 $tf_detection_path ${output_path}Fog_rgb_${device_name} $width $height 44 $device 1 0 $display

# caffe classification
./rocAL_unittests 6 $caffe_path ${output_path}Rotate_rgb_${device_name} $width $height 31 $device 1 0 $display
./rocAL_unittests 6 $caffe_path ${output_path}Brightness_rgb_${device_name} $width $height 32 $device 1 0 $display
./rocAL_unittests 6 $caffe_path ${output_path}Hue_rgb_${device_name} $width $height 48 $device 1 0 $display

# caffe detection
./rocAL_unittests 7 $caffe_path ${output_path}Saturation_rgb_${device_name} $width $height 49 $device 1 0 $display
./rocAL_unittests 7 $caffe_path ${output_path}ColorTwist_rgb_${device_name} $width $height 50 $device 1 0 $display
./rocAL_unittests 7 $caffe_path ${output_path}Rain_rgb_${device_name} $width $height 42 $device 1 0 $display

# caffe2 classification
./rocAL_unittests 8 $caffe2_path ${output_path}CropCenter_rgb_${device_name} $width $height 52 $device 1 0 $display
./rocAL_unittests 8 $caffe2_path ${output_path}ResizeCropMirror_rgb_${device_name} $width $height 53 $device 1 0 $display
./rocAL_unittests 8 $caffe2_path ${output_path}Snow_rgb_${device_name} $width $height 41 $device 1 0 $display

# caffe2 detection
./rocAL_unittests 9 $caffe2_path ${output_path}FishEye_rgb_${device_name} $width $height 10 $device 1 0 $display
./rocAL_unittests 9 $caffe2_path ${output_path}Pixelate_rgb_${device_name} $width $height 19 $device 1 0 $display
./rocAL_unittests 9 $caffe2_path ${output_path}CropCenter_rgb_${device_name}_cmn $width $height 55 $device 1 0 $display

#mxnet 
./rocAL_unittests 11 $mxnet_path ${output_path}Jitter_rgb_${device_name} $width $height 39 $device 1 0 $display
./rocAL_unittests 11 $mxnet_path ${output_path}Pixelate_rgb_${device_name} $width $height 19 $device 1 0 $display
./rocAL_unittests 11 $mxnet_path ${output_path}CropMirrorNormalize_rgb_${device_name}_mxnet $width $height 25 $device 1 0 $display

#CMN 
./rocAL_unittests 0 $image_path ${output_path}CropMirrorNormalize_rgb_${device_name}_FileReader $width $height 25 $device 1 0 $display
./rocAL_unittests 2 $coco_detection_path ${output_path}CropMirrorNormalize_rgb_${device_name}_coco $width $height 25 $device 1 0 $display
./rocAL_unittests 4 $tf_classification_path ${output_path}CropMirrorNormalize_rgb_${device_name}_tfClassification $width $height 25 $device 1 0 $display
./rocAL_unittests 5 $tf_detection_path ${output_path}CropMirrorNormalize_rgb_${device_name}_tfDetection $width $height 25 $device 1 0 $display
./rocAL_unittests 6 $caffe_path ${output_path}CropMirrorNormalize_rgb_${device_name}_caffeClassification $width $height 25 $device 1 0 $display
./rocAL_unittests 7 $caffe_path ${output_path}CropMirrorNormalize_rgb_${device_name}_caffeDetection $width $height 25 $device 1 0 $display
./rocAL_unittests 8 $caffe2_path ${output_path}CropMirrorNormalize_rgb_${device_name}_caffe2Classification $width $height 25 $device 1 0 $display
./rocAL_unittests 9 $caffe2_path ${output_path}CropMirrorNormalize_rgb_${device_name}_caffe2Detection $width $height 25 $device 1 0 $display
./rocAL_unittests 11 $mxnet_path ${output_path}CropMirrorNormalize_rgb_${device_name}_mxnet $width $height 25 $device 1 0 $display

# crop
./rocAL_unittests 0 $image_path ${output_path}Crop_rgb_${device_name}_FileReader $width $height 51 $device 1 0 $display
./rocAL_unittests 2 $coco_detection_path ${output_path}Crop_rgb_${device_name}_coco $width $height 51 $device 1 0 $display
./rocAL_unittests 4 $tf_classification_path ${output_path}Crop_rgb_${device_name}_tfClassification $width $height 51 $device 1 0 $display
./rocAL_unittests 5 $tf_detection_path ${output_path}Crop_rgb_${device_name}_tfDetection $width $height 51 $device 1 0 $display
./rocAL_unittests 6 $caffe_path ${output_path}Crop_rgb_${device_name}_caffeClassification $width $height 51 $device 1 0 $display
./rocAL_unittests 7 $caffe_path ${output_path}Crop_rgb_${device_name}_caffeDetection $width $height 51 $device 1 0 $display
./rocAL_unittests 8 $caffe2_path ${output_path}Crop_rgb_${device_name}_caffe2Classification $width $height 51 $device 1 0 $display
./rocAL_unittests 9 $caffe2_path ${output_path}Crop_rgb_${device_name}_caffe2Detection $width $height 51 $device 1 0 $display
./rocAL_unittests 11 $mxnet_path ${output_path}Crop_rgb_${device_name}_mxnet $width $height 51 $device 1 0 $display

#resize
# Last two parameter are interpolation type and scaling mode
./rocAL_unittests 0 $image_path ${output_path}Resize_rgb_${device_name}_bilinear_default_FileReader $width $height 0 $device 1 0 $display 1 0 
./rocAL_unittests 2 $coco_detection_path ${output_path}Resize_rgb_${device_name}_bilinear_stretch_coco $width $height 0 $device 1 0 $display 1 1
./rocAL_unittests 4 $tf_classification_path ${output_path}Resize_rgb_${device_name}_bilinear_notsmaller_tfClassification $width $height 0 $device 1 0 $display 1 2
./rocAL_unittests 5 $tf_detection_path ${output_path}Resize_rgb_${device_name}_bilinear_notlarger_tfDetection $width $height 0 $device 1 0 $display 1 3
./rocAL_unittests 6 $caffe_path ${output_path}Resize_rgb_${device_name}_bicubic_default_caffeClassification $width $height 0 $device 1 0 $display 1 0
./rocAL_unittests 7 $caffe_path ${output_path}Resize_rgb_${device_name}_nearestneighbour_default_caffeDetection $width $height 0 $device 1 0 $display 0 0 
./rocAL_unittests 8 $caffe2_path ${output_path}Resize_rgb_${device_name}_lanczos_default_caffe2Classification $width $height 0 $device 1 0 $display 3 0
./rocAL_unittests 9 $caffe2_path ${output_path}Resize_rgb_${device_name}_triangular_default_caffe2Detection $width $height 0 $device 1 0 $display 4 0
./rocAL_unittests 11 $mxnet_path ${output_path}Resize_rgb_${device_name}_gaussian_default_mxnet $width $height 0 $device 1 0 $display 5 0 
