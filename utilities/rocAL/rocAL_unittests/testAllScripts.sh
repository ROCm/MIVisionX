#!/bin/bash
cd build
image_path=/media/unittest/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/
coco_classification_path=/media/unittest/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017
coco_detection_path=/media/unittest/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/
tf_classification_path=/media/unittest/MIVisionX-data/rocal_data/tf/classification
tf_detection_path=/media/unittest/MIVisionX-data/rocal_data/tf/detection/
caffe_path=/media/unittest/MIVisionX-data/rocal_data/caffe/classification/
caffe2_path=/media/unittest/MIVisionX-data/rocal_data/caffe2/classification/
mxnet_path=/media/unittest/MIVisionX-data/rocal_data/mxnet/
output_path=../OUTPUT_FOLDER_NEW/
 
check=1
display=0
cpu=0
width=640 
height=480
device="host"
if [ $cpu -eq $check ]
then 
    device="hip"
fi
 #FileSource Reader
./rocAL_unittests 0 $image_path ${output_path}LensCorrection_rgb_${device}  $width $height 45 $cpu 1 0 $display
./rocAL_unittests 0 $image_path ${output_path}Exposure_rgb_${device}  $width $height 46 $cpu 1 0 $display
./rocAL_unittests 0 $image_path ${output_path}Flip_rgb_${device}  $width $height 47 $cpu 1 0 $display

#coco detection
./rocAL_unittests 2 $coco_detection_path ${output_path}Gamma_rgb_${device}  $width $height 33 $cpu 1 0 $display
./rocAL_unittests 2 $coco_detection_path ${output_path}Contrast_rgb_${device}  $width $height 34 $cpu 1 0 $display
./rocAL_unittests 2 $coco_detection_path ${output_path}Vignette_rgb_${device}  $width $height 38 0 1 $cpu $display

#tf classification
./rocAL_unittests 4 $tf_classification_path ${output_path}Blend_rgb_${device} $width $height 36 $cpu 1 0 $display
./rocAL_unittests 4 $tf_classification_path ${output_path}WarpAffine_rgb_${device}  $width $height 37 $cpu 1 0 $display
./rocAL_unittests 4 $tf_classification_path ${output_path}Blur_rgb_${device}  $width $height 35 $cpu 1 0 $display

#tf detection
./rocAL_unittests 5 $tf_detection_path ${output_path}Jitter_rgb_${device}  $width $height 39 $cpu 1 0 $display
./rocAL_unittests 5 $tf_detection_path ${output_path}ColorTemp_rgb_${device}  $width $height 43 $cpu 1 0 $display
./rocAL_unittests 5 $tf_detection_path ${output_path}CropResize_rgb_${device} $width $height 30 $cpu 1 0 $display 


# caffe classification
./rocAL_unittests 6 $caffe_path ${output_path}Rotate_rgb_${device}  $width $height 31 $cpu 1 0 $display
./rocAL_unittests 6 $caffe_path ${output_path}Brightness_rgb_${device} $width $height 32 $cpu 1 0 $display
./rocAL_unittests 6 $caffe_path ${output_path}Hue_rgb_${device}  $width $height 48 $cpu 1 0 $display

# caffe detection
./rocAL_unittests 7 $caffe_path ${output_path}Saturation_rgb_${device}  $width $height 49 $cpu 1 0 $display
./rocAL_unittests 7 $caffe_path ${output_path}ColorTwist_rgb_${device}  $width $height 50 $cpu 1 0 $display
./rocAL_unittests 7 $caffe_path ${output_path}Crop_rgb_${device}  $width $height 51 $cpu 1 0 $display

# caffe2 classification
./rocAL_unittests 8 $caffe2_path ${output_path}CropCenter_rgb_${device}  $width $height 52 $cpu 1 0 $display
./rocAL_unittests 8 $caffe2_path ${output_path}ResizeCropMirror_rgb_${device}  $width $height 53 $cpu 1 0 $display
./rocAL_unittests 8 $caffe2_path ${output_path}Resize_rgb_${device}_lanczos_Default_caffe2Classification  $width $height 0 $cpu 1 0 $display 3 0

# caffe2 detection
./rocAL_unittests 9 $caffe2_path ${output_path}FishEye_rgb_${device}  $width $height 10 $cpu 1 0 $display
./rocAL_unittests 9 $caffe2_path ${output_path}Pixelate_rgb_${device}  $width $height 19 $cpu 1 0 $display
./rocAL_unittests 9 $caffe2_path ${output_path}CropMirrorNormalize_rgb_${device}  $width $height 55 $cpu 1 0 $display

#mxnet 
./rocAL_unittests 11 $mxnet_path ${output_path}Resize_rgb_${device}_gaussian_Default_mxnet  $width $height 0 $cpu 1 0 $display 5 0 5 0 
./rocAL_unittests 11 $mxnet_path ${output_path}Pixelate_rgb_${device}  $width $height 19 $cpu 1 0 $display
./rocAL_unittests 11 $mxnet_path ${output_path}CropMirrorNormalize_rgb_${device}  $width $height 55 $cpu 1 0 $display


#CMN 
./rocAL_unittests 0 $image_path ${output_path}CropMirrorNormalize_rgb_${device}_FileReader  $width $height 55 $cpu 1 0 $display
./rocAL_unittests 2 $coco_detection_path ${output_path}CropMirrorNormalize_rgb_${device}_coco  $width $height 55 $cpu 1 0 $display
./rocAL_unittests 4 $tf_classification_path ${output_path}CropMirrorNormalize_rgb_${device}_tfClassification $width $height 55 $cpu 1 0 $display
./rocAL_unittests 5 $tf_detection_path ${output_path}CropMirrorNormalize_rgb_${device}_tfDetection  $width $height 55 $cpu 1 0 $display
./rocAL_unittests 6 $caffe_path ${output_path}CropMirrorNormalize_rgb_${device}_caffeClassification  $width $height 55 $cpu 1 0 $display
./rocAL_unittests 7 $caffe_path ${output_path}CropMirrorNormalize_rgb_${device}_caffeDetection  $width $height 55 $cpu 1 0 $display
./rocAL_unittests 8 $caffe2_path ${output_path}CropMirrorNormalize_rgb_${device}_caffe2Classification  $width $height 55 $cpu 1 0 $display
./rocAL_unittests 9 $caffe2_path ${output_path}CropMirrorNormalize_rgb_${device}_caffe2Detection  $width $height 55 $cpu 1 0 $display
./rocAL_unittests 11 $mxnet_path ${output_path}CropMirrorNormalize_rgb_${device}_mxnet  $width $height 55 $cpu 1 0 $display

# crop
./rocAL_unittests 0 $image_path ${output_path}Crop_rgb_${device}_FileReader  $width $height 51 $cpu 1 0 $display
./rocAL_unittests 2 $coco_detection_path ${output_path}Crop_rgb_${device}_coco  $width $height 51 $cpu 1 0 $display
./rocAL_unittests 4 $tf_classification_path ${output_path}Crop_rgb_${device}_tfClassification $width $height 51 $cpu 1 0 $display
./rocAL_unittests 5 $tf_detection_path ${output_path}Crop_rgb_${device}_tfDetection  $width $height 51 $cpu 1 0 $display
./rocAL_unittests 6 $caffe_path ${output_path}Crop_rgb_${device}_caffeClassification  $width $height 51 $cpu 1 0 $display
./rocAL_unittests 7 $caffe_path ${output_path}Crop_rgb_${device}_caffeDetection  $width $height 51 $cpu 1 0 $display
./rocAL_unittests 8 $caffe2_path ${output_path}Crop_rgb_${device}_caffe2Classification  $width $height 51 $cpu 1 0 $display
./rocAL_unittests 9 $caffe2_path ${output_path}Crop_rgb_${device}_caffe2Detection  $width $height 51 $cpu 1 0 $display
./rocAL_unittests 11 $mxnet_path ${output_path}Crop_rgb_${device}_mxnet  $width $height 51 $cpu 1 0 $display


#resize
# Last two parameter are interpolation type and scaling mode

./rocAL_unittests 0 $image_path ${output_path}Resize_rgb_${device}_bilinear_Default_FileReader  $width $height 0 $cpu 1 0 $display 1 0 
./rocAL_unittests 2 $coco_detection_path ${output_path}Resize_rgb_${device}_bilinear_Stretch_coco  $width $height 0 $cpu 1 0 $display 1 1
./rocAL_unittests 4 $tf_classification_path ${output_path}Resize_rgb_${device}_bilinear_NotSmaller_tfClassification $width $height 0 $cpu 1 0 $display 1 2
./rocAL_unittests 5 $tf_detection_path ${output_path}Resize_rgb_${device}_bilinear_Notlarger_tfDetection  $width $height 0 $cpu 1 0 $display 1 3
./rocAL_unittests 6 $caffe_path ${output_path}Resize_rgb_${device}_bicubic_Default_caffeClassification  $width $height 0 $cpu 1 0 $display 1 0
./rocAL_unittests 7 $caffe_path ${output_path}Resize_rgb_${device}_nearestneighbour_Default_caffeDetection  $width $height 0 $cpu 1 0 $display 0 0 
./rocAL_unittests 8 $caffe2_path ${output_path}Resize_rgb_${device}_lanczos_Default_caffe2Classification  $width $height 0 $cpu 1 0 $display 3 0
./rocAL_unittests 9 $caffe2_path ${output_path}Resize_rgb_${device}_triangular_Default_caffe2Detection  $width $height 0 $cpu 1 0 $display 4 0
./rocAL_unittests 11 $mxnet_path ${output_path}Resize_rgb_${device}_gaussian_Default_mxnet  $width $height 0 $cpu 1 0 $display 5 0 5 0 












