#!/bin/bash
cd build
image_path=/media/akilesh/unittest_script/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/
coco_classification_path=/media/akilesh/unittest_script/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017
coco_detection_path=/media/akilesh/unittest_script/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/
tf_classification_path=/media/akilesh/unittest_script/MIVisionX-data/rocal_data/tf/classification
tf_detection_path=/media/akilesh/unittest_script/MIVisionX-data/rocal_data/tf/detection/
caffe_path=/media/akilesh/unittest_script/MIVisionX-data/rocal_data/caffe/classification/
caffe2_path=/media/akilesh/unittest_script/MIVisionX-data/rocal_data/caffe2/classification/
mxnet_path=/media/akilesh/unittest_script/MIVisionX-data/rocal_data/mxnet/
output_path=../OUTPUT_FOLDER_NEW/
 
check=1
display=0
cpu=1
width=640 
height=480
device="host"
if [ $cpu -eq $check ]
then 
    device="hip"
fi
# echo $device
# exit
# partial decoder
# ./rocAL_unittests 1 $coco_classification_path ${output_path}image_partial_decode_CropResize_${cpu}_  $width $height 30 $cpu 1 0 $display
# ./rocAL_unittests 1 $coco_classification_path ${output_path}image_partial_decode_Rotate_${cpu}_  $width $height 31 $cpu 1 0 $display
# ./rocAL_unittests 1 $coco_classification_path ${output_path}image_partial_decode_Brightness_${cpu}_  $width $height 32 $cpu 1 0 $display

./rocAL_unittests 0 $image_path ${output_path}LensCorrection_rgb_${device}  $width $height 45 $cpu 1 0 $display
./rocAL_unittests 0 $image_path ${output_path}Exposure_rgb_${device}  $width $height 46 $cpu 1 0 $display
./rocAL_unittests 0 $image_path ${output_path}Flip_rgb_${device}  $width $height 47 $cpu 1 0 $display

#coco detection
./rocAL_unittests 2 $coco_detection_path ${output_path}Gamma_rgb_${device}  $width $height 33 $cpu 1 0 $display
./rocAL_unittests 2 $coco_detection_path ${output_path}Contrast_rgb_${device}  $width $height 34 $cpu 1 0 $display
./rocAL_unittests 2 $coco_detection_path ${output_path}Vignette_rgb_${device}  $width $height 38 0 1 $cpu $display

# coco detection partial
# ./rocAL_unittests 3 $coco_detection_path ${output_path}coco_detection_partial_Blend_${cpu}_  $width $height 36 $cpu 1 0 $display
# ./rocAL_unittests 3 $coco_detection_path ${output_path}coco_detection_partial_WarpAffine_${cpu}_  $width $height 37 $cpu 1 0 $display
# ./rocAL_unittests 3 $coco_detection_path ${output_path}coco_detection_partial_Vignette_${cpu}_  $width $height 38 $cpu 1 0 $display

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
./rocAL_unittests 8 $caffe2_path ${output_path}Resize_rgb_${device}  $width $height 0 $cpu 1 0 $display

# caffe2 detection
./rocAL_unittests 9 $caffe2_path ${output_path}FishEye_rgb_${device}  $width $height 10 $cpu 1 0 $display
./rocAL_unittests 9 $caffe2_path ${output_path}Pixelate_rgb_${device}  $width $height 19 $cpu 1 0 $display
./rocAL_unittests 9 $caffe2_path ${output_path}CropMirrorNormalize_rgb_${device}  $width $height 55 $cpu 1 0 $display

#mxnet 
./rocAL_unittests 11 $mxnet_path ${output_path}Resize_rgb_${device}  $width $height 0 $cpu 1 0 $display
./rocAL_unittests 11 $mxnet_path ${output_path}Pixelate_rgb_${device}  $width $height 19 $cpu 1 0 $display
./rocAL_unittests 11 $mxnet_path ${output_path}CropMirrorNormalize_rgb_${device}  $width $height 55 $cpu 1 0 $display


#mxnet 
# ./rocAL_unittests 11 $mxnet_path ${output_path}Resize_rgb_${device}  $width $height 0 $cpu 1 0 $display
# ./rocAL_unittests 11 $mxnet_path ${output_path}Fog_rgb_${device}  $width $height 44 $cpu 1 0 $display
# ./rocAL_unittests 11 $mxnet_path ${output_path}CropMirrorNormalize_rgb_${device}  $width $height 55 $cpu 1 0 $display