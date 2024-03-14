/*
MIT License

Copyright (c) 2018 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <unistd.h>
#include <vector>
#include <string>
#include <cstdlib>


#include "rocal_api.h"

#include "opencv2/opencv.hpp"
using namespace cv;

#if USE_OPENCV_4
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define CV_GRAY2RGB COLOR_GRAY2RGB
#define CV_RGB2BGR COLOR_RGB2BGR
#define CV_FONT_HERSHEY_SIMPLEX FONT_HERSHEY_SIMPLEX
#define CV_FILLED FILLED
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
#endif

#define DISPLAY 0
//#define RANDOMBBOXCROP

using namespace std::chrono;

std::string get_interpolation_type(unsigned int val, RocalResizeInterpolationType &interpolation_type) {
    switch(val) {
        case 0: {
            interpolation_type = ROCAL_NEAREST_NEIGHBOR_INTERPOLATION;
            return "NearestNeighbor";
        }
        case 2: {
            interpolation_type = ROCAL_CUBIC_INTERPOLATION;
            return "Bicubic";
        }
        case 3: {
            interpolation_type = ROCAL_LANCZOS_INTERPOLATION;
            return "Lanczos";
        }
        case 4: {
            interpolation_type = ROCAL_GAUSSIAN_INTERPOLATION;
            return "Gaussian";
        }
        case 5: {
            interpolation_type = ROCAL_TRIANGULAR_INTERPOLATION;
            return "Triangular";
        }
        default: {
            interpolation_type = ROCAL_LINEAR_INTERPOLATION;
            return "Bilinear";
        }
    }
}

std::string get_scaling_mode(unsigned int val, RocalResizeScalingMode &scale_mode) {
    switch(val) {
        case 1: {
            scale_mode = ROCAL_SCALING_MODE_STRETCH;
            return "Stretch";
        }
        case 2: {
            scale_mode = ROCAL_SCALING_MODE_NOT_SMALLER;
            return "NotSmaller";
        }
        case 3: {
            scale_mode = ROCAL_SCALING_MODE_NOT_LARGER;
            return "Notlarger";
        }
        default: {
            scale_mode = ROCAL_SCALING_MODE_DEFAULT;
            return "Default";
        }
    }
}

int test(int test_case, int reader_type, const char *path, const char *outName, int rgb, int gpu, int width, int height,int num_of_classes, int display_all, int resize_interpolation_type, int resize_scaling_mode);
int main(int argc, const char **argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT)
    {
        printf("Usage: rocal_unittests reader-type <image-dataset-folder> output_image_name <width> <height> test_case gpu=1/cpu=0 rgb=1/grayscale=0 one_hot_labels=num_of_classes/0  display_all=0(display_last_only)1(display_all)\n");
        return -1;
    }

    int argIdx = 0;
    int reader_type = atoi(argv[++argIdx]);
    const char *path = argv[++argIdx];
    const char *outName = argv[++argIdx];
    int width = atoi(argv[++argIdx]);
    int height = atoi(argv[++argIdx]);
    int display_all = 0;

    int rgb = 1; // process color images
    bool gpu = 1;
    int test_case = 3; // For Rotate
    int num_of_classes = 0;
    int resize_interpolation_type = 1; // For Bilinear interpolations
    int resize_scaling_mode = 0; // For Default scaling mode

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        num_of_classes = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        display_all = atoi(argv[++argIdx]);
    
    if (argc >= argIdx + MIN_ARG_COUNT)
        resize_interpolation_type = atoi(argv[++argIdx]);
    
    if (argc >= argIdx + MIN_ARG_COUNT)
        resize_scaling_mode = atoi(argv[++argIdx]);

    test(test_case, reader_type, path, outName, rgb, gpu, width, height, num_of_classes, display_all, resize_interpolation_type, resize_scaling_mode);

    return 0;
}

int test(int test_case, int reader_type, const char *path, const char *outName, int rgb, int gpu, int width, int height, int num_of_classes, int display_all, int resize_interpolation_type, int resize_scaling_mode)
{
    size_t num_threads = 1;
    unsigned int inputBatchSize = 2;
    int decode_max_width = width;
    int decode_max_height = height;
    int pipeline_type = -1;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << " , " << (rgb ? " Color " : " Grayscale ") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24
                                             : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(inputBatchSize,
                             gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                             1);

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>> Getting the path for MIVisionX-data  <<<<<<<<<<<<<<<<*/

    std::string rocal_data_path;
    if(std::getenv("ROCAL_DATA_PATH"))
        rocal_data_path = std::getenv("ROCAL_DATA_PATH");

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalSetSeed(0);

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalIntParam color_temp_adj = rocalCreateIntParameter(-50);
    RocalIntParam mirror = rocalCreateIntParameter(1);


    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/

#if defined RANDOMBBOXCROP
    bool all_boxes_overlap = true;
    bool no_crop = false;
#endif

    RocalImage input1;
    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    switch (reader_type)
    {
        case 1: //image_partial decode
        {
            std::cout << ">>>>>>> Running PARTIAL DECODE" << std::endl;
            pipeline_type = 1;
            rocalCreateLabelReader(handle, path);
            std::vector<float> area = {0.08, 1};
            std::vector<float> aspect_ratio = {3.0f/4, 4.0f/3};
            input1 = rocalFusedJpegCrop(handle, path, color_format, num_threads, false, area, aspect_ratio, 10, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 2: //coco detection
        {
            std::cout << ">>>>>>> Running COCO READER" << std::endl;
            pipeline_type = 2;
            if (strcmp(rocal_data_path.c_str(), "") == 0)
            {
                std::cout << "\n ROCAL_DATA_PATH env variable has not been set. ";
                exit(0);
            }
            // setting the default json path to ROCAL_DATA_PATH coco sample train annotation
            std::string json_path = rocal_data_path + "/rocal_data/coco/coco_10_img/annotations/instances_train2017.json";
            rocalCreateCOCOReader(handle, json_path.c_str(), true);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                input1 = rocalJpegCOCOFileSource(handle, path, json_path.c_str(), color_format, num_threads, false, true, false);
            else
                input1 = rocalJpegCOCOFileSource(handle, path, json_path.c_str(), color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 3: //coco detection partial
        {
            std::cout << ">>>>>>> Running COCO READER PARTIAL" << std::endl;
            pipeline_type = 2;
            if (strcmp(rocal_data_path.c_str(), "") == 0)
            {
                std::cout << "\n ROCAL_DATA_PATH env variable has not been set. ";
                exit(0);
            }
            // setting the default json path to ROCAL_DATA_PATH coco sample train annotation
            std::string json_path = rocal_data_path + "/rocal_data/coco/coco_10_img/annotations/instances_train2017.json";
            rocalCreateCOCOReader(handle, json_path.c_str(), true);
#if defined RANDOMBBOXCROP
            rocalRandomBBoxCrop(handle, all_boxes_overlap, no_crop);
#endif
            std::vector<float> area = {0.08, 1};
            std::vector<float> aspect_ratio = {3.0f/4, 4.0f/3};
            input1 = rocalJpegCOCOFileSourcePartial(handle, path, json_path.c_str(), color_format, num_threads, false, area, aspect_ratio, 10, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 4: //tf classification
        {
            std::cout << ">>>>>>> Running TF CLASSIFICATION READER" << std::endl;
            pipeline_type = 1;
            char key1[25] = "image/encoded";
            char key2[25] = "image/class/label";
            char key8[25] = "image/filename";
            rocalCreateTFReader(handle, path, true, key2, key8);
            input1 = rocalJpegTFRecordSource(handle, path, color_format, num_threads, false, key1, key8, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 5: //tf detection
        {
            std::cout << ">>>>>>> Running TF DETECTION READER" << std::endl;
            pipeline_type = 2;
            char key1[25] = "image/encoded";
            char key2[25] = "image/object/class/label";
            char key3[25] = "image/object/class/text";
            char key4[25] = "image/object/bbox/xmin";
            char key5[25] = "image/object/bbox/ymin";
            char key6[25] = "image/object/bbox/xmax";
            char key7[25] = "image/object/bbox/ymax";
            char key8[25] = "image/filename";
            rocalCreateTFReaderDetection(handle, path, true, key2, key3, key4, key5, key6, key7, key8);
            input1 = rocalJpegTFRecordSource(handle, path, color_format, num_threads, false, key1, key8, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 6: //caffe classification
        {
            std::cout << ">>>>>>> Running CAFFE CLASSIFICATION READER" << std::endl;
            pipeline_type = 1;
            rocalCreateCaffeLMDBLabelReader(handle, path);
            input1 = rocalJpegCaffeLMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 7: //caffe detection
        {
            std::cout << ">>>>>>> Running CAFFE DETECTION READER" << std::endl;
            pipeline_type = 2;
            rocalCreateCaffeLMDBReaderDetection(handle, path);
            input1 = rocalJpegCaffeLMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 8: //caffe2 classification
        {
            std::cout << ">>>>>>> Running CAFFE2 CLASSIFICATION READER" << std::endl;
            pipeline_type = 1;
            rocalCreateCaffe2LMDBLabelReader(handle, path, true);
            input1 = rocalJpegCaffe2LMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 9: //caffe2 detection
        {
            std::cout << ">>>>>>> Running CAFFE2 DETECTION READER" << std::endl;
            pipeline_type = 2;
            rocalCreateCaffe2LMDBReaderDetection(handle, path, true);
            input1 = rocalJpegCaffe2LMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 10: //coco reader keypoints
        {
            std::cout << ">>>>>>> Running COCO KEYPOINTS READER" << std::endl;
            pipeline_type = 3;
            if (strcmp(rocal_data_path.c_str(), "") == 0)
            {
                std::cout << "\n ROCAL_DATA_PATH env variable has not been set. ";
                exit(0);
            }
            // setting the default json path to ROCAL_DATA_PATH coco sample train annotation
            std::string json_path = rocal_data_path + "/rocal_data/coco/coco_10_img_keypoints/annotations/person_keypoints_val2017.json";
            float sigma = 3.0;
            rocalCreateCOCOReaderKeyPoints(handle, json_path.c_str(), true, sigma, (unsigned)width, (unsigned)height);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                input1 = rocalJpegCOCOFileSource(handle, path, json_path.c_str(), color_format, num_threads, false, true, false);
            else
                input1 = rocalJpegCOCOFileSource(handle, path, json_path.c_str(), color_format, num_threads, false, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        case 11: // mxnet reader
        {
            std::cout << ">>>>>>> Running MXNET READER" << std::endl;
            pipeline_type = 1;
            rocalCreateMXNetReader(handle, path, true);
            input1 = rocalMXNetRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
        default:
        {
            std::cout << ">>>>>>> Running IMAGE READER" << std::endl;
            pipeline_type = 1;
            rocalCreateLabelReader(handle, path);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                input1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, true);
            else
                input1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        }
        break;
    }

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = width, resize_h = height; // height and width

    RocalImage image0 = input1;
    // RocalImage image0 = rocalResize(handle, input1, resize_w, resize_h, false); // uncomment when processing images of different size
    RocalImage image1;
    
    if((test_case == 48 || test_case == 49 || test_case == 50) && rgb == 0) {
        std::cout << "Not a valid option! Exiting!\n";
        return -1;
    }
    switch (test_case)
    {
    case 0:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalResize" << std::endl;
        resize_w = 400;
        resize_h = 400;
        std::string interpolation_type_name, scaling_node_name;
        RocalResizeInterpolationType interpolation_type;
        RocalResizeScalingMode scale_mode;
        interpolation_type_name = get_interpolation_type(resize_interpolation_type, interpolation_type);
        scaling_node_name = get_scaling_mode(resize_scaling_mode, scale_mode);
        std::cerr<<" \n Interpolation_type_name " << interpolation_type_name;
        std::cerr<<" \n Scaling_node_name " << scaling_node_name;
        if (scale_mode != ROCAL_SCALING_MODE_DEFAULT && interpolation_type != ROCAL_LINEAR_INTERPOLATION) { // (Reference output available for bilinear interpolation for this  
            std::cerr<<" \n Running "<< scaling_node_name << " scaling mode with Bilinear interpolation for comparison \n";
            interpolation_type = ROCAL_LINEAR_INTERPOLATION;
        }
        if(scale_mode == ROCAL_SCALING_MODE_STRETCH) // For reference Output comparison 
            image1 = rocalResize(handle, image0, resize_w, 0, true, scale_mode, {}, 0, 0, interpolation_type);
        else
            image1 = rocalResize(handle, image0, resize_w, resize_h, true, scale_mode, {}, 0, 0, interpolation_type);
    }
    break;
    case 1:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalCropResize" << std::endl;
        image1 = rocalCropResize(handle, input1, resize_w, resize_h, true);
    }
    break;
    case 2:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalRotate" << std::endl;
        image1 = rocalRotate(handle, image0, true);
    }
    break;
    case 3:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBrightness" << std::endl;
        image1 = rocalBrightness(handle, image0, true);
    }
    break;
    case 4:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalGamma" << std::endl;
        image1 = rocalGamma(handle, image0, true);
    }
    break;
    case 5:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalContrast" << std::endl;
        image1 = rocalContrast(handle, image0, true);
    }
    break;
    case 6:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalFlip" << std::endl;
        image1 = rocalFlip(handle, image0, true);
    }
    break;
    case 7:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBlur" << std::endl;
        image1 = rocalBlur(handle, image0, true);
    }
    break;
    case 8:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBlend" << std::endl;
        RocalImage image0_b = rocalRotate(handle, image0, false);
        image1 = rocalBlend(handle, image0, image0_b, true);
    }
    break;
    case 9:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalWarpAffine" << std::endl;
        image1 = rocalWarpAffine(handle, image0, true);
    }
    break;
    case 10:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalFishEye" << std::endl;
        image1 = rocalFishEye(handle, image0, true);
    }
    break;
    case 11:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalVignette" << std::endl;
        image1 = rocalVignette(handle, image0, true);
    }
    break;
    case 12:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalJitter" << std::endl;
        image1 = rocalJitter(handle, image0, true);
    }
    break;
    case 13:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalSnPNoise" << std::endl;
        image1 = rocalSnPNoise(handle, image0, true);
    }
    break;
    case 14:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalSnow" << std::endl;
        image1 = rocalSnow(handle, image0, true);
    }
    break;
    case 15:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalRain" << std::endl;
        image1 = rocalRain(handle, image0, true);
    }
    break;
    case 16:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorTemp" << std::endl;
        image1 = rocalColorTemp(handle, image0, true);
    }
    break;
    case 17:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalFog" << std::endl;
        image1 = rocalFog(handle, image0, true);
    }
    break;
    case 18:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalLensCorrection" << std::endl;
        image1 = rocalLensCorrection(handle, image0, true);
    }
    break;
    case 19:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalPixelate" << std::endl;
        image1 = rocalPixelate(handle, image0, true);
    }
    break;
    case 20:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalExposure" << std::endl;
        image1 = rocalExposure(handle, image0, true);
    }
    break;
    case 21:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalHue" << std::endl;
        image1 = rocalHue(handle, image0, true);
    }
    break;
    case 22:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalSaturation" << std::endl;
        image1 = rocalSaturation(handle, image0, true);
    }
    break;
    case 23:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalCopy" << std::endl;
        image1 = rocalCopy(handle, image0, true);
    }
    break;
    case 24:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorTwist" << std::endl;
        image1 = rocalColorTwist(handle, image0, true);
    }
    break;
    case 25:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalCropMirrorNormalize" << std::endl;
        std::vector<float> mean;
        std::vector<float> std_dev;
        image1 = rocalCropMirrorNormalize(handle, image0, 1, 224, 224, 0.2, 0.2, 1, mean, std_dev, true, mirror);
    }
    break;
    case 26:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalCrop" << std::endl;
        image1 = rocalCrop(handle, image0, true);
    }
    break;
    case 27:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalResizeCropMirror" << std::endl;
        image1 = rocalResizeCropMirror(handle, image0, resize_w, resize_h, true);
    }
    break;

    case 30:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalCropResizeFixed" << std::endl;
        image1 = rocalCropResizeFixed(handle, image0, resize_w, resize_h, true, 0.25, 1.2, 0.6, 0.4);
    }
    break;
    case 31:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalRotateFixed" << std::endl;
        image1 = rocalRotateFixed(handle, image0, 50, true);
    }
    break;
    case 32:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBrightnessFixed" << std::endl;
        image1 = rocalBrightnessFixed(handle, image0, 1.90, 20, true);
    }
    break;
    case 33:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalGammaFixed" << std::endl;
        image1 = rocalGammaFixed(handle, image0, 0.5, true);
    }
    break;
    case 34:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalContrastFixed" << std::endl;
        image1 = rocalContrastFixed(handle, image0, 30, 80, true);
    }
    break;
    case 35:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBlurFixed" << std::endl;
        image1 = rocalBlurFixed(handle, image0, 5, true);
    }
    break;
    case 36:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBlendFixed" << std::endl;
        RocalImage image0_b = rocalRotateFixed(handle, image0, 50, false);
        image1 = rocalBlendFixed(handle, image0, image0_b, 0.5, true);
    }
    break;
    case 37:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalWarpAffineFixed" << std::endl;
        image1 = rocalWarpAffineFixed(handle, image0, 0.25, 0.25, 1, 1, 5, 5, true);
    }
    break;
    case 38:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalVignetteFixed" << std::endl;
        image1 = rocalVignetteFixed(handle, image0, 50, true);
    }
    break;
    case 39:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalJitterFixed" << std::endl;
        image1 = rocalJitterFixed(handle, image0, 3, true);
    }
    break;
    case 40:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalSnPNoiseFixed" << std::endl;
        image1 = rocalSnPNoiseFixed(handle, image0, 0.12, true);
    }
    break;
    case 41:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalSnowFixed" << std::endl;
        image1 = rocalSnowFixed(handle, image0, 0.2, true);
    }
    break;
    case 42:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalRainFixed" << std::endl;
        image1 = rocalRainFixed(handle, image0, 0.5, 2, 16, 0.25, true);
    }
    break;
    case 43:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorTempFixed" << std::endl;
        image1 = rocalColorTempFixed(handle, image0, 70, true);
    }
    break;
    case 44:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalFogFixed" << std::endl;
        image1 = rocalFogFixed(handle, image0, 0.5, true);
    }
    break;
    case 45:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalLensCorrectionFixed" << std::endl;
        image1 = rocalLensCorrectionFixed(handle, image0, 2.9, 1.2, true);
    }
    break;
    case 46:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalExposureFixed" << std::endl;
        image1 = rocalExposureFixed(handle, image0, 1, true);
    }
    break;
    case 47:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalFlipFixed" << std::endl;
        image1 = rocalFlipFixed(handle, image0, 2, true);
    }
    break;
    case 48:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalHueFixed" << std::endl;
        image1 = rocalHueFixed(handle, image0, 150, true);
    }
    break;
    case 49:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalSaturationFixed" << std::endl;
        image1 = rocalSaturationFixed(handle, image0, 0.3, true);
    }
    break;
    case 50:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorTwistFixed" << std::endl;
        image1 = rocalColorTwistFixed(handle, image0, 0.2, 10.0, 100.0, 0.25, true);
    }
    break;
    case 51:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalCropFixed" << std::endl;
        image1 = rocalCropFixed(handle, input1, 224, 224, 1, true, 0, 0, 2);
    }
    break;
    case 52:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalCropCenterFixed" << std::endl;
        image1 = rocalCropCenterFixed(handle, image0, 224, 224, 2, true);
    }
    break;
    case 53:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalResizeCropMirrorFixed" << std::endl;
        image1 = rocalResizeCropMirrorFixed(handle, image0, 300, 300, true, 250, 250, mirror);
    }
    break;
    case 54:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalSSDRandomCrop" << std::endl;
        image1 = rocalSSDRandomCrop(handle, input1, true);
    }
    break;
    case 55:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalCropMirrorNormalizeFixed_center crop" << std::endl;
        std::vector<float> mean;
        std::vector<float> std_dev;
        image1 = rocalCropMirrorNormalize(handle, image0, 1, 224, 224, 0.5, 0.5, 0.5, mean, std_dev, true);
    }
    break;

    default:
        std::cout << "Not a valid option! Exiting!\n";
        return -1;
    }

    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    printf("\n\nAugmented copies count %lu \n", rocalGetAugmentationBranchCount(handle));

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle);
    int w = rocalGetOutputWidth(handle);
    int p = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);
    const unsigned number_of_cols = 1; //1920 / w;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;
    int col_counter = 0;
    if(DISPLAY)
        cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    printf("Going to process images\n");
    printf("Remaining images %lu \n", rocalGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;

    while (rocalGetRemainingImages(handle) >= inputBatchSize)
    {
        index++;
        if (rocalRun(handle) != 0)
            break;
        int label_id[inputBatchSize];
        int numOfClasses = 0;
        int image_name_length[inputBatchSize];
        switch(pipeline_type)
        {
            case 1: //classification pipeline
            {
                rocalGetImageLabels(handle, label_id);
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                char img_name[img_size];
                numOfClasses = num_of_classes;
                int label_one_hot_encoded[inputBatchSize * numOfClasses];
                rocalGetImageName(handle, img_name);
                if (num_of_classes != 0)
                {
                    rocalGetOneHotImageLabels(handle, label_one_hot_encoded, numOfClasses,0);
                }
                std::cerr << "\nPrinting image names of batch: " << img_name<<"\n";
                for (unsigned int i = 0; i < inputBatchSize; i++)
                {
                    std::cerr<<"\t Printing label_id : " << label_id[i] << std::endl;
                    if(num_of_classes != 0)
                    {
                        std::cout << "One Hot Encoded labels:"<<"\t";
                        for (int j = 0; j < numOfClasses; j++)
                        {
                            int idx_value = label_one_hot_encoded[(i*numOfClasses)+j];
                            if(idx_value == 0)
                                std::cout << idx_value;
                            else
                            {
                                std::cout << idx_value;
                            }
                        }
                    }
                    std::cout << "\n";
                }
            }
            break;
            case 2: //detection pipeline
            {
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                char img_name[img_size];
                rocalGetImageName(handle, img_name);
                std::cerr << "\nPrinting image names of batch: " << img_name;
                int bb_label_count[inputBatchSize];
                int size = rocalGetBoundingBoxCount(handle, bb_label_count);
                for (int i = 0; i < (int)inputBatchSize; i++)
                    std::cerr << "\n Number of box:  " << bb_label_count[i];
                int bb_labels[size];
                rocalGetBoundingBoxLabel(handle, bb_labels);
                float bb_coords[size * 4];
                rocalGetBoundingBoxCords(handle, bb_coords);
                int img_sizes_batch[inputBatchSize * 2];
                rocalGetImageSizes(handle, img_sizes_batch);
                for (int i = 0; i < (int)inputBatchSize; i++)
                {
                    std::cout<<"\nwidth:"<<img_sizes_batch[i*2];
                    std::cout<<"\nHeight:"<<img_sizes_batch[(i*2)+1];
                }
            }
            break;
            case 3: // keypoints pipeline
            {
                int size = inputBatchSize;
                RocalJointsData *joints_data;
                rocalGetJointsDataPtr(handle, &joints_data);
                for (int i = 0; i < size; i++)
                {
                    std::cout << "ImageID: " << joints_data->image_id_batch[i] << std::endl;
                    std::cout << "AnnotationID: " << joints_data->annotation_id_batch[i] << std::endl;
                    std::cout << "ImagePath: " << joints_data->image_path_batch[i] << std::endl;
                    std::cout << "Center: " << joints_data->center_batch[i][0] << " " << joints_data->center_batch[i][1] << std::endl;
                    std::cout << "Scale: " << joints_data->scale_batch[i][0] << " " << joints_data->scale_batch[i][1] << std::endl;
                    std::cout << "Score: " << joints_data->score_batch[i] << std::endl;
                    std::cout << "Rotation: " << joints_data->rotation_batch[i] << std::endl;

                    for (int k = 0; k < 17; k++)
                    {
                    std::cout << "x : " << joints_data->joints_batch[i][k][0] << " , y : " << joints_data->joints_batch[i][k][1] << " , v : " << joints_data->joints_visibility_batch[i][k][0] << std::endl;
                    }
                }
            }
            break;
            default:
            {
                std::cout << "Not a valid pipeline type ! Exiting!\n";
                return -1;
            }
        }
        auto last_colot_temp = rocalGetIntValue(color_temp_adj);
        rocalUpdateIntParameter(last_colot_temp + 1, color_temp_adj);

        rocalCopyToOutput(handle, mat_input.data, h * w * p);

        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        std::string out_filename = std::string(outName) + ".png";   // in case the user specifies non png filename
        if (display_all)
            out_filename = std::string(outName) + std::to_string(index) + ".png";   // in case the user specifies non png filename

        if (color_format == RocalImageColor::ROCAL_COLOR_RGB24)
        {
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            if(DISPLAY)
                cv::imshow("output",mat_output);
            else
                cv::imwrite(out_filename, mat_color, compression_params);
        }
        else
        {
            if(DISPLAY)
            cv::imshow("output",mat_output);
            else
            cv::imwrite(out_filename, mat_output, compression_params);
        }
        col_counter = (col_counter + 1) % number_of_cols;
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalRelease(handle);
    mat_input.release();
    mat_output.release();
    if (!image1)
        return -1;
    return 0;
}
