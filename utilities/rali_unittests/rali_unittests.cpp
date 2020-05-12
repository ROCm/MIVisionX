/*
MIT License

Copyright (c) 2018 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <vector>

#include "rali_api.h"

using namespace cv;

#define DISPLAY
using namespace std::chrono;

int test(int test_case, const char* path, const char* outName, int rgb, int gpu, int width, int height);
int main(int argc, const char ** argv)
{
    // check command-line usage
    const size_t MIN_ARG_COUNT = 2;
    if(argc < MIN_ARG_COUNT){
   	 printf( "Usage: rali_unittests <image-dataset-folder> output_image_name <width> <height> test_case gpu=1/cpu=0 rgb=1/grayscale=0  \n" );
        return -1;
    }

    int argIdx = 0;
    const char * path = argv[++argIdx];
    const char * outName = argv[++argIdx];
    int width = atoi(argv[++argIdx]);
    int height = atoi(argv[++argIdx]);

    int rgb = 1;// process color images
    bool gpu = 1;
    int test_case = 3; // For Rotate 

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);
    
    test(test_case, path, outName, rgb, gpu, width, height);

    return 0;
}

int test(int test_case, const char* path, const char* outName, int rgb, int gpu, int width, int height)
{
    size_t num_threads = 1;
    int inputBatchSize = 1;
    int decode_max_width = 0;
    int decode_max_height = 0;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << " , "<< (rgb ? " Color ":" Grayscale ")<< std::endl;

    RaliImageColor color_format = (rgb != 0) ? RaliImageColor::RALI_COLOR_RGB24
                                             : RaliImageColor::RALI_COLOR_U8;

    auto handle = raliCreate(inputBatchSize,
                             gpu ? RaliProcessMode::RALI_PROCESS_GPU : RaliProcessMode::RALI_PROCESS_CPU, 0,
                             1);

    if (raliGetStatus(handle) != RALI_OK) {
        std::cout << "Could not create the Rali contex\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>> Creating Rali parameters  <<<<<<<<<<<<<<<<*/

    raliSetSeed(0);

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RaliFloatParam rand_crop_area = raliCreateFloatUniformRand(0.3, 0.5);
    RaliIntParam color_temp_adj = raliCreateIntParameter(-50);

    // Creating a custom random object to set a limited number of values to randomize the rotation angle
    const size_t num_values = 3;
    float values[num_values] = {0, 10, 135};
    double frequencies[num_values] = {1, 5, 5};
    RaliFloatParam rand_angle = raliCreateFloatRand(values, frequencies,
                                                    sizeof(values) / sizeof(values[0]));
    
    //num_values = 2;
    int new_values[2] = { 0, 1};
    double new_freq[2] = {40, 60};
    RaliIntParam rand_mirror = raliCreateIntRand(new_values, new_freq, 2);

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/

    RaliMetaData meta_data = raliCreateLabelReader(handle, path);
    
    RaliImage input1;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if (decode_max_height <= 0 || decode_max_width <= 0)
        input1 = raliJpegFileSource(handle, path, color_format, num_threads, false, false);
    else
        input1 = raliJpegFileSource(handle, path, color_format, num_threads, false, false,
                                    RALI_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);

    if (raliGetStatus(handle) != RALI_OK) {
        std::cout << "JPEG source could not initialize : " << raliGetErrorMessage(handle) << std::endl;
        return -1;
    }


    int resize_w = width, resize_h = height; // height and width

    //RaliImage image0 = input1;
    RaliImage image0 = raliResize(handle, input1, resize_w, resize_h, false);// input1; 
    RaliImage image0_b = raliRotate(handle, image0,false);

    RaliFlipAxis axis_h = RALI_FLIP_HORIZONTAL;
    RaliFlipAxis axis_v = RALI_FLIP_VERTICAL;

    RaliImage image1;

    switch (test_case) {
        case 0: {
            std::cout << ">>>>>>> Running " << "raliResize" << std::endl;
            auto image_int = raliResize(handle, image0, resize_w / 3, resize_h / 3, false);
            image1 = raliResize(handle, image_int, resize_w, resize_h, true);
        }
            break;
        case 1: {
            std::cout << ">>>>>>> Running " << "raliCropResize" << std::endl;
            image1 = raliCropResize(handle, input1, resize_w, resize_h, true);
        }
            break;
        case 2: {
            std::cout << ">>>>>>> Running " << "raliRotate" << std::endl;
            image1 = raliRotate(handle, image0, true);
        }
            break;
        case 3: {
            std::cout << ">>>>>>> Running " << "raliBrightness" << std::endl;
            image1 = raliBrightness(handle, image0, true);
        }
            break;
        case 4: {
            std::cout << ">>>>>>> Running " << "raliGamma" << std::endl;
            image1 = raliGamma(handle, image0, true);
        }
            break;
        case 5: {
            std::cout << ">>>>>>> Running " << "raliContrast" << std::endl;
            image1 = raliContrast(handle, image0, true);
        }
            break;
        case 6: {
            std::cout << ">>>>>>> Running " << "raliFlip" << std::endl;
            image1 = raliFlip(handle, image0, true);
        }
            break;
        case 7: {
            std::cout << ">>>>>>> Running " << "raliBlur" << std::endl;
            image1 = raliBlur(handle, image0, true);
        }
            break;
        case 8: {
            std::cout << ">>>>>>> Running " << "raliBlend" << std::endl;
            image1 = raliBlend(handle, image0, image0_b, true);
        }
            break;
        case 9: {
            std::cout << ">>>>>>> Running " << "raliWarpAffine" << std::endl;
           image1 = raliWarpAffine(handle, image0, true);
        }
            break;
        case 10: {
            std::cout << ">>>>>>> Running " << "raliFishEye" << std::endl;
            image1 = raliFishEye(handle, image0, true);
        }
            break;
        case 11: {
            std::cout << ">>>>>>> Running " << "raliVignette" << std::endl;
            image1 = raliVignette(handle, image0, true);
        }
            break;
        case 12: {
            std::cout << ">>>>>>> Running " << "raliJitter" << std::endl;
            image1 = raliJitter(handle, image0, true);
        }
            break;
        case 13: {
            std::cout << ">>>>>>> Running " << "raliSnPNoise" << std::endl;
            image1 = raliSnPNoise(handle, image0, true);
        }
            break;
        case 14: {
            std::cout << ">>>>>>> Running " << "raliSnow" << std::endl;
            image1 = raliSnow(handle, image0, true);
        }
            break;
        case 15: {
            std::cout << ">>>>>>> Running " << "raliRain" << std::endl;
            image1 = raliRain(handle, image0, true);
        }
            break;
        case 16: {
            std::cout << ">>>>>>> Running " << "raliColorTemp" << std::endl;
            image1 = raliColorTemp(handle, image0, true);
        }
            break;
        case 17: {
            std::cout << ">>>>>>> Running " << "raliFog" << std::endl;
            image1 = raliFog(handle, image0, true);
        }
            break;
        case 18: {
            std::cout << ">>>>>>> Running " << "raliLensCorrection" << std::endl;
            image1 = raliLensCorrection(handle, image0, true);
        }
            break;
        case 19: {
            std::cout << ">>>>>>> Running " << "raliPixelate" << std::endl;
            image1 = raliPixelate(handle, image0, true);
        }
            break;
        case 20: {
            std::cout << ">>>>>>> Running " << "raliExposure" << std::endl;
            image1 = raliExposure(handle, image0, true);
        }
        case 21: {
            std::cout << ">>>>>>> Running " << "raliHue" << std::endl;
            image1 = raliHue(handle, image0, true);
        }
        case 22: {
            std::cout << ">>>>>>> Running " << "raliSaturation" << std::endl;
            image1 = raliSaturation(handle, image0, true);
        }
            break;
        case 23: {
            std::cout << ">>>>>>> Running " << "raliCopy" << std::endl;
            image1 = raliCopy(handle, image0, true);
        }
            break;
        case 24: {
            std::cout << ">>>>>>> Running " << "raliColorTwist" << std::endl;
            image1 = raliColorTwist(handle, image0, true);
        }
            break;
	case 25: {
            std::cout << ">>>>>>> Running " << "raliCropMirrorNormalize" << std::endl;
	    std::vector<float> mean;
	    std::vector<float> std_dev;
            image1 = raliCropMirrorNormalize(handle, image0, 1, 200, 200, 50, 50, 1, mean, std_dev, true);
        }
            break;
	case 26: {
            std::cout << ">>>>>>> Running " << "raliCrop" << std::endl;
            image1 = raliCrop(handle, image0, true);
        }
            break;
	case 27: {
            std::cout << ">>>>>>> Running " << "raliResizeCropMirror" << std::endl;
            image1 = raliResizeCropMirror(handle, image0, resize_w, resize_h, true);
        }
            break;



        case 30: {
             std::cout << ">>>>>>> Running " << "raliCropResizeFixed" << std::endl;
             image1 = raliCropResizeFixed(handle, image0, resize_w, resize_h, true,  0.25, 1.2, 0.6, -0.4);
         }
             break;
	case 31: {
            std::cout << ">>>>>>> Running " << "raliRotateFixed" << std::endl;
            image1 = raliRotateFixed(handle, image0, 45, true);
        }
            break;
        case 32: {
            std::cout << ">>>>>>> Running " << "raliBrightnessFixed" << std::endl;
            image1 = raliBrightnessFixed(handle, image0, 1.90, 20, true);
        }
            break;
        case 33: {
            std::cout << ">>>>>>> Running " << "raliGammaFixed" << std::endl;
            image1 = raliGammaFixed(handle, image0, 0.5, true);
        }
            break;
        case 34: {
            std::cout << ">>>>>>> Running " << "raliContrastFixed" << std::endl;
            image1 = raliContrastFixed(handle, image0, 30, 80, true);
        }
            break;
        case 35: {
           std::cout << ">>>>>>> Running " << "raliBlurFixed" << std::endl;
           image1 = raliBlurFixed(handle, image0, 5, true);
        }
            break;
        case 36: {
            std::cout << ">>>>>>> Running " << "raliBlendFixed" << std::endl;
            image1 = raliBlendFixed(handle, image0, image0_b, 0.5, true);
        }
            break;
        case 37: {
            std::cout << ">>>>>>> Running " << "raliWarpAffineFixed" << std::endl;
            image1 = raliWarpAffineFixed(handle, image0, 0.25, 0.25, 1, 1, 5, 5, true);
        }
            break;
        case 38: {
            std::cout << ">>>>>>> Running " << "raliVignetteFixed" << std::endl;
            image1 = raliVignetteFixed(handle, image0, 50 , true);
        }
            break;
        case 39: {
            std::cout << ">>>>>>> Running " << "raliJitterFixed" << std::endl;
            image1 = raliJitterFixed(handle, image0, 3, true);
        }
            break;
        case 40: {
            std::cout << ">>>>>>> Running " << "raliSnPNoiseFixed" << std::endl;
            image1 = raliSnPNoiseFixed(handle, image0, 0.12, true);
        }
            break;
        case 41: {
            std::cout << ">>>>>>> Running " << "raliSnowFixed" << std::endl;
            image1 = raliSnowFixed(handle, image0, 0.2, true);
        }
            break;
        case 42: {
            std::cout << ">>>>>>> Running " << "raliRainFixed" << std::endl;
            image1 = raliRainFixed(handle, image0, 0.5, 2, 16, 0.25, true);
        }
            break;
        case 43: {
            std::cout << ">>>>>>> Running " << "raliColorTempFixed" << std::endl;
            image1 = raliColorTempFixed(handle, image0, 70, true);
        }
            break;
        case 44: {
            std::cout << ">>>>>>> Running " << "raliFogFixed" << std::endl;
            image1 = raliFogFixed(handle, image0, 0.5, true);
        }
            break;
        case 45: {
            std::cout << ">>>>>>> Running " << "raliLensCorrectionFixed" << std::endl;
            image1 = raliLensCorrectionFixed(handle, image0, 2.9, 1.2, true);
        }
            break;
        case 46: {
            std::cout << ">>>>>>> Running " << "raliExposureFixed" << std::endl;
            image1 = raliExposureFixed(handle, image0, 1, true);
        }
            break;
        case 47: {
            std::cout << ">>>>>>> Running " << "raliFlipFixed" << std::endl;
            image1 = raliFlipFixed(handle, image0, 2, true);
        }
            break;
	case 48: {
            std::cout << ">>>>>>> Running " << "raliHueFixed" << std::endl;
            image1 = raliHueFixed(handle, image0, 150, true);
        }
            break;
        case 49: {
            std::cout << ">>>>>>> Running " << "raliSaturationFixed" << std::endl;
            image1 = raliSaturationFixed(handle, image0, 0.3, true);
        }
            break;
        case 50: {
            std::cout << ">>>>>>> Running " << "raliColorTwistFixed" << std::endl;
            image1 = raliColorTwistFixed(handle, image0, 0.2, 10.0, 100.0, 0.25, true);
        }
            break;
        case 51: {
            std::cout << ">>>>>>> Running " << "raliCropFixed" << std::endl;
            image1 = raliCropFixed(handle, image0, 100, 100, 1, true, 25, 25, 2);
        }
            break;
        case 52: {
            std::cout << ">>>>>>> Running " << "raliCropCenterFixed" << std::endl;
            image1 = raliCropCenterFixed(handle, image0, 100, 100, 2, true);
        }
            break;
        case 53: {
            std::cout << ">>>>>>> Running " << "raliResizeCropMirrorFixed" << std::endl;
            image1 = raliResizeCropMirrorFixed(handle, image0, 100, 100, true, 50, 50, 0);
        }
            break;
	
	default:
            std::cout << "Not a valid option! Exiting!\n";
            return -1;
    }

    // Calling the API to verify and build the augmentation graph
    raliVerify(handle);

    if (raliGetStatus(handle) != RALI_OK) {
        std::cout << "Could not verify the augmentation graph " << raliGetErrorMessage(handle);
        return -1;
    }

    printf("\n\nAugmented copies count %d \n", raliGetAugmentationBranchCount(handle));


    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = raliGetAugmentationBranchCount(handle) * raliGetOutputHeight(handle);
    int w = raliGetOutputWidth(handle);
    int p = ((color_format == RaliImageColor::RALI_COLOR_RGB24) ? 3 : 1);
    const unsigned number_of_cols = 1;//1920 / w;
    auto cv_color_format = ((color_format == RaliImageColor::RALI_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;
    int col_counter = 0;
    //cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    printf("Going to process images\n");
    printf("Remaining images %d \n", raliGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    while (raliGetRemainingImages(handle) >= inputBatchSize) {
        if (raliRun(handle) != 0)
            break;
        
        char img_name[50];
        int label_id[inputBatchSize];     
        raliGetImageLabels(handle, label_id);
        for(int i = 0; i < inputBatchSize; i++) {
            raliGetImageName(handle, img_name, i);
            std::cerr << "\nPrinting image name : " << img_name<<"\t Printing label_id : " << label_id[i] << std::endl;
        }

        auto last_colot_temp = raliGetIntValue(color_temp_adj);
        raliUpdateIntParameter(last_colot_temp + 1, color_temp_adj);

        raliCopyToOutput(handle, mat_input.data, h * w * p);

        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        if (color_format == RaliImageColor::RALI_COLOR_RGB24) {
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            cv::imwrite(outName, mat_color, compression_params);
        } else {
            cv::imwrite(outName, mat_output, compression_params);

        }
        col_counter = (col_counter + 1) % number_of_cols;
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rali_timing = raliGetTimingInfo(handle);
    std::cout << "Load     time " << rali_timing.load_time << std::endl;
    std::cout << "Decode   time " << rali_timing.decode_time << std::endl;
    std::cout << "Process  time " << rali_timing.process_time << std::endl;
    std::cout << "Transfer time " << rali_timing.transfer_time << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    raliRelease(handle);
    mat_input.release();
    mat_output.release();

    return 0;
}
