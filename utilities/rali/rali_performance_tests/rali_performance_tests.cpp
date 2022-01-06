/*
MIT License

Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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


int test(int test_case, const char* path, int rgb, int processing_device, int width, int height, int batch_size, int shards, int shuffle);
int main(int argc, const char ** argv)
{
    // check command-line usage
    const size_t MIN_ARG_COUNT = 2;
    printf( "Usage: rali_performance_tests <image-dataset-folder> <width> <height> <test_case> <batch_size> <gpu=1/cpu=0> <rgb=1/grayscale=0> <shard_count>  <shuffle=1>\n" );
    if(argc < MIN_ARG_COUNT)
        return -1;

    int argIdx = 0;
    const char * path = argv[++argIdx];
    int width = atoi(argv[++argIdx]);
    int height = atoi(argv[++argIdx]);

    int rgb = 1;// process color images
    bool processing_device = 1;
    int test_case = 0;
    int batch_size = 10;
    int shards = 4;
    int shuffle = 0;

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        batch_size = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        processing_device = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
	shards = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
	shuffle = atoi(argv[++argIdx]);

    test(test_case, path, rgb, processing_device, width, height, batch_size, shards, shuffle);

    return 0;
}

int test(int test_case, const char* path, int rgb, int processing_device, int width, int height, int batch_size, int shards, int shuffle)
{
    size_t num_threads = shards;
    int inputBatchSize = batch_size;
    int decode_max_width = 0;
    int decode_max_height = 0;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (processing_device ? "GPU" : "CPU") << " , "<< (rgb ? " Color ":" Grayscale ")<< std::endl;
    printf(">>> Batch size = %d -- shard count = %d\n", inputBatchSize, num_threads);

    RaliImageColor color_format = (rgb != 0) ? RaliImageColor::RALI_COLOR_RGB24 : RaliImageColor::RALI_COLOR_U8;

    auto handle = raliCreate(inputBatchSize, processing_device ? RaliProcessMode::RALI_PROCESS_GPU : RaliProcessMode::RALI_PROCESS_CPU, 0, 1);

    if (raliGetStatus(handle) != RALI_OK) {
        std::cout << "Could not create the Rali context\n";
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


    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RaliImage image0;
    RaliImage image0_b;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if (decode_max_height <= 0 || decode_max_width <= 0)
        image0 = raliJpegFileSource(handle, path, color_format, num_threads, false, shuffle, true);
    else
        image0 = raliJpegFileSource(handle, path, color_format, num_threads, false, shuffle, false,
                                    RALI_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);

    if (raliGetStatus(handle) != RALI_OK) {
        std::cout << "JPEG source could not initialize : " << raliGetErrorMessage(handle) << std::endl;
        return -1;
    }


    int resize_w = width, resize_h = height;

    RaliFlipAxis axis_h = RALI_FLIP_HORIZONTAL;
    RaliFlipAxis axis_v = RALI_FLIP_VERTICAL;

    RaliImage image1;

    switch (test_case) {
        case 0: {
            std::cout << ">>>>>>> Running " << "raliResize" << std::endl;
            raliResize(handle, image0, resize_w, resize_h, true);
        }
            break;
        case 1: {
            std::cout << ">>>>>>> Running " << "raliCropResize" << std::endl;
            raliCropResize(handle, image0, resize_w, resize_h, true, rand_crop_area);
        }
            break;
        case 2: {
            std::cout << ">>>>>>> Running " << "raliRotate" << std::endl;
            raliRotate(handle, image0, true, rand_angle);
        }
            break;
        case 3: {
            std::cout << ">>>>>>> Running " << "raliBrightness" << std::endl;
            raliBrightness(handle, image0, true);
        }
            break;
        case 4: {
            std::cout << ">>>>>>> Running " << "raliGamma" << std::endl;
            raliGamma(handle, image0, true);
        }
            break;
        case 5: {
            std::cout << ">>>>>>> Running " << "raliContrast" << std::endl;
            raliContrast(handle, image0, true);
        }
            break;
        case 6: {
            std::cout << ">>>>>>> Running " << "raliFlip" << std::endl;
            raliFlip(handle, image0, true);
        }
            break;
        case 7: {
            std::cout << ">>>>>>> Running " << "raliBlur" << std::endl;
            raliBlur(handle, image0, true);
        }
            break;
        case 8: {
            std::cout << ">>>>>>> Running " << "raliBlend" << std::endl;
            image0_b = raliRotateFixed(handle, image0, 30, false);
            raliBlend(handle, image0, image0_b, true);
        }
            break;
        case 9: {
            std::cout << ">>>>>>> Running " << "raliWarpAffine" << std::endl;
            raliWarpAffine(handle, image0, true);
        }
            break;
        case 10: {
            std::cout << ">>>>>>> Running " << "raliFishEye" << std::endl;
            raliFishEye(handle, image0, true);
        }
            break;
        case 11: {
            std::cout << ">>>>>>> Running " << "raliVignette" << std::endl;
            raliVignette(handle, image0, true);
        }
            break;
        case 12: {
            std::cout << ">>>>>>> Running " << "raliJitter" << std::endl;
            raliJitter(handle, image0, true);
        }
            break;
        case 13: {
            std::cout << ">>>>>>> Running " << "raliSnPNoise" << std::endl;
            raliSnPNoise(handle, image0, true);
        }
            break;
        case 14: {
            std::cout << ">>>>>>> Running " << "raliSnow" << std::endl;
            raliSnow(handle, image0, true);
        }
            break;
        case 15: {
            std::cout << ">>>>>>> Running " << "raliRain" << std::endl;
            raliRain(handle, image0, true);
        }
            break;
        case 16: {
            std::cout << ">>>>>>> Running " << "raliColorTemp" << std::endl;
            raliColorTemp(handle, image0, true, color_temp_adj);
        }
            break;
        case 17: {
            std::cout << ">>>>>>> Running " << "raliFog" << std::endl;
            raliFog(handle, image0, true);
        }
            break;
        case 18: {
            std::cout << ">>>>>>> Running " << "raliLensCorrection" << std::endl;
            raliLensCorrection(handle, image0, true);
        }
            break;
        case 19: {
            std::cout << ">>>>>>> Running " << "raliPixelate" << std::endl;
            raliPixelate(handle, image0, true);
        }
            break;
        case 20: {
            std::cout << ">>>>>>> Running " << "raliExposure" << std::endl;
            raliExposure(handle, image0, true);
        }
            break;
        case 21: {
            std::cout << ">>>>>>> Running " << "raliHue" << std::endl;
            raliHue(handle, image0, true);
        }
            break;
        case 22: {
            std::cout << ">>>>>>> Running " << "raliSaturation" << std::endl;
            raliSaturation(handle, image0, true);
        }
            break;
        case 23: {
            std::cout << ">>>>>>> Running " << "raliCopy" << std::endl;
            raliCopy(handle, image0, true);
        }
            break;
        case 24: {
            std::cout << ">>>>>>> Running " << "raliColorTwist" << std::endl;
            raliColorTwist(handle, image0, true);
        }
            break;
        case 25: {
            std::cout << ">>>>>>> Running " << "raliCropMirrorNormalize" << std::endl;
	    std::vector<float> mean;
	    std::vector<float> std_dev;
            raliCropMirrorNormalize(handle, image0, 3, 200, 200, 50, 50, 1, mean, std_dev, true);
        }
            break;
        case 26: {
            std::cout << ">>>>>>> Running " << "raliCrop " << std::endl;
            raliCrop(handle, image0, true);
        }
            break;
        case 27: {
            std::cout << ">>>>>>> Running " << "raliResizeCropMirror" << std::endl;
            raliResizeCropMirror(handle, image0, resize_w, resize_h, true);
        }
            break;
        case 28: {
            std::cout << ">>>>>>> Running " << "No-Op" << std::endl;
            raliNop(handle, image0, true);
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



    printf("Augmented copies count %d\n", raliGetAugmentationBranchCount(handle));



    printf("Going to process images\n");
//    printf("Remaining images %d \n", raliGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    int i = 0;
    while (i++ < 100 && !raliIsEmpty(handle)){  
        
        if (raliRun(handle) != 0)
            break;

        //auto last_colot_temp = raliGetIntValue(color_temp_adj);
        //raliUpdateIntParameter(last_colot_temp + 1, color_temp_adj);

        
        //raliCopyToOutput(handle, mat_input.data, h * w * p);

    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rali_timing = raliGetTimingInfo(handle);
    std::cout << "Load     time " << rali_timing.load_time << std::endl;
    std::cout << "Decode   time " << rali_timing.decode_time << std::endl;
    std::cout << "Process  time " << rali_timing.process_time << std::endl;
    std::cout << "Transfer time " << rali_timing.transfer_time << std::endl;
    std::cout << "Total time " << dur << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    
    raliRelease(handle);
 

    return 0;
}
