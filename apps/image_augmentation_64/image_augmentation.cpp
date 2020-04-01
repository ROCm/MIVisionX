/*
MIT License

Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
#include <thread>
#include "rali_api.h"


using namespace cv;
#define CV
#define DISPLAY
using namespace std::chrono;


void setof16_mode1(RaliContext handle, RaliImage input_image, size_t h_img, size_t w_img)
{
    auto resized_image = raliResize(handle, input_image, h_img, w_img, true);

    auto warped = raliWarpAffine(handle, input_image,true);

    auto contrast_img = raliContrast(handle, input_image,true);

    auto rain_img = raliRain(handle, input_image, true);

    auto bright_img = raliBrightness(handle, input_image,true);

    auto temp_img = raliColorTemp(handle, input_image, true);

    auto exposed_img = raliExposure(handle, input_image, true);

    auto vignette_img = raliVignette(handle, input_image, true);

    auto blur_img = raliBlur(handle, input_image, true);

    auto snow_img = raliSnow(handle, input_image, true);

    auto pixelate_img = raliPixelate(handle, input_image, true);

    auto snp_img = raliSnPNoise(handle, input_image, true);

    auto gamma_img = raliGamma(handle, input_image, true);

    auto rotate_img = raliRotate(handle, input_image, true);

    auto flip_img = raliFlipFixed(handle, input_image, true, 1);

    auto blend_img = raliBlend(handle, input_image, contrast_img, true);
}

void model_batch_64(RaliContext handle, RaliImage jpg_img, size_t h_img, size_t w_img)
{
    auto input = raliResize(handle, jpg_img, h_img, w_img, false);
            
    auto rot150_img = raliRotateFixed(handle,  input, false, 150);

    auto flip_img = raliFlip(handle,  input, false);

    auto rot45_img = raliRotateFixed(handle, input, false, 45);

    setof16_mode1(handle, input, h_img, w_img);

    setof16_mode1(handle, rot45_img, h_img, w_img);

    setof16_mode1(handle, flip_img, h_img, w_img);

    setof16_mode1(handle, rot150_img , h_img, w_img);
}


int main(int argc, const char ** argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if(argc < MIN_ARG_COUNT) {
        printf( "Usage: image_augmentation <image-dataset-folder> gpu=1/cpu=0 batch_size display-on-off internal_shard_count rgb-on-off decode_max_width decode_max_height \n" );
        return -1;
    }
    int argIdx = 0;
    const char * folderPath1 = argv[++argIdx];

    bool display = 1;// Display the images
    int aug_depth = 1;// how deep is the augmentation tree
    int rgb = 1;// process color images
    int decode_max_width = 0;
    int decode_max_height = 0;
    bool gpu = 1;
    size_t internal_shard_count = 4;
    size_t inputBatchSize = 1;
    if(argc >= argIdx+MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        inputBatchSize = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        display = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        internal_shard_count = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_max_width = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_max_height = atoi(argv[++argIdx]);


    std::cout << ">>> Running on " << (gpu?"GPU":"CPU") << std::endl;

    RaliImageColor color_format = (rgb != 0) ? RaliImageColor::RALI_COLOR_RGB24 : RaliImageColor::RALI_COLOR_U8;

    auto handle = raliCreate(inputBatchSize, gpu?RaliProcessMode::RALI_PROCESS_GPU:RaliProcessMode::RALI_PROCESS_CPU, 0,1);

    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "Could not create the Rali contex\n";
        return -1;
    }



    /*>>>>>>>>>>>>>>>> Creating Rali parameters  <<<<<<<<<<<<<<<<*/

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RaliFloatParam rand_crop_area = raliCreateFloatUniformRand( 0.3, 0.5 );
    RaliIntParam color_temp_adj = raliCreateIntParameter(0);

    // Creating a custom random object to set a limited number of values to randomize the rotation angle
    const size_t num_values = 3;
    float values[num_values] = {0,10,135};
    double frequencies[num_values] = {1, 5, 5};

    RaliFloatParam rand_angle =   raliCreateFloatRand( values , frequencies, num_values);


    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RaliImage input1;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if(decode_max_height <= 0 || decode_max_width <= 0)
        input1 = raliJpegFileSource(handle, folderPath1,  color_format, internal_shard_count, false, false);
    else
        input1 = raliJpegFileSource(handle, folderPath1,  color_format, internal_shard_count, false, false,
                                    RALI_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);

    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "JPEG source could not initialize : "<<raliGetErrorMessage(handle) << std::endl;
        return -1;
    }

    model_batch_64(handle ,input1, 224, 224);


    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = raliGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
    }
    // Calling the API to verify and build the augmentation graph
    if(raliVerify(handle) != RALI_OK)
    {
        std::cout << "Could not verify the augmentation graph" << std::endl;
        return -1;
    }

    std::cout << "Remaining images " << raliGetRemainingImages(handle) << std::endl;

    std::cout << "Augmented copies count " << raliGetAugmentationBranchCount(handle) << std::endl;


    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/

    int h = raliGetAugmentationBranchCount(handle) * raliGetOutputHeight(handle);
    int w = raliGetOutputWidth(handle);
    int p = ((color_format ==  RaliImageColor::RALI_COLOR_RGB24 ) ? 3 : 1);
    const unsigned number_of_cols = 10;
    auto cv_color_format = ((color_format ==  RaliImageColor::RALI_COLOR_RGB24 ) ? CV_8UC3 : CV_8UC1);
    int col_counter = 0;
#ifdef CV
    cv::Mat mat_output(h, w*number_of_cols, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;

    cv::namedWindow( "output", CV_WINDOW_AUTOSIZE );
#endif
    printf("Going to process images\n");

    int color_temp_increment = 1;
    std::vector<int> labels;
    labels.resize(inputBatchSize);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    while (!raliIsEmpty(handle))
    {
        if (raliRun(handle) != 0)
            break;

        if (!display)
            continue;

#if 0
        if(raliGetIntValue(color_temp_adj) <= -99 || raliGetIntValue(color_temp_adj)>=99)
            color_temp_increment *= -1;

        raliUpdateIntParameter(raliGetIntValue(color_temp_adj)+color_temp_increment, color_temp_adj);
#endif
#ifdef CV
        raliCopyToOutput(handle, mat_input.data, h*w*p);
        mat_input.copyTo(mat_output(cv::Rect(  col_counter*w, 0, w, h)));
        if(color_format ==  RaliImageColor::RALI_COLOR_RGB24 )
        {
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            cv::imshow("output",mat_color);
        }
        else
        {
            cv::imshow("output",mat_output);
        }
        cv::waitKey(0);
#endif
        col_counter = (col_counter + 1) % number_of_cols;
    }
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        auto dur = duration_cast<microseconds>( t2 - t1 ).count();
        auto rali_timing = raliGetTimingInfo(handle);
        std::cout << "Load     time "<< rali_timing.load_time << std::endl;
        std::cout << "Decode   time "<< rali_timing.decode_time << std::endl;
        std::cout << "Process  time "<< rali_timing.process_time << std::endl;
        std::cout << "Transfer time "<< rali_timing.transfer_time << std::endl;
        std::cout << ">>>>> Total Elapsed Time " << dur/1000000 << " sec " << dur%1000000 << " us " << std::endl;
        raliResetLoaders(handle);

    raliRelease(handle);
#ifdef CV
    mat_input.release();
    mat_output.release();
#endif
    return 0;
}
