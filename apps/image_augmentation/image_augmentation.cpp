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

#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>

#include "rali_api.h"

using namespace cv;

#define DISPLAY
using namespace std::chrono;


int main(int argc, const char ** argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if(argc < MIN_ARG_COUNT) {
        printf( "Usage: image_augmentation <image_dataset_folder/video_file> <processing_device=1/cpu=0>  decode_width decode_height video_mode gray_scale/rgb display_on_off decode_shard_count  <shuffle:0/1> <jpeg_dec_mode<0(tjpeg)/1(opencv)>\n" );
        return -1;
    }
    int argIdx = 0;
    const char * folderPath1 = argv[++argIdx];
    int video_mode = 0;// 0 means no video decode, 1 means hardware, 2 means software decoding
    bool display = 1;// Display the images
    int aug_depth = 1;// how deep is the augmentation tree
    int rgb = 1;// process color images
    int decode_width = 0;
    int decode_height = 0;
    bool processing_device = 1;
    size_t shard_count = 2;
    int shuffle = 0;
    int dec_mode = 0;

    if(argc >= argIdx+MIN_ARG_COUNT)
        processing_device = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_width = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_height = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        video_mode = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        display = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        shard_count = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        shuffle = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        dec_mode = atoi(argv[++argIdx]);

    int inputBatchSize = 2;

    std::cout << ">>> Running on " << (processing_device?"GPU":"CPU") << std::endl;

    RaliImageColor color_format = (rgb != 0) ? RaliImageColor::RALI_COLOR_RGB24 : RaliImageColor::RALI_COLOR_U8;

    auto handle = raliCreate(inputBatchSize, processing_device?RaliProcessMode::RALI_PROCESS_GPU:RaliProcessMode::RALI_PROCESS_CPU, 0,1);

    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "Could not create the rocAL contex\n";
        return -1;
    }

    RaliDecoderType dec_type = (dec_mode==0)? RaliDecoderType::RALI_DECODER_TJPEG : RaliDecoderType::RALI_DECODER_OPENCV;

    /*>>>>>>>>>>>>>>>> Creating rocAL parameters  <<<<<<<<<<<<<<<<*/

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


    if(video_mode != 0)
    {
        if (decode_height <= 0 || decode_width <= 0)
        {
            std::cout << "Output width and height is needed for video decode\n";
            return -1;
        }
        input1 = raliVideoFileSource(handle, folderPath1, color_format, ((video_mode == 1) ? RaliDecodeDevice::RALI_HW_DECODE:RaliDecodeDevice::RALI_SW_DECODE)
                , true, decode_width, decode_height, false);
    }
    else
    {
	    // The jpeg file loader can automatically select the best size to decode all images to that size
         // User can alternatively set the size or change the policy that is used to automatically find the size
         if (dec_type == RaliDecoderType::RALI_DECODER_OPENCV) std::cout << "Using OpenCV decoder for Jpeg Source\n";
         if(decode_height <= 0 || decode_width <= 0)
             input1 = raliJpegFileSource(handle, folderPath1,  color_format, shard_count, false, shuffle, false);
        else
             input1 = raliJpegFileSource(handle, folderPath1,  color_format, shard_count, false, shuffle, false,  RALI_USE_USER_GIVEN_SIZE, decode_width, decode_height, dec_type);

    }

    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "JPEG source could not initialize : "<<raliGetErrorMessage(handle) << std::endl;
        return -1;
    }

    RaliImage image0;
    int resize_w = 112, resize_h = 112;
    if(video_mode)
    {
        resize_h = decode_height;
        resize_w = decode_width;
        image0 = input1;
    }
    else
    {
        image0 = raliResize(handle, input1, resize_w, resize_h, true);
    }
    RaliImage image1 = raliRain(handle, image0, false);

    RaliImage image11 = raliFishEye(handle, image1, false);

    raliRotate(handle, image11, true, rand_angle);

    // Creating successive blur nodes to simulate a deep branch of augmentations
    RaliImage image2 = raliCropResize(handle, image0, resize_w, resize_h, false, rand_crop_area);;
    for(int i = 0 ; i < aug_depth; i++)
    {
        image2 = raliBlurFixed(handle, image2, 17.25, (i == (aug_depth -1)) ? true:false );
    }

    RaliImage image4 = raliColorTemp(handle, image0, false, color_temp_adj);

    RaliImage image5 = raliWarpAffine(handle, image4, false);

    RaliImage image6 = raliJitter(handle, image5, false);

    raliVignette(handle, image6, true);

    RaliImage image7 = raliPixelate(handle, image0, false);

    RaliImage image8 = raliSnow(handle, image0, false);

    RaliImage image9 = raliBlend(handle, image7, image8, false);

    RaliImage image10 = raliLensCorrection(handle, image9, false);

    raliExposure(handle, image10, true);

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
    std::cout << "output width "<< w << " output height "<< h << " color planes "<< p << std::endl;
    const unsigned number_of_cols = video_mode ? 1 : 10;
    auto cv_color_format = ((color_format ==  RaliImageColor::RALI_COLOR_RGB24 ) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w*number_of_cols, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;
    int col_counter = 0;
    if (display)
        cv::namedWindow( "output", CV_WINDOW_AUTOSIZE );

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int counter = 0;
    int color_temp_increment = 1;
    while (!raliIsEmpty(handle))
    {
        if(raliRun(handle) != 0)
            break;

        if(raliGetIntValue(color_temp_adj) <= -99 || raliGetIntValue(color_temp_adj)>=99)
            color_temp_increment *= -1;

        raliUpdateIntParameter(raliGetIntValue(color_temp_adj)+color_temp_increment, color_temp_adj);

        raliCopyToOutput(handle, mat_input.data, h*w*p);
        counter += inputBatchSize;
        if(!display)
            continue;

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
        cv::waitKey(1);
        col_counter = (col_counter+1)%number_of_cols;
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>( t2 - t1 ).count();
    auto rali_timing = raliGetTimingInfo(handle);
    std::cout << "Load     time "<< rali_timing.load_time << std::endl;
    std::cout << "Decode   time "<< rali_timing.decode_time << std::endl;
    std::cout << "Process  time "<< rali_timing.process_time << std::endl;
    std::cout << "Transfer time "<< rali_timing.transfer_time << std::endl;
    std::cout << ">>>>> "<< counter << " images/frames Processed. Total Elapsed Time " << dur/1000000 << " sec " << dur%1000000 << " us " << std::endl;
    raliRelease(handle);
    mat_input.release();
    mat_output.release();
    return 0;
}
