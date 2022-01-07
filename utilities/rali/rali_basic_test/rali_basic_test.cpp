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

#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>

#include "rali_api.h"
#define TEST_2
using namespace cv;

int main(int argc, const char ** argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if(argc < MIN_ARG_COUNT) {
        std::cout <<  "Usage: image_augmentation <image_dataset_folder> <label_text_file_path> <test_case:0/1> <processing_device=1/cpu=0>  decode_width decode_height <gray_scale:0/rgb:1> decode_shard_counts \n";
        return -1;
    }
    int argIdx = 0;
    const char * folderPath1 = argv[++argIdx];
    const char * label_text_file_path = argv[++argIdx];
    int rgb = 1;// process color images
    int decode_width = 0;
    int decode_height = 0;
    int test_case = 0;
    bool processing_device = 0;
    size_t decode_shard_counts = 1;

    if(argc >= argIdx+MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        processing_device = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_width = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_height = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_shard_counts = atoi(argv[++argIdx]);


    int inputBatchSize = 4;

    std::cout << ">>> Running on " << (processing_device?"GPU":"CPU") << std::endl;

    RaliImageColor color_format = (rgb != 0) ? RaliImageColor::RALI_COLOR_RGB24 : RaliImageColor::RALI_COLOR_U8;

    auto handle = raliCreate(inputBatchSize, processing_device?RaliProcessMode::RALI_PROCESS_GPU:RaliProcessMode::RALI_PROCESS_CPU, 0,1);

    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "Could not create the Rali contex\n";
        return -1;
    }

       /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RaliImage input1;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if(decode_height <= 0 || decode_width <= 0)
        input1 = raliJpegFileSource(handle, folderPath1,  color_format, decode_shard_counts, false, false);
    else
        input1 = raliJpegFileSource(handle, folderPath1,  color_format, decode_shard_counts, false, false,
                                    RALI_USE_USER_GIVEN_SIZE, decode_width, decode_height);
    if(strcmp(label_text_file_path, "") == 0)
        raliCreateLabelReader(handle, folderPath1);
    else
        raliCreateTextFileBasedLabelReader(handle, label_text_file_path);

    auto image0 = raliFlipFixed(handle, input1, 1, false);
    auto image1 = raliColorTwistFixed(handle, image0, 1.2, 0.4, 1.2, 0.8, false);
    raliCropResizeFixed(handle, image1, 224, 224, true, 0.9, 1.1, 0.1, 0.1 );

    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "JPEG source could not initialize : "<<raliGetErrorMessage(handle) << std::endl;
        return -1;
    }

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

    std::cout << "Augmented copies count " << raliGetAugmentationBranchCount(handle) << std::endl;


    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = raliGetAugmentationBranchCount(handle) * raliGetOutputHeight(handle);
    int w = raliGetOutputWidth(handle);
    int p = ((color_format ==  RaliImageColor::RALI_COLOR_RGB24 ) ? 3 : 1);
    std::cout << "output width "<< w << " output height "<< h << " color planes "<< p << std::endl;
    auto cv_color_format = ((color_format ==  RaliImageColor::RALI_COLOR_RGB24 ) ? CV_8UC3 : CV_8UC1);

    const int total_tests = 4;
    int test_id = -1;
    int run_len[] = {2*inputBatchSize,4*inputBatchSize,1*inputBatchSize, 50*inputBatchSize};

    std::vector<std::vector<char>> names;
    std::vector<int> labels;
    names.resize(inputBatchSize);
    labels.resize(inputBatchSize);

    while( ++test_id < total_tests)
    {
        std::cout << "#### Started test id " << test_id <<"\n";
        std::cout << "Available images = " << raliGetRemainingImages(handle) << std::endl;
        int porcess_image_count = ((test_case == 0) ? raliGetRemainingImages(handle) : run_len[test_id]);
        std::cout << ">>>>> Going to process " << porcess_image_count << " images , press a key" << std::endl;
        cv::waitKey(0);
        const unsigned number_of_cols =  porcess_image_count/inputBatchSize;
        cv::Mat mat_output(h, w*number_of_cols, cv_color_format);
        cv::Mat mat_input(h, w, cv_color_format);
        cv::Mat mat_color;
        auto win_name = "output";
        cv::namedWindow( win_name, CV_WINDOW_AUTOSIZE );

        int col_counter = 0;
        int counter = 0;

        while((test_case == 0) ? !raliIsEmpty(handle) : (counter < run_len[test_id]))
        {
            if (raliRun(handle) != 0)
                break;

            raliCopyToOutput(handle, mat_input.data, h * w * p);

            counter += inputBatchSize;
            raliGetImageLabels(handle, labels.data());
            for(int i = 0; i < inputBatchSize; i++)
            {
                names[i] = std::move(std::vector<char>(raliGetImageNameLen(handle, 0), '\n'));
                raliGetImageName(handle, names[i].data(), i);
                std::string id(names[i].begin(), names[i].end());
                std::cout << "name "<< id << " label "<< labels[i] << " - ";
            }
            std::cout << std::endl;

            mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));

            if (color_format == RaliImageColor::RALI_COLOR_RGB24) {
                cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
                cv::imshow(win_name, mat_color);
            } else {
                cv::imshow(win_name, mat_output);
            }
            // The delay here simulates possible latency between runs due to training
            cv::waitKey(200);
            col_counter = (col_counter + 1) % number_of_cols;
        }
        std::cout << ">>>>> Done test id " << test_id << " processed " << counter << " images ,press a key \n";
        cv::waitKey(0);
        std::cout << "#### Going to reset\n";
        raliResetLoaders(handle);
        mat_input.release();
        mat_output.release();
        mat_color.release();
        cvDestroyWindow(win_name);
        std::cout << "#### Done reset\n";
    }

    raliRelease(handle);

    return 0;
}
