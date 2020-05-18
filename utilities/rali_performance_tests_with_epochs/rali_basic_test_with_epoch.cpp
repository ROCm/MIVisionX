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

#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>

#include "rali_api.h"
#define TEST_2
using namespace cv;
using namespace std::chrono;

int main(int argc, const char ** argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if(argc < MIN_ARG_COUNT) {
        std::cout <<  "Usage: ./rali_image_tests_for_epoch <image_dataset_folder>  <test_case:0/1>  <fuction-case> <num_of_epochs> <batch-Size> <processing_device=1/cpu=0>  decode_width decode_height <gray_scale:0/rgb:1> decode_shard_counts \n";
        return -1;
    }
    int argIdx = 0;
    const char * folderPath1 = argv[++argIdx];
    int rgb = 1;// process color images
    int decode_width = 0;
    int decode_height = 0;
    int test_case = 0;
    int function_case = 0;
    int num_of_epochs = 1;
    int batch_size = 1;
    bool processing_device = 0;
    size_t decode_shard_counts = 1;

    if(argc >= argIdx+MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);
    if(argc >= argIdx+MIN_ARG_COUNT)
        function_case = atoi(argv[++argIdx]);    
    if(argc >= argIdx+MIN_ARG_COUNT)
        num_of_epochs = atoi(argv[++argIdx]);
    std::cout << "Number of Epochs is " << num_of_epochs << std::endl;
    if(argc >= argIdx+MIN_ARG_COUNT)
        batch_size = atoi(argv[++argIdx]);
    std::cout << "Batch_size is " << batch_size << std::endl;
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

    test_case = 0; // Should be modified later - Making sure always 0
    int inputBatchSize = batch_size;
    //char function_name[100];
    double avg_load_time = 0, avg_decode_time = 0, avg_process_time = 0, avg_transfer_time = 0;

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
        input1 = raliJpegFileSource(handle, folderPath1,  color_format, decode_shard_counts, false, false, false);
    else
        input1 = raliJpegFileSource(handle, folderPath1,  color_format, decode_shard_counts, false, false, false,
                                    RALI_USE_USER_GIVEN_SIZE, decode_width, decode_height);
    raliCreateLabelReader(handle, folderPath1);
    int resize_w = decode_width, resize_h = decode_height;
    RaliImage image0, image1;
        RaliImage input1_b = raliRotate(handle, input1,false);
    switch (function_case)
    {
    case 0:
            std::cout << "Mixed - Flip, ColorTwist, Crop Resize" << std::endl;
            image0 = raliFlipFixed(handle, input1, 1, false);
            image1 = raliColorTwistFixed(handle, image0, 1.2, 0.4, 1.2, 0.8, false);
            raliCropResizeFixed(handle, image1, 224, 224, true, 0.9, 1.1, 0.1, 0.1 );
            break;
    case 1:
            std::cout << "Gamma Correction - Fixed" << std::endl;
            image0 = raliGammaFixed(handle, input1, 5.0, true);
            break;
    case 2: {
            std::cout << ">>>>>>> Running " << "raliResize" << std::endl;
            auto image_int = raliResize(handle, input1, resize_w / 3, resize_h / 3, false);
            image1 = raliResize(handle, image_int, resize_w, resize_h, true);
             break;
            }
    case 3: {
            std::cout << ">>>>>>> Running " << "raliCropResize" << std::endl;
            image1 = raliCropResize(handle, input1, resize_w, resize_h, true);
        }
    case 4: {
            std::cout << ">>>>>>> Running " << "raliRotate" << std::endl;
            image1 = raliRotate(handle, input1, true);
        }
            break;
    case 5: {
            std::cout << ">>>>>>> Running " << "raliBrightness" << std::endl;
            image1 = raliBrightness(handle, input1, true);
        }
            break;
    case 6: {
            std::cout << ">>>>>>> Running " << "raliContrast" << std::endl;
            image1 = raliContrast(handle, input1, true);
        }
            break;
    case 7: {
            std::cout << ">>>>>>> Running " << "raliFlip Horizontal" << std::endl;
            image1 = raliFlip(handle, input1, true);
        }
            break;
    case 8: {
            std::cout << ">>>>>>> Running " << "raliBlur" << std::endl;
            image1 = raliBlur(handle, input1, true);
        }
            break;
    case 9: {
            std::cout << ">>>>>>> Running " << "raliBlend" << std::endl;
            image1 = raliBlend(handle, input1, input1_b, true);
        }
            break;
    
    case 10: {
            std::cout << ">>>>>>> Running " << "raliWarpAffine" << std::endl;
           image1 = raliWarpAffine(handle, input1, true);
        }
        break;
    case 11: {
            std::cout << ">>>>>>> Running " << "raliFishEye" << std::endl;
            image1 = raliFishEye(handle, input1, true);
        }
        break;
    case 12: {
            std::cout << ">>>>>>> Running " << "raliVignette" << std::endl;
            image1 = raliVignette(handle, input1, true);
        }
        break;
    case 13: {
            std::cout << ">>>>>>> Running " << "raliJitter" << std::endl;
            image1 = raliJitter(handle, input1, true);
        }
        break;
    case 14: {
            std::cout << ">>>>>>> Running " << "raliSnPNoise" << std::endl;
            image1 = raliSnPNoise(handle, input1, true);
        }
        break;
    case 15: {
            std::cout << ">>>>>>> Running " << "raliSnow" << std::endl;
            image1 = raliSnow(handle, input1, true);
        }
        break;
    case 16: {
            std::cout << ">>>>>>> Running " << "raliRain" << std::endl;
            image1 = raliRain(handle, input1, true);
        }
        break;
            
    case 17: {
            std::cout << ">>>>>>> Running " << "raliColorTemp" << std::endl;
            image1 = raliColorTemp(handle, input1, true);
        }
        break;
    case 18: {
            std::cout << ">>>>>>> Running " << "raliFog" << std::endl;
            image1 = raliFog(handle, input1, true);
        }
        break;
    case 19: {
            std::cout << ">>>>>>> Running " << "raliLensCorrection" << std::endl;
            image1 = raliLensCorrection(handle, input1, true);
        }
        break;
    case 20: {
            std::cout << ">>>>>>> Running " << "raliPixelate" << std::endl;
            image1 = raliPixelate(handle, input1, true);
        }
        break;
    case 21: {
            std::cout << ">>>>>>> Running " << "raliExposure" << std::endl;
            image1 = raliExposure(handle, input1, true);
        }
        break;
    case 22:{
            std::cout << ">>>>>>> Running " << "raliColorTwistBatch" << std::endl;
            image1 = raliColorTwist(handle, input1, true);
        }
        break;
	case 23: {
                std::cout << ">>>>>>> Running " << "raliHue" << std::endl;
                image1 = raliHue(handle, input1, true);
            }
            break;
	case 24: {
                std::cout << ">>>>>>> Running " << "raliSaturation" << std::endl;
                image1 = raliSaturation(handle, input1, true);
        }
	    break;
	case 25: {
                resize_w = 250;
                resize_h = 300;
                std::cout << ">>>>>>> Running " << "raliCrop Fixed Corner" << std::endl;
                image1 = raliCropFixed(handle, input1, resize_w, resize_h, 5,  true, 0.5, 0.5, 0.5);
        }
        break;

    case 26: {
                resize_w = 250;
                resize_h = 300;
                std::cout << ">>>>>>> a " << "raliCropFixed Centric" << std::endl;
                image1 = raliCropCenterFixed(handle, input1, resize_w, resize_h, 0, true);
        }
        break;
    case 27: {
                std::cout << ">>>>>>> Running " << "raliCrop Random" << std::endl;
                image1 = raliCrop(handle, input1, true);
        }
        break;
        
    default:
        break;
        
    }
    
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

    const int total_tests = num_of_epochs;
    int test_id = -1;
    int run_len[] = {2*inputBatchSize,4*inputBatchSize,1*inputBatchSize, 50*inputBatchSize};

    std::vector<std::vector<char>> names;
    std::vector<int> labels;
    names.resize(inputBatchSize);
    labels.resize(inputBatchSize);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    while( ++test_id < total_tests)
    {
        std::cout << "#### Started Epoch id " << test_id <<"\n";
        std::cout << "Available images = " << raliGetRemainingImages(handle) << std::endl;
        int porcess_image_count = ((test_case == 0) ? raliGetRemainingImages(handle) : run_len[test_id]);
        std::cout << ">>>>> Going to process " << porcess_image_count << " images , press a key" << std::endl;
        // cv::waitKey(20);
        const unsigned number_of_cols =  porcess_image_count/inputBatchSize;
        cv::Mat mat_output(h, w*number_of_cols, cv_color_format);
        cv::Mat mat_input(h, w, cv_color_format);
        cv::Mat mat_color;
        // auto win_name = "output";
        //cv::namedWindow( win_name, CV_WINDOW_AUTOSIZE );
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
                //cv::imshow(win_name, mat_color);
            } else {
                //cv::imshow(win_name, mat_output);
            }
            // The delay here simulates possible latency between runs due to training
            //cv::waitKey(200);
            col_counter = (col_counter + 1) % number_of_cols;
        }
        std::cout << ">>>>> Done test id " << test_id << " processed " << counter << " images ,press a key \n";
       // cv::waitKey(20);
          
        
       
        //std::cout << "#### Going to reset\n";
        
        raliResetLoaders(handle);
        mat_input.release();
        mat_output.release();
        mat_color.release();
        //cvDestroyWindow(win_name);
        //std::cout << "#### Done reset\n";
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    auto rali_timing = raliGetTimingInfo(handle);
    
    avg_load_time   += rali_timing.load_time;
    avg_decode_time += rali_timing.decode_time;
    avg_process_time += rali_timing.process_time;
    avg_transfer_time += rali_timing.transfer_time;
    std::cout << std::endl;
    std::cout << "Load     time " << rali_timing.load_time << std::endl;
    std::cout << "Decode   time " << rali_timing.decode_time << std::endl;
    std::cout << "Process  time " << rali_timing.process_time << std::endl;
    std::cout << "Transfer time " << rali_timing.transfer_time << std::endl << std::endl;;

    std::cout << "Average Load     time " << (avg_load_time / num_of_epochs * 1.0) << std::endl;
    std::cout << "Average Decode   time " << (avg_decode_time / num_of_epochs * 1.0) << std::endl;
    std::cout << "Average Process  time " << (avg_process_time / num_of_epochs * 1.0) << std::endl;
    std::cout << "Average Transfer time " << (avg_transfer_time / num_of_epochs * 1.0) << std::endl;

    raliRelease(handle);

    return 0;
}

