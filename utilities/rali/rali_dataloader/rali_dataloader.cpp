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
        printf( "Usage: image_augmentation <image_dataset_folder/video_file> <processing_device=1/cpu=0>  decode_width decode_height batch_size gray_scale/rgb/rgbplanar display_on_off \n" );
        return -1;
    }
    int argIdx = 0;
    const char * folderPath1 = argv[++argIdx];
    bool display = 0;// Display the images
    //int aug_depth = 1;// how deep is the augmentation tree
    int rgb = 2;// process color images
    int decode_width = 32;
    int decode_height = 32;
    int inputBatchSize = 4;
    bool processing_device = 1;

    if(argc >= argIdx+MIN_ARG_COUNT)
        processing_device = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_width = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_height = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        inputBatchSize = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        display = atoi(argv[++argIdx]);


    std::cout << ">>> Running on " << (processing_device?"GPU":"CPU") << std::endl;
    RaliImageColor color_format = RaliImageColor::RALI_COLOR_RGB_PLANAR;
    if (rgb == 0) 
      color_format = RaliImageColor::RALI_COLOR_U8;
    else if (rgb == 1)
      color_format = RaliImageColor::RALI_COLOR_RGB24;
    else if (rgb == 2)
      color_format = RaliImageColor::RALI_COLOR_RGB_PLANAR;

    auto handle = raliCreate(inputBatchSize, processing_device?RaliProcessMode::RALI_PROCESS_GPU:RaliProcessMode::RALI_PROCESS_CPU, 0,1);

    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "Could not create the Rali contex\n";
        return -1;
    }


    /*>>>>>>>>>>>>>>>> Creating Rali parameters  <<<<<<<<<<<<<<<<*/

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    //RaliFloatParam rand_crop_area = raliCreateFloatUniformRand( 0.3, 0.5 );
    //RaliIntParam color_temp_adj = raliCreateIntParameter(0);

    // Creating a custom random object to set a limited number of values to randomize the rotation angle
    // create Cifar10 meta data reader
    //raliCreateTextCifar10LabelReader(handle, folderPath1, "data_batch");


    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RaliImage input1;

    input1 = raliRawCIFAR10Source(handle, folderPath1,  color_format, true, decode_width, decode_height, "data_batch_", false);

    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "JPEG source could not initialize : "<<raliGetErrorMessage(handle) << std::endl;
        return -1;
    }
    // create Cifar10 meta data reader
    raliCreateTextCifar10LabelReader(handle, folderPath1, "data_batch");

#if 0
    const size_t num_values = 3;
    float values[num_values] = {0,10,135};
    double frequencies[num_values] = {1, 5, 5};

    RaliFloatParam rand_angle =   raliCreateFloatRand( values , frequencies, num_values);
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
#else
    // uncomment the following to add augmentation if needed
     RaliImage image0;
     image0 = input1;
    // just do one augmentation to test
    // image1 = raliRain(handle, image0, true);
#endif

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
    int n = raliGetAugmentationBranchCount(handle);
    int h = n * raliGetOutputHeight(handle);
    int w = raliGetOutputWidth(handle);
    int p = (((color_format ==  RaliImageColor::RALI_COLOR_RGB24 ) || 
              (color_format ==  RaliImageColor::RALI_COLOR_RGB_PLANAR )) ? 3 : 1);
    std::cout << "output width "<< w << " output height "<< h << " color planes "<< p << std::endl;
    const unsigned number_of_cols = 1;    // no augmented case
    float out_tensor[h*w*p*inputBatchSize];
    auto cv_color_format = ((p==3) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w*number_of_cols, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;
    int col_counter = 0;
    cv::namedWindow( "output", CV_WINDOW_AUTOSIZE );
 
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int counter = 0;
    std::vector<std::string> names;
    std::vector<int> labels;
    int ImageNameLen[inputBatchSize];
    names.resize(inputBatchSize);
    labels.resize(inputBatchSize);
    int iter_cnt = 0;
    float  pmul = 2.0f/255;
    float  padd = -1.0f;
    while (!raliIsEmpty(handle) && (iter_cnt < 100))
    {
        if(raliRun(handle) != 0)
            break;

        if(display)
            raliCopyToOutput(handle, mat_input.data, h*w*p);
        else
            raliCopyToOutputTensor32(handle, out_tensor, RaliTensorLayout::RALI_NCHW, pmul, pmul, pmul, padd, padd, padd, 0);
        counter += inputBatchSize;
        raliGetImageLabels(handle, labels.data());
        unsigned imagename_size = raliGetImageNameLen(handle,ImageNameLen);
        char imageNames[imagename_size]; 
        raliGetImageName(handle,imageNames);
        std::string imageNamesStr(imageNames);

        int pos = 0;
        for(int i = 0; i < inputBatchSize; i++)
        {
            names[i] = imageNamesStr.substr(pos, ImageNameLen[i]);
            pos += ImageNameLen[i];
            std::cout << "name "<< names[i] << " label "<< labels[i] << " - ";
        }
        std::cout << std::endl;
        iter_cnt ++;

        if(!display)
            continue;

        if(color_format ==  RaliImageColor::RALI_COLOR_RGB24 )
        {
            mat_input.copyTo(mat_output(cv::Rect(  col_counter*w, 0, w, h)));
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            cv::imshow("output",mat_color);
        }
        else if (color_format == RaliImageColor::RALI_COLOR_RGB_PLANAR )
        {
            // convert planar to packed for OPENCV
            for (int j = 0; j < n ; j++) {
                int const kWidth = w;
                int const kHeight = raliGetOutputHeight(handle);
                int single_h = kHeight/inputBatchSize;
                for (int n = 0; n<inputBatchSize; n++) {
                    unsigned  channel_size = kWidth*single_h*p;
                    unsigned char *interleavedp = mat_output.data + channel_size*n;
                    unsigned char *planarp = mat_input.data + channel_size*n;
                    for (int i = 0; i < (kWidth * single_h); i++) {
                        interleavedp[i * 3 + 0] = planarp[i + 0 * kWidth * single_h];
                        interleavedp[i * 3 + 1] = planarp[i + 1 * kWidth * single_h];
                        interleavedp[i * 3 + 2] = planarp[i + 2 * kWidth * single_h];
                    }
                }
            }
            cv::imshow("output",mat_output);
        }
        else
        {
            mat_input.copyTo(mat_output(cv::Rect(  col_counter*w, 0, w, h)));
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
