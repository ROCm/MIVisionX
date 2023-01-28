/*
MIT License

Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#include <vector>

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
#define cvDestroyWindow destroyWindow
#endif

#define DISPLAY 0
using namespace std::chrono;


int main(int argc, const char ** argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if(argc < MIN_ARG_COUNT) {
        printf( "Usage: rocal_dataloader_tf <Folder> <processing_device=1/cpu=0>  <decode_width> <decode_height> <batch_size> <gray_scale/rgb/rgbplanar> display_on_off \n" );
        return -1;
    }
    int argIdx = 0;
    const char * folderPath1 = argv[++argIdx];
    bool display = 0;// Display the images
    //int aug_depth = 1;// how deep is the augmentation tree
    int rgb = 0;// process gray images
    int decode_width = 28;          // mnist data_set
    int decode_height = 28;
    int inputBatchSize = 16;
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
    RocalImageColor color_format = RocalImageColor::ROCAL_COLOR_U8;
    if (rgb == 0)
      color_format = RocalImageColor::ROCAL_COLOR_U8;
    else if (rgb == 1)
      color_format = RocalImageColor::ROCAL_COLOR_RGB24;
    else if (rgb == 2)
      color_format = RocalImageColor::ROCAL_COLOR_RGB_PLANAR;

    auto handle = rocalCreate(inputBatchSize, processing_device?RocalProcessMode::ROCAL_PROCESS_GPU:RocalProcessMode::ROCAL_PROCESS_CPU, 0,1);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }


    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RocalImage input1;
    //hardcoding the following for mnist tfrecords
    std::string feature_map_image = "image/encoded";
    std::string feature_map_filename = "image/filename";
    std::string feature_map_label = "image/class/label";

    rocalCreateTFReader(handle, folderPath1, 1, feature_map_label.c_str(), feature_map_filename.c_str());
    input1 = rocalJpegTFRecordSource(handle, folderPath1, ROCAL_COLOR_RGB24, 1, false, feature_map_image.c_str(), feature_map_filename.c_str(), false, false, ROCAL_USE_USER_GIVEN_SIZE, decode_width, decode_height);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "JPEG source could not initialize : "<<rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

#if 0
    const size_t num_values = 3;
    float values[num_values] = {0,10,135};
    double frequencies[num_values] = {1, 5, 5};

    RocalFloatParam rand_angle =   rocalCreateFloatRand( values , frequencies, num_values);
    // Creating successive blur nodes to simulate a deep branch of augmentations
    RocalImage image2 = rocalCropResize(handle, image0, resize_w, resize_h, false, rand_crop_area);;
    for(int i = 0 ; i < aug_depth; i++)
    {
        image2 = rocalBlurFixed(handle, image2, 17.25, (i == (aug_depth -1)) ? true:false );
    }


    RocalImage image4 = rocalColorTemp(handle, image0, false, color_temp_adj);

    RocalImage image5 = rocalWarpAffine(handle, image4, false);

    RocalImage image6 = rocalJitter(handle, image5, false);

    rocalVignette(handle, image6, true);



    RocalImage image7 = rocalPixelate(handle, image0, false);

    RocalImage image8 = rocalSnow(handle, image0, false);

    RocalImage image9 = rocalBlend(handle, image7, image8, false);

    RocalImage image10 = rocalLensCorrection(handle, image9, false);

    rocalExposure(handle, image10, true);
#else
    // uncomment the following to add augmentation if needed
    RocalImage image0;
    image0 = input1;
    // just do one augmentation to test
    // rocalResize(handle, image0, true);
    rocalResize(handle, image0, 300, 300, true);
#endif

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = rocalGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
    }
    // Calling the API to verify and build the augmentation graph
    if(rocalVerify(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph" << std::endl;
        return -1;
    }

    std::cout << "Remaining images " << rocalGetRemainingImages(handle) << std::endl;

    std::cout << "Augmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int n = rocalGetAugmentationBranchCount(handle);
    int h = n * rocalGetOutputHeight(handle);
    int w = rocalGetOutputWidth(handle);
    int p = (((color_format ==  RocalImageColor::ROCAL_COLOR_RGB24 ) ||
              (color_format ==  RocalImageColor::ROCAL_COLOR_RGB_PLANAR )) ? 3 : 1);
    printf("After get output dims\n");
    std::cout << "output width "<< w << " output height "<< h << " color planes "<< p << std::endl;
    const unsigned number_of_cols = 1;    // no augmented case
    printf("Before memalloc\n");

    auto cv_color_format = ((p==3) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w*number_of_cols, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    int col_counter = 0;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int counter = 0;
    std::vector<std::vector<char>> names;
    std::vector<int> labels;
    names.resize(inputBatchSize);
    labels.resize(inputBatchSize);
    int iter_cnt = 0;
    while (!rocalIsEmpty(handle) && (iter_cnt < 100))
    {
       // if ((iter_cnt %16) == 0)
            printf("Processing iter: %d\n", iter_cnt);
        if(rocalRun(handle) != 0) {
            std::cout << "rocalRun Failed" << std::endl;
            break;
        }

            rocalCopyToOutput(handle, mat_input.data, h*w*p);
        counter += inputBatchSize;
#if 0
        if (processing_device == 1)
            rocalGetImageLabels(handle, labels.data(), ROCAL_MEMCPY_TO_HOST);
        else
            rocalGetImageLabels(handle, labels.data());
        int img_name_size = rocalGetImageNameLen(handle, image_name_length);
        char img_name[img_name_size];
        rocalGetImageName(handle, img_name);
        std::string imageNamesStr(img_name);
        int pos = 0;
        for(int i = 0; i < inputBatchSize; i++)
        {
            names[i] = imageNamesStr.substr(pos, ImageNameLen[i]);
            pos += ImageNameLen[i];
            std::cout << "name: " << names[i] << " label: "<< labels[i] << " - ";
        }
        std::cout << std::endl;
#endif
        iter_cnt ++;
        if(!display)
            continue;

        if(color_format ==  RocalImageColor::ROCAL_COLOR_RGB24 )
        {
            cv::Mat mat_color;
            mat_input.copyTo(mat_output(cv::Rect(  col_counter*w, 0, w, h)));
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            if(DISPLAY)
            cv::imshow("output",mat_output);
            else
            cv::imwrite("output.png",mat_output);
        }
        else if (color_format == RocalImageColor::ROCAL_COLOR_RGB_PLANAR )
        {
            // convert planar to packed for OPENCV
            for (int j = 0; j < n ; j++) {
                int const kWidth = w;
                int const kHeight = rocalGetOutputHeight(handle);
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
            if(DISPLAY)
            cv::imshow("output",mat_output);
            else
            cv::imwrite("output.png",mat_output);
        }
        else
        {
            mat_input.copyTo(mat_output(cv::Rect(  col_counter*w, 0, w, h)));
            if(DISPLAY)
            cv::imshow("output",mat_output);
            else
            cv::imwrite("output.png",mat_output);
        }
        if(DISPLAY)
        cv::waitKey(1);
        col_counter = (col_counter+1)%number_of_cols;
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>( t2 - t1 ).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time "<< rocal_timing.load_time << std::endl;
    std::cout << "Decode   time "<< rocal_timing.decode_time << std::endl;
    std::cout << "Process  time "<< rocal_timing.process_time << std::endl;
    std::cout << "Transfer time "<< rocal_timing.transfer_time << std::endl;
    std::cout << ">>>>> "<< counter << " images/frames Processed. Total Elapsed Time " << dur/1000000 << " sec " << dur%1000000 << " us " << std::endl;
    rocalRelease(handle);
    mat_input.release();
    mat_output.release();
    return 0;
}
