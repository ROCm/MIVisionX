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
#include <iostream>
#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>

#include "rali_api.h"

using namespace cv;

#define DISPLAY
using namespace std::chrono;
std::mutex g_mtx;           // mutex for critical section

int thread_func(const char *path, int gpu_mode, RaliImageColor color_format, int shard_id, int num_shards, int dec_width, int dec_height, int batch_size, bool shuffle, bool display )
{
    std::unique_lock<std::mutex> lck (g_mtx,std::defer_lock);
    std::cout << ">>> Running on " << (gpu_mode>=0?"GPU":"CPU") << std::endl;
    color_format = RaliImageColor::RALI_COLOR_RGB24;
    int gpu_id = (gpu_mode < 0)? 0: gpu_mode;

    lck.lock();
    //looks like OpenVX has some issue loading kernels from multiple threads at the same time
    auto handle = raliCreate(batch_size, (gpu_mode<0)?RaliProcessMode::RALI_PROCESS_CPU:RaliProcessMode::RALI_PROCESS_GPU, gpu_id,1);
    lck.unlock();
    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "Could not create the Rali context" << "<" << shard_id << num_shards << " >" << std::endl;
        return -1;
    }
    std::cout << "RALI created context for " << "<" << shard_id << num_shards << " >" << std::endl;
    // create JPEG data loader based on numshards and shard_id
    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    RaliImage input1;
    if(dec_width <= 0 || dec_height <= 0)
        input1 = raliJpegFileSourceSingleShard(handle, path, color_format, shard_id, num_shards, false, shuffle, false);
    else
        input1 = raliJpegFileSourceSingleShard(handle, path, color_format, shard_id, num_shards, false,
                                shuffle, false,  RALI_USE_USER_GIVEN_SIZE_RESTRICTED, dec_width, dec_height);

    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "raliJpegFileSourceSingleShard<"<<shard_id<<" , "<< num_shards << ">" << " could not initialize : "<<raliGetErrorMessage(handle) << std::endl;
        return -1;
    }
    // create meta data reader
    raliCreateLabelReader(handle, path);

    /*>>>>>>>>>>>>>>>> Creating Rali parameters  <<<<<<<<<<<<<<<<*/

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    //RaliFloatParam rand_crop_area = raliCreateFloatUniformRand( 0.3, 0.5 );
    //RaliIntParam color_temp_adj = raliCreateIntParameter(0);

    // uncomment the following to add augmentation if needed
    //image0 = input1;
    raliResize(handle, input1, 224, 224, true);

    if(raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = raliGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
        return  -1;
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
    std::cout << "output width "<< w << " output height "<< h << " color planes "<< p << " n "<< n << std::endl;
    const unsigned number_of_cols = 1;    // no augmented case
    float *out_tensor =new float[h*w*p+256];
  //  printf("Allocated output tensor of size(flat) %d\n", h*w*p+256);
    auto cv_color_format = ((p==3) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w*number_of_cols, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;
    int col_counter = 0;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int counter = 0;
    std::vector<std::vector<char>> names;
    std::vector<int> labels;
    names.resize(batch_size);
    labels.resize(batch_size);
#if 1
    //cv::namedWindow( "output", CV_WINDOW_AUTOSIZE );
    int iter_cnt = 0;
    float  pmul = 2.0f/255;
    float  padd = -1.0f;
    while (!raliIsEmpty(handle) /*&& (iter_cnt < 100)*/)
    {
      //  std::cout << "processing iter: " << iter_cnt << std::endl;
        if(raliRun(handle) != 0)
            break;

        if(display)
            raliCopyToOutput(handle, mat_input.data, h*w*p);
        else
            raliCopyToOutputTensor32(handle, out_tensor, RaliTensorLayout::RALI_NCHW, pmul, pmul, pmul, padd, padd, padd, 0);
        counter += batch_size;
        raliGetImageLabels(handle, labels.data());
        for(int i = 0; i < batch_size; i++)
        {
            names[i] = std::move(std::vector<char>(raliGetImageNameLen(handle, 0), '\n'));
            raliGetImageName(handle, names[i].data(), i);
            std::string id(names[i].begin(), names[i].end());
         //   std::cout << "name "<< id << " label "<< labels[i] << " - ";
        }
       // std::cout << std::endl;
        iter_cnt ++;

        if(!display)
            continue;
        mat_input.copyTo(mat_output(cv::Rect(  col_counter*w, 0, w, h)));
        cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
        //cv::imshow("output",mat_color);
        //cv::waitKey(1);
        col_counter = (col_counter+1)%number_of_cols;
    }
#endif
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>( t2 - t1 ).count();
    auto rali_timing = raliGetTimingInfo(handle);
    std::cout << "Load     time:"<< shard_id<<" " <<rali_timing.load_time << std::endl;
    std::cout << "Decode   time:"<< shard_id<<" " <<rali_timing.decode_time << std::endl;
    std::cout << "Process  time: "<< shard_id<<" " <<rali_timing.process_time << std::endl;
    std::cout << "Transfer time: "<< shard_id<<" " <<rali_timing.transfer_time << std::endl;
    std::cout << ">>>>> "<< counter << " images/frames Processed. Total Elapsed Time " << dur/1000000 << " sec " << dur%1000000 << " us " << std::endl;
    raliRelease(handle);
    mat_input.release();
    mat_output.release();
    if (out_tensor) delete [] out_tensor;
    return 0;
}

int main(int argc, const char ** argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT) {
        printf("Usage: rali_dataloader_mt <image_dataset_folder/video_file> <num_gpus>1(gpu)/cpu=0>  num_shards, decode_width decode_height batch_size shuffle display_on_off \n");
        return -1;
    }
    int argIdx = 0;
    const char *path = argv[++argIdx];
    bool display = 0;// Display the images
    //int aug_depth = 1;// how deep is the augmentation tree
    int decode_width = 1024;
    int decode_height = 1024;
    int inputBatchSize = 16;
    int num_shards = 2;
    bool shuffle = 0;
    int num_gpus = 0;

    if (argc >= argIdx + MIN_ARG_COUNT)
        num_gpus = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        num_shards = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        decode_width = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        decode_height = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        inputBatchSize = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        shuffle = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        display = atoi(argv[++argIdx]);

    // launch threads process shards
    std::thread loader_threads[num_shards];
    auto gpu_id = num_gpus ? 0 : -1;
    int th_id;
    for (th_id = 0; th_id < num_shards; th_id++) {
        loader_threads[th_id] = std::thread(thread_func, path, gpu_id, RaliImageColor::RALI_COLOR_RGB24, th_id, num_shards, decode_width, decode_height, inputBatchSize,
                                                shuffle, display);
        if (num_gpus) gpu_id = (gpu_id +1) % num_gpus;
    }
    for (auto& th:loader_threads ) {
        th.join();
    }
}
