/*
MIT License

Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#include <string>
#include <string.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include "rali_api.h"

using namespace cv;
using namespace std::chrono;

bool IsPathExist(const char *s)
{
    struct stat buffer;
    return (stat(s, &buffer) == 0);
}

int check_extension(std::string file_name)
{
    // store the position of last '.' in the file name
    int position = file_name.find_last_of(".");
    // store the characters after the '.' from the file_name string
    std::string result = file_name.substr(position + 1);
    if ((result.compare("txt") == 0) || (result.size() == 0) || (result.compare("mp4") == 0))
        return -1;
    return 0;
}

int main(int argc, const char **argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT)
    {
        printf("Usage: rali_video_unittests <video_file/video_dataset_folder/text file> <reader_case> <processing_device=1/cpu=0> <batch_size> <sequence_length> <frame_step> <frame_stride> <gray_scale/rgb> <display_on_off> <shuffle:0/1> <resize_width> <resize_height> <filelist_framenum:0/1> <enable_meta_data:0/1> <enable_framenumber:0/1> <enable_timestamps:0/1> <enable_sequence_rearrange:0/1>\n");
        return -1;
    }

    int argIdx = 0;
    const char *source_path = argv[++argIdx];
    int reader_case = 0;
    bool save_frames = 1; // Saves the frames
    int rgb = 1;      // process color images
    unsigned resize_width = 0;
    unsigned resize_height = 0;
    bool processing_device = 1;
    size_t shard_count = 1;
    bool shuffle = false;
    unsigned input_batch_size = 1;
    unsigned sequence_length = 1;
    unsigned ouput_frames_per_sequence = 1;
    unsigned frame_step = 1;
    unsigned frame_stride = 1;
    bool file_list_frame_num = true;
    bool enable_metadata = false;
    bool enable_framenumbers = false;
    bool enable_timestamps = true;
    bool enable_sequence_rearrange = false;
    bool is_output = true;
    unsigned video_mode = 0;
    // auto decoder_mode = ((video_mode == 1) ? RaliDecodeDevice::RALI_HW_DECODE : RaliDecodeDevice::RALI_SW_DECODE); // Hardware decoder support will be added in future
    auto decoder_mode = RaliDecodeDevice::RALI_SW_DECODE;

    if (argc >= argIdx + MIN_ARG_COUNT)
        reader_case = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        processing_device = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        input_batch_size = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        ouput_frames_per_sequence = sequence_length = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        frame_step = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        frame_stride = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        save_frames = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        shuffle = atoi(argv[++argIdx]) ? true : false;
    if (argc >= argIdx + MIN_ARG_COUNT)
        resize_width = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        resize_height = atoi(argv[++argIdx]);
    if (argc >= argIdx + MIN_ARG_COUNT)
        file_list_frame_num = atoi(argv[++argIdx]) ? true : false;
    if (argc >= argIdx + MIN_ARG_COUNT)
        enable_metadata = atoi(argv[++argIdx]) ? true : false;
    if (argc >= argIdx + MIN_ARG_COUNT)
        enable_framenumbers = atoi(argv[++argIdx]) ? true : false;
    if (argc >= argIdx + MIN_ARG_COUNT)
        enable_timestamps = atoi(argv[++argIdx]) ? true : false;
    if (argc >= argIdx + MIN_ARG_COUNT)
        enable_sequence_rearrange = atoi(argv[++argIdx]) ? true : false;

    if (!IsPathExist(source_path))
    {
        std::cout << "\nThe folder/file path does not exist\n";
        return -1;
    }
    if (enable_sequence_rearrange)
    {
        is_output = false;
    }
    std::cerr << "Batch size : " << input_batch_size << std::endl;
    std::cerr << "Sequence length : " << sequence_length << std::endl;
    std::cerr << "Frame step : " << frame_step << std::endl;
    std::cerr << "Frame stride : " << frame_stride << std::endl;
    if (reader_case == 2)
    {
        std::cerr << "Resize Width : " << resize_width << std::endl;
        std::cerr << "Resize height : " << resize_height << std::endl;
    }

    RaliImageColor color_format = (rgb != 0) ? RaliImageColor::RALI_COLOR_RGB24 : RaliImageColor::RALI_COLOR_U8;
    RaliContext handle;
    handle = raliCreate(input_batch_size, processing_device ? RaliProcessMode::RALI_PROCESS_GPU : RaliProcessMode::RALI_PROCESS_CPU, 0, 1);
    if (raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "Could not create the Rali contex\n";
        return -1;
    }
    if (reader_case == 3)
    {
        if (check_extension(source_path) < 0)
        {
            std::cerr << "\n[ERR]   Text file/ Video File passed as input to SEQUENCE READER\n";
            return -1;
        }
        if (enable_metadata)
        {
            std::cout << "METADATA cannot be enabled for SEQUENCE READER";
            enable_metadata = false;
        }
        if (enable_framenumbers)
            enable_framenumbers = false;
        if (enable_timestamps)
            enable_timestamps = false;
    }
    else if (enable_metadata)
    {
        std::cout << "\n>>>> META DATA READER\n";
        RaliMetaData meta_data = raliCreateVideoLabelReader(handle, source_path, sequence_length, frame_step, frame_stride, file_list_frame_num);
    }

    RaliImage input1;
    switch (reader_case)
    {
        default:
        {
            std::cout << "\n>>>> VIDEO READER\n";
            input1 = raliVideoFileSource(handle, source_path, color_format, decoder_mode, shard_count, sequence_length, shuffle, is_output, false, frame_step, frame_stride, file_list_frame_num);
            break;
        }
        case 2:
        {
            std::cout << "\n>>>> VIDEO READER RESIZE\n";
            if (resize_width == 0 || resize_height == 0)
            {
                std::cerr << "\n[ERR]Resize width and height are passed as NULL values\n";
                return -1;
            }
            input1 = raliVideoFileResize(handle, source_path, color_format, decoder_mode, shard_count, sequence_length, resize_width, resize_height, shuffle, is_output, false, frame_step, frame_stride, file_list_frame_num);
            break;
        }
        case 3:
        {
            std::cout << "\n>>>> SEQUENCE READER\n";
            enable_framenumbers = enable_timestamps = 0;
            input1 = raliSequenceReader(handle, source_path, color_format, shard_count, sequence_length, is_output, shuffle, false, frame_step, frame_stride);
            break;
        }
    }
    if (enable_sequence_rearrange)
    {
        std::cout << "\n>>>> ENABLE SEQUENCE REARRANGE\n";
        unsigned int new_order[] = {0, 0, 1, 1, 0}; // The integers in new order should range only from 0 to sequence_length - 1
        unsigned new_sequence_length = sizeof(new_order) / sizeof(new_order[0]);
        ouput_frames_per_sequence = new_sequence_length;
        input1 = raliSequenceRearrange(handle, input1, new_order, new_sequence_length, sequence_length, true);
    }
    RaliIntParam color_temp_adj = raliCreateIntParameter(0);
    
    // Calling the API to verify and build the augmentation graph
    if (raliGetStatus(handle) != RALI_OK)
    {
        std::cerr << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = raliGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
    }
    
    // Calling the API to verify and build the augmentation graph
    if (raliVerify(handle) != RALI_OK)
    {
        std::cerr << "[ERR]Could not verify the augmentation graph" << std::endl;
        return -1;
    }
    std::cout << "\nRemaining images " << raliGetRemainingImages(handle) << std::endl;
    std::cout << "Augmented copies count " << raliGetAugmentationBranchCount(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    if(save_frames)
        mkdir("output_images", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); // Create directory in which images will be stored
    int h = raliGetAugmentationBranchCount(handle) * raliGetOutputHeight(handle);
    int w = raliGetOutputWidth(handle);
    int p = ((color_format == RaliImageColor::RALI_COLOR_RGB24) ? 3 : 1);
    int single_image_height = h / (input_batch_size * ouput_frames_per_sequence);
    std::cout << "output width " << w << " output height " << h << " color planes " << p << std::endl;
    auto cv_color_format = ((color_format == RaliImageColor::RALI_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color, mat_output;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int counter = 0;
    int color_temp_increment = 1;
    int count = 0;
    while (!raliIsEmpty(handle))
    {
        count++;
        if (raliRun(handle) != 0)
            break;

        if (raliGetIntValue(color_temp_adj) <= -99 || raliGetIntValue(color_temp_adj) >= 99)
            color_temp_increment *= -1;

        raliUpdateIntParameter(raliGetIntValue(color_temp_adj) + color_temp_increment, color_temp_adj);
        raliCopyToOutput(handle, mat_input.data, h * w * p);
        counter += input_batch_size;
        if (save_frames)
        {
            std::string batch_path = "output_images/" + std::to_string(count);
            int status = mkdir(batch_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (status) continue;
            for(unsigned b = 0; b < input_batch_size; b++) // Iterates over each sequence in the batch
            {
		        std::string seq_path = batch_path + "/seq_" + std::to_string(b);
                std::string save_video_path = seq_path + "_output_video.avi" ;

		        int frame_width = static_cast<int>(w); //get the width of frames of the video
                int frame_height = static_cast<int>(single_image_height); //get the height of frames of the video
                Size frame_size(frame_width, frame_height);
                int frames_per_second = 10;

                //Create and initialize the VideoWriter object
                VideoWriter video_writer(save_video_path, VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                                               frames_per_second, frame_size, true);
		        //If the VideoWriter object is not initialized successfully, exit the program
                if (video_writer.isOpened() == false)
                {
                    std::cout << "Cannot save the video to a file" << std::endl;
			        return -1;
                }

                for(unsigned i = 0; i < ouput_frames_per_sequence; i++) // Iterates over the frames in each sequence
                {
                    std::string save_image_path = seq_path + "_output_" + std::to_string(i) + ".png";
		            mat_output=mat_input(cv::Rect(0, ((b * single_image_height * ouput_frames_per_sequence) + (i * single_image_height)), w, single_image_height));
                    if (color_format == RaliImageColor::RALI_COLOR_RGB24)
                    {
                        cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
                        cv::imwrite(save_image_path, mat_color);
			            video_writer.write(mat_color);
                    }
                    else
                    {
                        cv::imwrite(save_image_path, mat_output);
			            video_writer.write(mat_output);
                    }
                }
		        video_writer.release();
            }
        }
        if (enable_metadata)
        {
            int label_id[input_batch_size];
            int image_name_length[input_batch_size];
            raliGetImageLabels(handle, label_id);
            int img_size = raliGetImageNameLen(handle, image_name_length);
            char img_name[img_size];
            raliGetImageName(handle, img_name);

            std::cout << "\nPrinting image names of batch: " << img_name << "\n";
            std::cout << "\t Printing label_id : ";
            for (unsigned i = 0; i < input_batch_size; i++)
            {
                std::cout << label_id[i] << "\t";
            }
            std::cout << std::endl;
        }
        if (enable_framenumbers || enable_timestamps)
        {
            unsigned int start_frame_num[input_batch_size];
            float frame_timestamps[input_batch_size * sequence_length];
            raliGetSequenceStartFrameNumber(handle, start_frame_num);
            if (enable_timestamps)
            {
                raliGetSequenceFrameTimestamps(handle, frame_timestamps);
            }
            for (unsigned i = 0; i < input_batch_size; i++)
            {
                if (enable_framenumbers)
                    std::cout << "\nFrame number : " << start_frame_num[i] << std::endl;
                if (enable_timestamps)
                    for (unsigned j = 0; j < sequence_length; j++)
                        std::cout << "T" << j << " : " << frame_timestamps[(i * sequence_length) + j] << "\t";
                std::cout << "\n";
            }
        }
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rali_timing = raliGetTimingInfo(handle);
    std::cout << "Load     time " << rali_timing.load_time << std::endl;
    std::cout << "Decode   time " << rali_timing.decode_time << std::endl;
    std::cout << "Process  time " << rali_timing.process_time << std::endl;
    std::cout << "Transfer time " << rali_timing.transfer_time << std::endl;
    std::cout << ">>>>> " << counter << " images/frames Processed. Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    raliRelease(handle);
    mat_input.release();
    return 0;
}
