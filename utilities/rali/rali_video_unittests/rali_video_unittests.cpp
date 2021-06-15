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

#define DISPLAY
using namespace std::chrono;

bool IsPathExist(const char *s)
{
    struct stat buffer;
    return (stat(s, &buffer) == 0);
}

int check_extension(std::string file_name)
{
    //store the position of last '.' in the file name
    int position = file_name.find_last_of(".");
    //store the characters after the '.' from the file_name string
    std::string result = file_name.substr(position + 1);
    if ((result.compare("txt") != 0) || (result.size() == 0))
        return -1;
    return 0;
}

int main(int argc, const char **argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT)
    {
        printf("Usage: rali_video_unittests <video_file/video_dataset_folder/text file> <video_reader_case> <processing_device=1/cpu=0> <batch_size> <sequence_length> <frame_step> <frame_stride> <gray_scale/rgb> <display_on_off> <shuffle:0/1> <decode_width> <decode_height> <filelist_framenum:0/1> <enable_meta_data:0/1> <enable_framenumber:0/1> <enable_timestamps:0/1> <enable_sequence_rearrange:0/1>\n");
        return -1;
    }

    int argIdx = 0;
    const char *source_path = argv[++argIdx];
    int video_reader_case = 0;
    bool display = 1; // Display the images
    int rgb = 1;      // process color images
    unsigned decode_width = 0;
    unsigned decode_height = 0;
    bool processing_device = 1;
    size_t shard_count = 1;
    bool shuffle = false;
    unsigned input_batch_size = 1;
    unsigned sequence_length = 1;
    unsigned frame_step = 1;
    unsigned frame_stride = 1;
    bool file_list_frame_num = true;
    bool enable_metadata = false;
    bool enable_framenumbers = false;
    bool enable_timestamps = true;
    bool enable_sequence_rearrange = false;
    bool is_output = true;

    if (argc >= argIdx + MIN_ARG_COUNT)
        video_reader_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        processing_device = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        input_batch_size = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        sequence_length = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        frame_step = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        frame_stride = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        display = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        shuffle = atoi(argv[++argIdx]) ? true : false;

    if (argc >= argIdx + MIN_ARG_COUNT)
        decode_width = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        decode_height = atoi(argv[++argIdx]);

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

    if (video_reader_case == 3)
    {
        if (check_extension(source_path) == 0)
        {
            std::cerr << "\n[ERR]Text file passed as input to SEQUENCE READER\n";
            return -1;
        }
    }

    if (enable_sequence_rearrange)
    {
        is_output = false;
    }

    std::cerr << "Batch size : " << input_batch_size << std::endl;
    std::cerr << "Sequence length : " << sequence_length << std::endl;
    std::cerr << "Frame step : " << frame_step << std::endl;
    std::cerr << "Frame stride : " << frame_stride << std::endl;
    std::cerr << "Decode Width : " << decode_width << std::endl;
    std::cerr << "Decode height : " << decode_height << std::endl;

    RaliImageColor color_format = (rgb != 0) ? RaliImageColor::RALI_COLOR_RGB24 : RaliImageColor::RALI_COLOR_U8;
    RaliContext handle;
    handle = raliCreate(input_batch_size, processing_device ? RaliProcessMode::RALI_PROCESS_GPU : RaliProcessMode::RALI_PROCESS_CPU, 0, 1);
    if (raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "Could not create the Rali contex\n";
        return -1;
    }

    if (enable_metadata)
    {
        if (video_reader_case == 5)
        {
            std::cout << "METADATA READER cannot be enabled for SEQUENCE READER";
            enable_metadata = false;
        }
        else
        {
            std::cout << "\n>>>> META DATA READER\n";
            RaliMetaData meta_data = raliCreateVideoLabelReader(handle, source_path, file_list_frame_num);
        }
    }

    RaliImage input1;

    switch (video_reader_case)
    {
    default:
    {
        std::cout << "\n>>>> VIDEO READER\n";
        input1 = raliVideoFileSource(handle, source_path, color_format, shard_count, sequence_length, frame_step, frame_stride, shuffle, is_output, false, file_list_frame_num);
        break;
    }
    case 2:
    {
        std::cout << "\n>>>> VIDEO READER RESIZE\n";
        if (decode_width <= 0 || decode_height <= 0)
        {
            std::cerr << "\n[ERR]Decoded width and height passed as NULL values\n";
            return -1;
        }
        input1 = raliVideoFileResize(handle, source_path, color_format, shard_count, sequence_length, frame_step, frame_stride, decode_width, decode_height, shuffle, is_output, false, file_list_frame_num);
        break;
    }
    case 3:
    {
        std::cout << "\n>>>> SEQUENCE READER\n";
        enable_framenumbers = enable_timestamps = 0;
        input1 = raliSequenceReader(handle, source_path, color_format, shard_count, sequence_length, frame_step, frame_stride, is_output, shuffle, false, RALI_USE_USER_GIVEN_SIZE, decode_width, decode_height);
        break;
    }
    }

    if (enable_sequence_rearrange)
    {
        std::cout << "\n>>>> ENABLE SEQUENCE REARRANGE\n";
        unsigned int new_order[] = {0, 0, 1, 1, 0}; // The integers in new order should range only from 0 to sequence_length - 1
        unsigned new_sequence_length = sizeof(new_order) / sizeof(new_order[0]);
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
    int h = raliGetAugmentationBranchCount(handle) * raliGetOutputHeight(handle);
    int w = raliGetOutputWidth(handle);
    int p = ((color_format == RaliImageColor::RALI_COLOR_RGB24) ? 3 : 1);
    std::cout << "output width " << w << " output height " << h << " color planes " << p << std::endl;
    const unsigned number_of_cols = 1;
    auto cv_color_format = ((color_format == RaliImageColor::RALI_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w * number_of_cols, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;
    int col_counter = 0;
    // if (display)
    //     cv::namedWindow( "output", CV_WINDOW_AUTOSIZE );

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
        if (!display)
            continue;

        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        if (color_format == RaliImageColor::RALI_COLOR_RGB24)
        {
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            cv::imwrite("output_" + std::to_string(count) + ".png", mat_color);
        }
        else
        {
            cv::imwrite("output.png", mat_output);
        }
        // cv::waitKey(1);
        if (enable_metadata)
        {
            int label_id[input_batch_size * sequence_length];
            int image_name_length[input_batch_size * sequence_length];
            raliGetImageLabels(handle, label_id);
            int img_size = raliGetImageNameLen(handle, image_name_length);
            char img_name[img_size];
            raliGetImageName(handle, img_name);

            std::cout << "\nPrinting image names of batch: " << img_name << "\n";
            for (int i = 0; i < 1; i++)
            {
                std::cout << "\t Printing label_id : " << label_id[i * sequence_length] << std::endl;
                std::cout << "\n";
            }
        }
        // Add API to get size;
        if(enable_framenumbers || enable_timestamps)
        {
            unsigned int start_frame_num[input_batch_size];
            float frame_timestamps[input_batch_size * sequence_length];
            raliGetSequenceStartFrameNumber(handle, start_frame_num);
            if(enable_timestamps)
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
        col_counter = (col_counter + 1) % number_of_cols;
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
    mat_output.release();
    return 0;
}