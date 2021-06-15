#pragma once
#include <boost/filesystem.hpp>
#include <dirent.h>
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}
#include <sstream>
#include <iostream>
#include <fstream>
#include <tuple>
#include "commons.h"


namespace filesys = boost::filesystem;

typedef struct video_properties
{
    unsigned width, height, videos_count;
    unsigned frame_rate = 0;
    std::vector<size_t> frames_count;
    std::vector<std::string> video_file_names;
    std::vector<std::tuple<int, int>> start_end_frame_num;
    std::vector<std::tuple<float, float>> start_end_timestamps;
    std::vector<int> labels;
} video_properties;

void substring_extraction(std::string const &str, const char delim,  std::vector<std::string> &out);
std::vector<unsigned> open_video_context(const char *video_file_path);
video_properties get_video_properties_from_txt_file(const char *file_path, bool file_list_frame_num);
video_properties find_video_properties(const char *source_path, bool file_list_frame_num);