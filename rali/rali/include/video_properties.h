#pragma once
#include <boost/filesystem.hpp>
#include <dirent.h>
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}
#include "commons.h"


namespace filesys = boost::filesystem;

typedef struct video_properties
{
    int width, height, videos_count;
    std::vector<size_t> frames_count;
    std::vector<std::string> video_file_names;
} video_properties;

std::vector<unsigned> open_video_context(const char *video_file_path);

video_properties find_video_properties(const char *source_path);