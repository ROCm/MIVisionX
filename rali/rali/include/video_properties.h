#include <boost/filesystem.hpp>

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}
#include "commons.h"

namespace filesys = boost::filesystem;

typedef struct video_properties{
    int width, height, videos_count;
    std::vector<size_t> frames_count;
    std::vector<std::string> video_file_names;
} video_properties;

std::vector<unsigned> open_video_context(const char *video_file_path)
{
    std::vector<unsigned> video_prop;
    AVFormatContext* pFormatCtx = NULL;
    AVCodecContext* pCodecCtx = NULL;
    int videoStream = -1;
    int i = 0;
    // open video file
    std::cerr << "The video file path : " << video_file_path << "\n";
    int ret = avformat_open_input(&pFormatCtx, video_file_path, NULL, NULL);
    if (ret != 0) {
        std::cerr<<"\nUnable to open video file:"<< video_file_path<<"\n";
        exit(0);
    }

    // Retrieve stream information
    ret = avformat_find_stream_info(pFormatCtx, NULL);
    assert(ret >= 0);

    for(i = 0; i < pFormatCtx->nb_streams; i++) {
        if (pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO && videoStream < 0) {
            videoStream = i;
        }
    } // end for i
    assert(videoStream != -1);

    // Get a pointer to the codec context for the video stream
    pCodecCtx=pFormatCtx->streams[videoStream]->codec;
    assert(pCodecCtx != NULL);
    //std::cerr<<"\n width:: "<<pCodecCtx->width;
    //std::cerr<<"\n height:: "<<pCodecCtx->height;
    video_prop.push_back(pCodecCtx->width);
    video_prop.push_back(pCodecCtx->height);
    video_prop.push_back(pFormatCtx->streams[videoStream]->nb_frames);
    avcodec_close(pCodecCtx);
    avformat_close_input(&pFormatCtx);
    return video_prop;
}


video_properties find_video_properties(const char *source_path)
{
    // based on assumption that user can give single video file or path to folder containing
    // multiple video files.
    // check for videos in the path  is of same resolution. If not throw error and exit.
    video_properties props;
    std::vector<unsigned> video_prop;
    video_prop.resize(3);
    unsigned max_width = 0, max_height = 0;
    {
        std::string _full_path = source_path;
        filesys::path pathObj(_full_path);
        std::cerr<<"\n full path:: "<<_full_path << "\n";
        if(filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) // Single file as input
        {
            video_prop = open_video_context(source_path);
            props.width = video_prop[0];
            props.height = video_prop[1];
            props.videos_count = 1;
            props.frames_count.push_back(video_prop[2]);
            props.video_file_names.push_back(_full_path);
        }
        else
        {
            DIR *_sub_dir;
            struct dirent *_entity;
            std::string _full_path = source_path;
            if ((_sub_dir = opendir (_full_path.c_str())) == nullptr)
                THROW("VideoReader ShardID ERROR: Failed opening the directory at " + source_path);

            std::vector<std::string> video_files;
            unsigned video_count=0;

            while((_entity = readdir(_sub_dir)) != nullptr)
            {
                std::string entry_name(_entity->d_name);
                if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
                video_files.push_back(entry_name);
                ++video_count;
            }
            closedir(_sub_dir);         
            std::sort(video_files.begin(), video_files.end());
            for(int i=0; i < video_count; i++)
            {
                _full_path.append(video_files[i]);
                video_prop = open_video_context(_full_path.c_str());
                if((video_prop[0] > max_width || video_prop[1] > max_height) && (max_width != 0 && max_height != 0))
                {
                    max_width = video_prop[0];
                    std::cerr << "[WARN] The given video files are of different resolution\n";
                }
                if(video_prop[1] > max_height)
                    max_height = video_prop[1];
                
                props.video_file_names.push_back(_full_path);
                props.frames_count.push_back(video_prop[2]);
                _full_path = source_path;
            }
            props.width = max_width;
            props.height = max_height;
            props.videos_count = video_count;
        }
    }
    return props;
}