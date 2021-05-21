#include "video_properties.h"

std::vector<unsigned> open_video_context(const char *video_file_path)
{
    std::vector<unsigned> video_prop;
    AVFormatContext *pFormatCtx = NULL;
    AVCodecContext *pCodecCtx = NULL;
    int videoStream = -1;
    unsigned int i = 0;
    // open video file
    // std::cerr << "The video file path : " << video_file_path << "\n";
    int ret = avformat_open_input(&pFormatCtx, video_file_path, NULL, NULL);
    if (ret != 0)
    {
        std::cerr << "\nUnable to open video file:" << video_file_path << "\n";
        exit(0);
    }

    // Retrieve stream information
    ret = avformat_find_stream_info(pFormatCtx, NULL);
    assert(ret >= 0);

    for (i = 0; i < pFormatCtx->nb_streams; i++)
    {
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO && videoStream < 0)
        {
            videoStream = i;
        }
    } // end for i
    assert(videoStream != -1);

    // Get a pointer to the codec context for the video stream
    pCodecCtx = pFormatCtx->streams[videoStream]->codec;
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

    DIR *_sub_dir;
    struct dirent *_entity;

    video_properties props;
    std::vector<unsigned> video_prop;
    video_prop.resize(3);
    unsigned max_width = 0, max_height = 0;
    {
        std::string _full_path = source_path;
        filesys::path pathObj(_full_path);
        if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) // Single file as input
        {
            video_prop = open_video_context(source_path);
            props.width = video_prop[0];
            props.height = video_prop[1];
            props.videos_count = 1;
            props.frames_count.push_back(video_prop[2]);
            props.video_file_names.push_back(_full_path);
        }
        else if (filesys::exists(pathObj) && filesys::is_directory(pathObj))
        {
            //subfolder_reading(source_path, props);

            std::vector<std::string> video_files;
            unsigned video_count = 0;

            std::string _folder_path = source_path;
            if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
                THROW("ERROR: Failed opening the directory at " + _folder_path);

            std::vector<std::string> entry_name_list;
            //std::string _full_path = _folder_path;

            while ((_entity = readdir(_sub_dir)) != nullptr)
            {
                std::string entry_name(_entity->d_name);
                if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                    continue;
                entry_name_list.push_back(entry_name);
            }
            closedir(_sub_dir);
            std::sort(entry_name_list.begin(), entry_name_list.end());

            for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count)
            {
                std::string subfolder_path = _folder_path + "/" + entry_name_list[dir_count];
                // std::cerr << "\nSubfodlerfile/ path :" << subfolder_path.c_str();
                filesys::path pathObj(subfolder_path);
                if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj))
                {
                    video_prop = open_video_context(subfolder_path.c_str());
                    max_width = video_prop[0];
                    max_height = video_prop[1];
                    //props.width = video_prop[0];
                    //props.height = video_prop[1];
                    props.frames_count.push_back(video_prop[2]);
                    props.video_file_names.push_back(subfolder_path);
                    video_count++;
                }
                else if (filesys::exists(pathObj) && filesys::is_directory(pathObj))
                {
                    std::string _full_path = subfolder_path;
                    if ((_sub_dir = opendir(_full_path.c_str())) == nullptr)
                        THROW("VideoReader ERROR: Failed opening the directory at " + source_path);

                    while ((_entity = readdir(_sub_dir)) != nullptr)
                    {
                        std::string entry_name(_entity->d_name);
                        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                            continue;
                        video_files.push_back(entry_name);
                        // std::cerr << "\n  Inside video files : " << entry_name;
                        //++video_count;
                    }
                    closedir(_sub_dir);
                    std::sort(video_files.begin(), video_files.end());
                    for (unsigned i = 0; i < video_files.size(); i++)
                    {
                        std::string file_path = _full_path;
                        file_path.append("/");
                        file_path.append(video_files[i]);
                        _full_path = file_path;

                        // std::cerr << "\n Props file name : " << _full_path;

                        video_prop = open_video_context(_full_path.c_str());
                        if (video_prop[0] > max_width || video_prop[1] > max_height && (max_width != 0 && max_height != 0))
                        {
                            max_width = video_prop[0];
                            std::cerr << "[WARN] The given video files are of different resolution\n";
                        }
                        if (video_prop[1] > max_height)
                            max_height = video_prop[1];

                        props.video_file_names.push_back(_full_path);
                        props.frames_count.push_back(video_prop[2]);
                        video_count++;
                        _full_path = subfolder_path;
                    }
                    //exit(0);
                    video_files.clear();
                }
            }
            props.videos_count = video_count;
            props.width = max_width;
            props.height = max_height;
        }
    }
    return props;
}