#include "video_properties.h"
#include <cmath>

void substring_extraction(std::string const &str, const char delim, std::vector<std::string> &out)
{
    size_t start;
    size_t end = 0;

    while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
    {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}

std::vector<unsigned> open_video_context(const char *video_file_path)
{
    std::vector<unsigned> props;
    AVFormatContext *pFormatCtx = NULL;
    AVCodecContext *pCodecCtx = NULL;
    int videoStream = -1;
    unsigned int i = 0;
    // open video file
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
    }
    assert(videoStream != -1);

    // Get a pointer to the codec context for the video stream
    pCodecCtx = pFormatCtx->streams[videoStream]->codec;
    assert(pCodecCtx != NULL);
    props.push_back(pCodecCtx->width);
    props.push_back(pCodecCtx->height);
    props.push_back(pFormatCtx->streams[videoStream]->nb_frames);
    props.push_back(pFormatCtx->streams[videoStream]->avg_frame_rate.num);
    props.push_back(pFormatCtx->streams[videoStream]->avg_frame_rate.den);
    avcodec_close(pCodecCtx);
    avformat_close_input(&pFormatCtx);
    return props;
}

video_properties get_video_properties_from_txt_file(const char *file_path, bool file_list_frame_num)
{
    std::ifstream text_file(file_path);

    if (text_file.good())
    {
        //_text_file.open(path.c_str(), std::ifstream::in);
        video_properties video_props;
        std::vector<unsigned> props;
        std::string line;
        int label;
        unsigned int max_width = 0;
        unsigned int max_height = 0;
        unsigned int start, end;
        float start_time, end_time;
        int video_count = 0;
        std::string video_file_name;
        while (std::getline(text_file, line))
        {
            start = end = 0;
            std::istringstream line_ss(line);
            if (!(line_ss >> video_file_name >> label))
                continue;
            props = open_video_context(video_file_name.c_str());
            if (props[0] != max_width)
            {
                if (max_width != 0)
                    std::cerr << "[WARN] The given video files are of different resolution\n";
                max_width = props[0];
            }
            if (props[1] != max_height)
            {
                if (max_height != 0)
                    std::cerr << "[WARN] The given video files are of different resolution\n";
                max_height = props[1];
            }
            if (!file_list_frame_num)
            {
                if (line_ss >> start_time)
                {
                    if (line_ss >> end_time)
                    {
                        if (start_time >= end_time)
                        {
                            std::cerr << "[WRN] Start and end time/frame are not satisfying the condition, skipping the file " << video_file_name << "\n";
                            continue;
                        }
                        start = static_cast<unsigned int>(std::ceil(start_time * (props[3] / (double)props[4])));
                        end = static_cast<unsigned int>(std::floor(end_time * (props[3] / (double)props[4])));;
                    }
                }
                video_props.start_end_timestamps.push_back(std::make_tuple(start_time, end_time));
            }
            else
            {
                if (line_ss >> start)
                {
                    if (line_ss >> end)
                    {
                        if (start >= end)
                        {
                            std::cerr << "[WRN] Start and end time/frame are the same, skipping the file " << video_file_name << "\n";
                            continue;
                        }
                    }
                }
            }
            end = end != 0 ? end : props[2];
            if (end > props[2])
                THROW("The given frame numbers in txt file exceeds the maximum frames in the video" + video_file_name)

            video_file_name = std::to_string(video_count) + "#" + video_file_name;
            video_props.video_file_names.push_back(video_file_name);
            video_props.labels.push_back(label);
            video_props.start_end_frame_num.push_back(std::make_tuple(start, end));
            video_props.frames_count.push_back(end - start);
            video_props.frame_rate.push_back(std::make_tuple(props[3], props[4]));
            video_count++;
        }
        video_props.width = max_width;
        video_props.height = max_height;
        video_props.videos_count = video_count;
        return video_props;
    }
    else
    {
        THROW("Can't open the metadata file at " + std::string(file_path))
    }
}

video_properties find_video_properties(const char *source_path, bool file_list_frame_num)
{
    DIR *_sub_dir;
    struct dirent *_entity;
    std::string video_file_path;
    video_properties video_props;
    std::vector<unsigned> props;
    unsigned int max_width = 0;
    unsigned int max_height = 0;
    std::string _full_path = source_path;
    filesys::path pathObj(_full_path);

    if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) // Single file as input
    {
        if (pathObj.has_extension() && pathObj.extension().string() == ".txt")
        {
            video_props = get_video_properties_from_txt_file(source_path, file_list_frame_num);
        }
        else
        {
            props = open_video_context(source_path);
            video_props.width = props[0];
            video_props.height = props[1];
            video_props.videos_count = 1;
            video_props.frames_count.push_back(props[2]);
            video_props.frame_rate.push_back(std::make_tuple(props[3], props[4]));
            video_props.start_end_frame_num.push_back(std::make_tuple(0, (int)props[2]));
            video_file_path = std::to_string(0) + "#" + _full_path;
            video_props.video_file_names.push_back(video_file_path);
        }
    }
    else if (filesys::exists(pathObj) && filesys::is_directory(pathObj))
    {
        //subfolder_reading(source_path, video_props);
        std::vector<std::string> video_files;
        unsigned video_count = 0;
        std::vector<std::string> entry_name_list;
        std::string _folder_path = source_path;
        if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
            THROW("ERROR: Failed opening the directory at " + _folder_path);

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
            filesys::path pathObj(subfolder_path);
            if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj))
            {
                props = open_video_context(subfolder_path.c_str());
                if (props[0] != max_width)
                {
                    if (max_width != 0)
                        std::cerr << "[WARN] The given video files are of different resolution\n";
                    max_width = props[0];
                }
                if (props[1] != max_height)
                {
                    if (max_height != 0)
                        std::cerr << "[WARN] The given video files are of different resolution\n";
                    max_height = props[1];
                }
                video_props.frames_count.push_back(props[2]);
                video_props.frame_rate.push_back(std::make_tuple(props[3], props[4]));
                video_file_path = std::to_string(video_count) + "#" + subfolder_path;
                video_props.video_file_names.push_back(video_file_path);
                video_props.start_end_frame_num.push_back(std::make_tuple(0, (int)props[2]));
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
                }
                closedir(_sub_dir);
                std::sort(video_files.begin(), video_files.end());
                for (unsigned i = 0; i < video_files.size(); i++)
                {
                    std::string file_path = _full_path;
                    file_path.append("/");
                    file_path.append(video_files[i]);
                    _full_path = file_path;

                    props = open_video_context(_full_path.c_str());
                    if (props[0] != max_width)
                    {
                        if (max_width != 0)
                            std::cerr << "[WARN] The given video files are of different resolution\n";
                        max_width = props[0];
                    }
                    if (props[1] != max_height)
                    {
                        if (max_height != 0)
                            std::cerr << "[WARN] The given video files are of different resolution\n";
                        max_height = props[1];
                    }
                    video_file_path = std::to_string(video_count) + "#" + _full_path;
                    video_props.video_file_names.push_back(video_file_path);
                    video_props.frames_count.push_back(props[2]);
                    video_props.frame_rate.push_back(std::make_tuple(props[3], props[4]));
                    video_props.start_end_frame_num.push_back(std::make_tuple(0, (int)props[2]));
                    video_count++;
                    _full_path = subfolder_path;
                }
                video_files.clear();
            }
        }
        video_props.videos_count = video_count;
        video_props.width = max_width;
        video_props.height = max_height;
    }
    return video_props;
}