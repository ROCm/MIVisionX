/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include "video_label_reader.h"
#include <string.h>
#include <iostream>
#include <utility>
#include <algorithm>
#include <boost/filesystem.hpp>
#include "commons.h"
#include "exception.h"

using namespace std;

namespace filesys = boost::filesystem;

VideoLabelReader::VideoLabelReader()
{
    _src_dir = nullptr;
    _entity = nullptr;
    _sub_dir = nullptr;
}

void VideoLabelReader::init(const MetaDataConfig &cfg)
{
    _path = cfg.path();
    _output = new LabelBatch();
}
bool VideoLabelReader::exists(const std::string &image_name)
{
    return _map_content.find(image_name) != _map_content.end();
}

void VideoLabelReader::add(std::string image_name, int label, unsigned int video_frame_count, unsigned int start_frame)
{
    std::vector<unsigned> video_prop;
    video_prop = open_video_context(image_name.c_str());
    unsigned frame_count = video_frame_count ? video_frame_count : video_prop[2];
    if (video_frame_count + start_frame > video_prop[2])
        THROW("The given frame numbers in txt file exceeds the maximum frames in the video" + image_name)
    std::vector<std::string> substrings;
    char delim = '/';
    substring_extraction(image_name, delim, substrings);
    std::string file_name = substrings[substrings.size() - 1];
    for (unsigned i = start_frame; i < (start_frame + frame_count); i++)
    {
        pMetaData info = std::make_shared<Label>(label);
        std::string frame_name = std::to_string(_video_idx) + "#" + file_name + "_" + std::to_string(i);
        if (exists(frame_name))
        {
            WRN("Entity with the same name exists")
            return;
        }
        _map_content.insert(pair<std::string, std::shared_ptr<Label>>(frame_name, info));
    }
    _video_idx++;
}

void VideoLabelReader::print_map_contents()
{
    std::cerr << "\nMap contents: \n";
    for (auto &elem : _map_content)
    {
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_label() << std::endl;
    }
}

void VideoLabelReader::release()
{
    _map_content.clear();
}

void VideoLabelReader::release(std::string image_name)
{
    if (!exists(image_name))
    {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void VideoLabelReader::lookup(const std::vector<std::string> &image_names)
{
    if (image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    for (unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Video label reader folders Given name not present in the map" + image_name)
        _output->get_label_batch()[i] = it->second->get_label();
    }
}

void VideoLabelReader::read_text_file(const std::string &_path)
{
    std::ifstream text_file(_path);

    if (text_file.good())
    {
        std::string line;
        int label, start, end;
        float start_time, end_time;
        std::string video_file_name;
        std::vector<unsigned> props;
        while (std::getline(text_file, line))
        {
            start = end = 0;
            std::istringstream line_ss(line);
            if (!(line_ss >> video_file_name >> label))
                continue;
            props = open_video_context(video_file_name.c_str());
            if (!_file_list_frame_num)
            {
                if (line_ss >> start_time)
                {
                    if (line_ss >> end_time)
                    {
                        if (start_time >= end_time)
                        {
                            WRN("Start and end time/frame are not satisfying the condition, skipping the file" + video_file_name)
                            continue;
                        }
                        start = static_cast<int>(std::ceil(start_time * (props[3] / (double)props[4])));
                        end = static_cast<int>(std::floor(end_time * (props[3] / (double)props[4])));
                    }
                }
                end = end != 0 ? end : props[2];
            }
            else
            {
                if (line_ss >> start)
                {
                    if (line_ss >> end)
                    {
                        if (start >= end)
                        {
                            WRN("Start and end time/frame are not satisfying the condition, skipping the file" + video_file_name)
                            continue;
                        }
                        end = end ? end : props[2];
                    }
                }
            }
            add(video_file_name, label, (end - start), start);
        }
    }
    else
        THROW("Can't open the metadata file at " + std::string(_path))
}

void VideoLabelReader::read_all(const std::string &_path)
{
    std::string _folder_path = _path;
    filesys::path pathObj(_folder_path);
    if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj))
    {
        if (pathObj.has_extension() && pathObj.extension().string() == ".txt")
        {
            read_text_file(_path);
        }
        else if (pathObj.has_extension() && pathObj.extension().string() == ".mp4")
        {
            add(_path, 0);
        }
    }
    else
    {
        if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
            THROW("ERROR: Failed opening the directory at " + _folder_path);
        std::vector<std::string> entry_name_list;
        std::string _full_path = _folder_path;
        while ((_entity = readdir(_sub_dir)) != nullptr)
        {
            std::string entry_name(_entity->d_name);
            if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                continue;
            entry_name_list.push_back(entry_name);
        }
        std::sort(entry_name_list.begin(), entry_name_list.end());
        closedir(_sub_dir);
        for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count)
        {
            std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
            filesys::path pathObj(subfolder_path);
            if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj))
            {
                // ignore files with extensions .tar, .zip, .7z
                auto file_extension_idx = subfolder_path.find_last_of(".");
                if (file_extension_idx != std::string::npos)
                {
                    std::string file_extension = subfolder_path.substr(file_extension_idx + 1);
                    if ((file_extension == "tar") || (file_extension == "zip") || (file_extension == "7z") || (file_extension == "rar"))
                        continue;
                }
                read_files(_folder_path);
                for (unsigned i = 0; i < _subfolder_video_file_names.size(); i++)
                {
                    add(_subfolder_video_file_names[i], i);
                }
                break; // assume directory has only files.
            }
            else if (filesys::exists(pathObj) && filesys::is_directory(pathObj))
            {
                _folder_path = subfolder_path;
                _subfolder_video_file_names.clear();
                read_files(_folder_path);
                for (unsigned i = 0; i < _subfolder_video_file_names.size(); i++)
                {
                    std::vector<std::string> substrings;
                    char delim = '/';
                    substring_extraction(_subfolder_video_file_names[i], delim, substrings);
                    int label = atoi(substrings[substrings.size() - 2].c_str());
                    add(_subfolder_video_file_names[i], label);
                }
            }
        }
    }
    // print_map_contents();
}

void VideoLabelReader::read_files(const std::string &_path)
{
    if ((_src_dir = opendir(_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _path);
    while ((_entity = readdir(_src_dir)) != nullptr)
    {
        if (_entity->d_type != DT_REG)
            continue;
        std::string file_path = _path;
        file_path.append("/");
        file_path.append(_entity->d_name);
        _file_names.push_back(file_path);
        _subfolder_video_file_names.push_back(file_path);
    }
    if (_file_names.empty())
        WRN("LabelReader: Could not find any file in " + _path)
    closedir(_src_dir);
    std::sort(_subfolder_video_file_names.begin(), _subfolder_video_file_names.end());
}
