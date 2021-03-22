/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cassert>
#include <algorithm>
#include <commons.h>
#include "video_reader.h"
#include <boost/filesystem.hpp>
namespace filesys = boost::filesystem;

#ifdef RALI_VIDEO

VideoReader::VideoReader():
_shuffle_time("shuffle_time", DBG_TIMING)
{
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _file_id = 0;
    _shuffle = false;
    _file_count_all_shards = 0;
}

unsigned VideoReader::count()
{
    if(_loop)
        return _video_file_names.size();

    int ret = ((int)_video_file_names.size() -_read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status VideoReader::initialize(ReaderConfig desc)
{
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _folder_path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    ret = subfolder_reading();
    std::cerr << "\n\n Reading video files ...";
    _video_file_count = _video_file_names.size();
    // the following code is required to make every shard the same size:: required for multi-gpu training
    if (_shard_count > 1 && _batch_count > 1) { // check needed
        int _num_batches = _video_file_names.size()/_batch_count; // check needed
        int max_batches_per_shard = (_file_count_all_shards + _shard_count-1)/_shard_count; // check needed
        max_batches_per_shard = (max_batches_per_shard + _batch_count-1)/_batch_count; // check needed
        if (_num_batches < max_batches_per_shard) { // check needed
            replicate_last_batch_to_pad_partial_shard(); // check needed
        }
    }
    if( ret==Reader::Status::OK && _shuffle)
        std::random_shuffle(_video_file_names.begin(), _video_file_names.end());
    return ret;
}

void VideoReader::incremenet_read_ptr()
{
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _video_file_names.size();
}

size_t VideoReader::open()
{
    // opening file is not needed for decoding video, since the low level decoder directly handles it
    return 0;
}

size_t VideoReader::read(unsigned char* buf, size_t read_size)
{
    auto file_path = _video_file_names[_curr_file_idx];// Get next file name
    incremenet_read_ptr();
    _current_fPtr = fopen(file_path.c_str(), "rb");// Open the file,
    if (!_current_fPtr || read_size < file_path.size())
        return 0; 
    fclose(_current_fPtr);
    _current_fPtr = nullptr;
    // for video instead of reading the file, retun the filename instead
    strcpy((char *)buf, file_path.c_str());
    return file_path.size();
}


int VideoReader::close()
{
    return 0;
}

VideoReader::~VideoReader()
{
    //release();
}

void VideoReader::reset()
{
    if (_shuffle) std::random_shuffle(_video_file_names.begin(), _video_file_names.end());
    _read_counter = 0;
    _curr_file_idx = 0;
}

Reader::Status VideoReader::subfolder_reading()
{
    if ((_sub_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("VideoReader ShardID ["+ TOSTR(_shard_id)+ "] ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while((_entity = readdir (_sub_dir)) != nullptr)
    {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
    }
    closedir(_sub_dir);
    std::sort(entry_name_list.begin(), entry_name_list.end());

    auto ret = Reader::Status::OK;
    for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
        std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
        filesys::path pathObj(subfolder_path);
        if(filesys::exists(pathObj) && filesys::is_regular_file(pathObj))
        {
            // ignore files with extensions .tar, .zip, .7z
            auto file_extension_idx = subfolder_path.find_last_of(".");
            if (file_extension_idx  != std::string::npos) {
                std::string file_extension = subfolder_path.substr(file_extension_idx+1);
                for (auto & c: file_extension) c = toupper(c);
                if (!((file_extension == "MP4") || (file_extension == "M4V") || (file_extension == "MPG") || (file_extension == "MPEG")))
                    continue;
            }
            ret = open_folder();
            break;  // assume directory has only files.
        }
        else if(filesys::exists(pathObj) && filesys::is_directory(pathObj))
        {
            _folder_path = subfolder_path;
            if(open_folder() != Reader::Status::OK)
                WRN("VideoReader ShardID ["+ TOSTR(_shard_id)+ "] VideoReader cannot access the storage at " + _folder_path);
        }
    }
    if(_in_batch_read_count > 0 && _in_batch_read_count < _batch_count)
    {
        replicate_last_image_to_fill_last_shard();
        LOG("VideoReader ShardID [" + TOSTR(_shard_id) + "] Replicated " + _folder_path+_last_file_name + " " + TOSTR((_batch_count - _in_batch_read_count) ) + " times to fill the last batch")
    }
    if(!_video_file_names.empty())
        LOG("VideoReader ShardID ["+ TOSTR(_shard_id)+ "] Total of " + TOSTR(_video_file_names.size()) + " images loaded from " + _full_path )
    return ret;
}
void VideoReader::replicate_last_image_to_fill_last_shard()
{
    for(size_t i = _in_batch_read_count; i < _batch_count; i++)
        _video_file_names.push_back(_last_file_name);
}

void VideoReader::replicate_last_batch_to_pad_partial_shard()
{
    if (_video_file_names.size() >=  _batch_count) {
        for (size_t i = 0; i < _batch_count; i++)
            _video_file_names.push_back(_video_file_names[i - _batch_count]);
    }
}

Reader::Status VideoReader::open_folder()
{
    if ((_src_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("VideoReader ShardID ["+ TOSTR(_shard_id)+ "] ERROR: Failed opening the directory at " + _folder_path);


    while((_entity = readdir (_src_dir)) != nullptr)
    {
        if(_entity->d_type != DT_REG)
            continue;

        if(get_file_shard_id() != _shard_id )
        {
            _file_count_all_shards++;
            incremenet_file_id();
            continue;
        }
        _in_batch_read_count++;
        _in_batch_read_count = (_in_batch_read_count%_batch_count == 0) ? 0 : _in_batch_read_count;
        std::string file_path = _folder_path;
        file_path.append("/");
        file_path.append(_entity->d_name);
        _last_file_name = file_path;
        _video_file_names.push_back(file_path);
        std::cerr << "\nVideo file names : " << file_path;
        _file_count_all_shards++;
        incremenet_file_id();
    }
    if(_video_file_names.empty())
        WRN("VideoReader ShardID ["+ TOSTR(_shard_id)+ "] Did not load any file from " + _folder_path)

    closedir(_src_dir);
    return Reader::Status::OK;
}

size_t VideoReader::get_file_shard_id()
{
    if(_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    //return (_file_id / (_batch_count)) % _shard_count;
    return _file_id  % _shard_count;
}
#endif
