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
        return _total_video_frames_count;
    
    int ret = (int)(_total_video_frames_count - _read_counter);
    return ((ret <= 0) ? 0 : ret);
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
    _video_count = desc.get_video_count();
    // ret = subfolder_reading();
    // _video_file_count = _video_file_names.size();
    _video_file_names = desc.get_video_file_names();
    _sequence_length = desc.get_sequence_length();
    _video_frame_count = desc.get_frame_count();
    _total_video_frames_count = 0;

    // get the width and height for every video _actual_decoded & original
    // fill the _video_frame_start_idx & _video_idx  based on sequence length and frame count
    // shuffle both _video_frame_start_idx & _video_idx ( can do this later)
    //for sample test
    //_video_frame_count[3] = {30, 25, 54};

    size_t count_sequence;
    for(size_t i = 0; i < _video_count; i++)
    {
        count_sequence = 0;
        // std::cerr << "\n Frames per video : " << _video_frame_count[i];
        int loop_index;

        loop_index = _video_frame_count[i] / _sequence_length;
        for(int j = 0; j < loop_index; j++)
        {
            _frame_sequences.push_back(std::make_tuple(i, count_sequence));
            count_sequence = count_sequence + _sequence_length;
        }
        _total_video_frames_count += (loop_index * _sequence_length);
    }
    

    std::cerr << "The total frames count : " << _total_video_frames_count << "\n";
    desc.set_total_frames_count(_total_video_frames_count);

    // // the following code is required to make every shard the same size:: required for multi-gpu training
    // if (_shard_count > 1 && _batch_count > 1) { // check needed
    //     int _num_batches = _frame_sequences.size()/_batch_count; // check needed
    //     int max_batches_per_shard = (_file_count_all_shards + _shard_count-1)/_shard_count; // check needed
    //     max_batches_per_shard = (max_batches_per_shard + _batch_count-1)/_batch_count; // check needed
    //     if (_num_batches < max_batches_per_shard) { // check needed
    //         replicate_last_batch_to_pad_partial_shard(); // check needed
    //     }
    // }
    if( ret==Reader::Status::OK && _shuffle)
        std::random_shuffle(_frame_sequences.begin(), _frame_sequences.end());
    return ret;
}

void VideoReader::incremenet_read_ptr()
{
    _read_counter += _sequence_length;
    _curr_file_idx = (_curr_file_idx + 1) % _frame_sequences.size();
}

size_t VideoReader::open()
{
    // opening file is not needed for decoding video, since the low level decoder directly handles it
    return 0;
}

size_t VideoReader::read(unsigned char* buf, size_t read_size)
{
    auto file_path = _video_file_names[std::get<0>(_frame_sequences[_curr_file_idx])];// Get next file name
    _last_id = file_path;
    size_t start_frame = std::get<1>(_frame_sequences[_curr_file_idx]);
    incremenet_read_ptr();
    // _current_fPtr = fopen(file_path.c_str(), "rb");// Open the file,
    // if (!_current_fPtr || read_size < file_path.size())
    //     return 0;
    // fclose(_current_fPtr);
    // _current_fPtr = nullptr;
    // for video instead of reading the file, retun the filename instead
    // strcpy((char *)buf, file_path.c_str());
    // return file_path.size();
    return start_frame;
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
    if (_shuffle) std::random_shuffle(_frame_sequences.begin(), _frame_sequences.end());
    _read_counter = 0;
    _curr_file_idx = 0;
}

Reader::Status VideoReader::subfolder_reading()
{
    auto ret = Reader::Status::OK;
    filesys::path pathObj(_folder_path);
    if(filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) // Single file as input
    {

        _video_file_names.push_back(_folder_path);
    }
    else
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
    }
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
        // file_path.append("/");
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
