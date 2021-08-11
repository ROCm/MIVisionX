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

VideoReader::VideoReader() : _shuffle_time("shuffle_time", DBG_TIMING)
{
    _curr_sequence_idx = 0;
    _loop = false;
    _sequence_id = 0;
    _shuffle = false;
    _sequence_count_all_shards = 0;
}

unsigned VideoReader::count()
{
    if (_loop)
        return _total_video_frames_count;
    int ret = (int)(_total_video_frames_count - _read_counter);
    return ((ret <= 0) ? 0 : ret);
}

Reader::Status VideoReader::initialize(ReaderConfig desc)
{
    auto ret = Reader::Status::OK;
    _sequence_id = 0;
    _folder_path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _video_count = desc.get_video_count();
    _video_file_names.resize(_video_count);
    _start_end_frame.resize(_video_count);
    _video_file_names = desc.get_video_file_names();
    _sequence_length = desc.get_sequence_length();
    _step = desc.get_frame_step();
    _stride = desc.get_frame_stride();
    _video_frame_count = desc.get_video_frames_count();
    _start_end_frame = desc.get_start_end_frame_vector();
    _user_batch_count = desc.get_batch_size();
    _batch_count = _user_batch_count / _sequence_length;
    _total_video_frames_count = 0;
    ret = get_sequences();

    // the following code is required to make every shard the same size:: required for multi-gpu training
    if (_shard_count > 1 && _batch_count > 1) {
        int _num_batches = _frame_sequences.size()/_batch_count;
        int max_batches_per_shard = (_sequence_count_all_shards + _shard_count-1)/_shard_count;
        max_batches_per_shard = (max_batches_per_shard + _batch_count-1)/_batch_count;
        if (_num_batches < max_batches_per_shard) {
            replicate_last_batch_to_pad_partial_shard();
        }
    }
    if (ret == Reader::Status::OK && _shuffle)
        std::random_shuffle(_frame_sequences.begin(), _frame_sequences.end());
    return ret;
}

void VideoReader::incremenet_read_ptr()
{
    _read_counter += _sequence_length;
    _curr_sequence_idx = (_curr_sequence_idx + 1) % _frame_sequences.size();
}

size_t VideoReader::open()
{
    // opening file is not needed for decoding video, since the low level decoder directly handles it
    return 0;
}

size_t VideoReader::read(unsigned char *buf, size_t read_size)
{
    auto file_path = _video_file_names[std::get<0>(_frame_sequences[_curr_sequence_idx])]; // Get next file name
    _last_id = file_path;
    size_t start_frame = std::get<1>(_frame_sequences[_curr_sequence_idx]);
    incremenet_read_ptr();
    return start_frame;
}

int VideoReader::close()
{
    return 0;
}

VideoReader::~VideoReader()
{
}

void VideoReader::reset()
{
    if (_shuffle)
        std::random_shuffle(_frame_sequences.begin(), _frame_sequences.end());
    _read_counter = 0;
    _curr_sequence_idx = 0;
}

Reader::Status VideoReader::get_sequences()
{
    Reader::Status status = Reader::Status::OK;
    for (size_t i = 0; i < _video_count; i++)
    {
        unsigned start = std::get<0>(_start_end_frame[i]);
        // unsigned end = std::get<1>(_start_end_frame[i]);
        size_t max_sequence_frames = (_sequence_length - 1) * _stride;
        for(size_t sequence_start = start; (sequence_start + max_sequence_frames) <  (start + _video_frame_count[i]); sequence_start += _step)
        {
            if(get_sequence_shard_id() != _shard_id )
            {
                _sequence_count_all_shards++;
                incremenet_sequence_id();
                continue;
            }
            _in_batch_read_count++;
            _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;
            _frame_sequences.push_back(std::make_tuple(i, sequence_start));
            _last_sequence = _frame_sequences.back();
            _total_video_frames_count += _sequence_length;
            _sequence_count_all_shards++;
            incremenet_sequence_id();
        }
    }
    if(_in_batch_read_count > 0 && _in_batch_read_count < _batch_count)
    {
        replicate_last_sequence_to_fill_last_shard();
        LOG("VideoReader ShardID [" + TOSTR(_shard_id) + "] Replicated the last sequence " + TOSTR((_batch_count - _in_batch_read_count) ) + " times to fill the last batch")
    }
    if(_frame_sequences.empty())
        WRN("VideoReader ShardID ["+ TOSTR(_shard_id)+ "] Did not load any sequences from " + _folder_path)
    return status;
}

void VideoReader::replicate_last_sequence_to_fill_last_shard()
{
    for (size_t i = _in_batch_read_count; i < _batch_count; i++)
    {
        _frame_sequences.push_back(_last_sequence);
        _total_video_frames_count += _sequence_length;
    }
}

void VideoReader::replicate_last_batch_to_pad_partial_shard()
{
    if (_frame_sequences.size() >= _batch_count)
    {
        for (size_t i = 0; i < _batch_count; i++)
        {
            _frame_sequences.push_back(_frame_sequences[i - _batch_count]);
            _total_video_frames_count += _sequence_length;
        }
    }
}

size_t VideoReader::get_sequence_shard_id()
{
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    //return (_sequence_id / (_batch_count)) % _shard_count;
    return _sequence_id % _shard_count;
}
#endif
