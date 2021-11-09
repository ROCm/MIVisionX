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

#pragma once
#include <vector>
#include <tuple>
#include <string>
#include <memory>
#include "commons.h"
#include "meta_data_reader.h"
#include "video_properties.h"

enum class VideoStorageType
{
    VIDEO_FILE_SYSTEM = 0
};

struct VideoReaderConfig
{
    explicit VideoReaderConfig(VideoStorageType type, std::string path = "", bool shuffle = false, bool loop = false) : 
                            _type(type), _path(path), _shuffle(shuffle), _loop(loop) {}
    virtual VideoStorageType type() { return _type; };
    void set_path(const std::string &path) { _path = path; }
    void set_shard_id(size_t shard_id) { _shard_id = shard_id; }
    void set_shard_count(size_t shard_count) { _shard_count = shard_count; }
    /// \param read_batch_count Tells the reader it needs to read the video sequences of load_batch_count. If available video sequences not divisible to load_batch_count,
    /// the reader will repeat video sequences to make available sequences an even multiple of this load_batch_count
    void set_batch_count(size_t read_batch_count) { _batch_count = read_batch_count; }
    /// \param loop if True the reader's available video sequences still the same no matter how many sequences have been read
    bool shuffle() { return _shuffle; }
    bool loop() { return _loop; }
    void set_shuffle(bool shuffle) { _shuffle = shuffle; }
    void set_loop(bool loop) { _loop = loop; }
    void set_meta_data_reader(std::shared_ptr<MetaDataReader> meta_data_reader) { _meta_data_reader = meta_data_reader; }
    void set_sequence_length(unsigned sequence_length) { _sequence_length = sequence_length; }
    void set_frame_step(unsigned step) { _video_frame_step = step; }
    void set_frame_stride(unsigned stride) { _video_frame_stride = stride; }
    void set_total_frames_count(size_t total) { _total_frames_count = total; }
    void set_video_properties(VideoProperties video_prop) { _video_prop = video_prop;}
    size_t get_shard_count() { return _shard_count; }
    size_t get_shard_id() { return _shard_id; }
    size_t get_batch_size() { return _batch_count; }
    size_t get_sequence_length() { return _sequence_length; }
    size_t get_frame_step() { return _video_frame_step; }
    size_t get_frame_stride() { return _video_frame_stride; }
    size_t get_total_frames_count() { return _total_frames_count; }
    VideoProperties get_video_properties() { return _video_prop; }
    std::string path() { return _path; }
    std::shared_ptr<MetaDataReader> meta_data_reader() { return _meta_data_reader; }
private:
    VideoStorageType _type = VideoStorageType::VIDEO_FILE_SYSTEM;
    std::string _path = "";
    size_t _shard_count = 1;
    size_t _shard_id = 0;
    size_t _batch_count = 1;     //!< The reader will repeat images if necessary to be able to have images in multiples of the _batch_count.
    size_t _sequence_length = 1; // Video reader module sequence length
    size_t _video_frame_step;
    size_t _video_frame_stride = 1;
    VideoProperties _video_prop;
    size_t _total_frames_count;
    bool _shuffle = false;
    bool _loop = false;
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
};

#ifdef RALI_VIDEO
struct SequenceInfo 
{
    size_t start_frame_number;
    std::string video_file_name;
};

class VideoReader
{
public:
    enum class Status
    {
        OK = 0
    };
    //! Looks up the folder which contains the files, amd loads the video sequences
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    virtual Status initialize(VideoReaderConfig desc) = 0;
    
    //! Reads the next resource item
    virtual SequenceInfo get_sequence_info() = 0;

    //! Resets the object's state to read from the first file in the folder
    virtual void reset() = 0;

    //! Returns the name of the latest file opened
    virtual std::string id() = 0;

    //! Returns the number of items remained in this resource
    virtual unsigned count_items() = 0;

    //! return shuffle_time if applicable
    virtual unsigned long long get_shuffle_time() = 0;

    virtual ~VideoReader() = default;
};
#endif
