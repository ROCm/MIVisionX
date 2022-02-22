/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <string>
#include <memory>
#include <dirent.h>
#include <map>
#include <iterator>
#include <algorithm>
#include <fstream>
#include "reader.h"
#include "timing_debug.h"

class MXNetRecordIOReader : public Reader{
public:
    //! Reads the MXNet Record File, and loads the image ids and other necessary info
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded images
     \return Size of the loaded resource
    */
    size_t read_data(unsigned char* buf, size_t max_size) override;
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override;
    //! Resets the object's state to read from the first file in the folder
    void reset() override;

    //! Returns the id of the latest file opened
    std::string id() override { return _last_id;};

    unsigned count_items() override;

    ~MXNetRecordIOReader() override;

    int close() override;

    MXNetRecordIOReader();
    unsigned long long get_shuffle_time() override {return 0;}
private:
    //! opens the folder containnig the images
    Reader::Status record_reading();
    Reader::Status MXNet_reader();
    std::string _path;
    DIR *_src_dir;
    struct dirent *_entity;
    std::string _image_key;
    std::vector<std::string> _file_names;
    std::map<std::string, std::tuple<unsigned int, int64_t, int64_t> > _record_properties;
    unsigned  _curr_file_idx;
    unsigned _current_file_size;
    std::string _last_id, _last_file_name;
    unsigned int _last_file_size;
    int64_t _last_seek_pos;
    int64_t _last_data_size;
    size_t _shard_id = 0;
    size_t _shard_count = 1;// equivalent of batch size
    //!< _batch_count Defines the quantum count of the images to be read. It's usually equal to the user's batch size.
    /// The loader will repeat images if necessary to be able to have images available in multiples of the load_batch_count,
    /// for instance if there are 10 images in the dataset and _batch_count is 3, the loader repeats 2 images as if there are 12 images available.
    size_t _batch_count = 1;
    size_t _file_id = 0;
    size_t _in_batch_read_count = 0;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;
    //!< _file_count_all_shards total_number of files in to figure out the max_batch_size (usually needed for distributed training).
    size_t  _file_count_all_shards;
    void incremenet_read_ptr();
    int release();
    size_t get_file_shard_id();
    void incremenet_file_id() { _file_id++; }
    void replicate_last_image_to_fill_last_shard();
    void replicate_last_batch_to_pad_partial_shard();
    void read_image(unsigned char* buff, int64_t seek_position, int64_t data_size);
    void read_image_names();
    uint32_t DecodeFlag(uint32_t rec) {return (rec >> 29U) & 7U; };
    uint32_t DecodeLength(uint32_t rec) {return rec & ((1U << 29U) - 1U); };
    std::vector<std::tuple<int64_t, int64_t>> _indices;// used to store seek position and record size for a particular record.
    std::ifstream _file_contents;
    const uint32_t _kMagic = 0xced7230a;
    int64_t _seek_pos, _data_size_to_read;
    ImageRecordIOHeader _hdr;
    TimingDBG _shuffle_time;
};

