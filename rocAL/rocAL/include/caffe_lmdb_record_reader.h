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

#pragma once
#include <vector>
#include <string>
#include <memory>
#include <dirent.h>
#include <map>
#include <iterator>
#include <algorithm>
#include "reader.h"
#include <google/protobuf/message_lite.h>
#include <lmdb.h>
#include "caffe_protos.pb.h"
#include "timing_debug.h"


class CaffeLMDBRecordReader : public Reader{
public:
    //! Reads the TFRecord File, and loads the image ids and other necessary info
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded images
     \return Size of the loaded resource
    */
    size_t read(unsigned char* buf, size_t max_size) override;
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override;
    //! Resets the object's state to read from the first file in the folder
    void reset() override;

    //! Returns the id of the latest file opened
    std::string id() override { return _last_id;};

    unsigned count() override;

    ~CaffeLMDBRecordReader() override;

    int close() override;

    CaffeLMDBRecordReader();
    unsigned long long get_shuffle_time() override {return 0;}
private:
    //! opens the folder containnig the images
    Reader::Status folder_reading();
    Reader::Status Caffe_LMDB_reader();
    std::string _folder_path;
    std::string _path;
    DIR *_sub_dir;
    std::vector<std::string> _file_names;
    std::map<std::string, unsigned int > _file_size;
    unsigned  _curr_file_idx;
    unsigned _current_file_size;
    std::string _last_id;
    std::string _last_file_name;
    unsigned int _last_file_size;
    size_t _shard_id = 0;
    size_t _shard_count = 1;// equivalent of batch size
    bool _last_rec;
    //!< _batch_count Defines the quantum count of the images to be read. It's usually equal to the user's batch size.
    /// The loader will repeat images if necessary to be able to have images available in multiples of the load_batch_count,
    /// for instance if there are 10 images in the dataset and _batch_count is 3, the loader repeats 2 images as if there are 12 images available.
    size_t _batch_count = 1;
    size_t _file_id = 0;
    size_t _in_batch_read_count = 0;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;
    MDB_env* _mdb_env,  *_read_mdb_env;
    MDB_dbi _mdb_dbi, _read_mdb_dbi;
    MDB_val _mdb_key, _mdb_value, _read_mdb_key, _read_mdb_value;
    MDB_txn* _mdb_txn, *_read_mdb_txn;
    MDB_cursor* _mdb_cursor, *_read_mdb_cursor;
    uint _file_byte_size;
    void incremenet_read_ptr();
    int release();
    size_t get_file_shard_id();
    void incremenet_file_id() { _file_id++; }
    void replicate_last_image_to_fill_last_shard();
    void read_image(unsigned char* buff, std::string _file_name);
    void read_image_names();
    std::map <std::string, uint> _image_record_starting;
    TimingDBG _shuffle_time;
    int _open_env = 1;
    int rc;
    void open_env_for_read_image();
};

