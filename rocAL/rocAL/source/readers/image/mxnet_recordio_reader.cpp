/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#include <commons.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <memory.h>
#include <stdint.h>
#include "mxnet_recordio_reader.h"
#include <boost/filesystem.hpp>

namespace filesys = boost::filesystem;

using namespace std;

MXNetRecordIOReader::MXNetRecordIOReader():
_shuffle_time("shuffle_time", DBG_TIMING)
{
    _src_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _loop = false;
    _shuffle = false;
    _file_id = 0;
    _file_count_all_shards = 0;
}

unsigned MXNetRecordIOReader::count_items()
{
    if (_loop)
        return _file_names.size();

    int ret = ((int)_file_names.size() - _read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status MXNetRecordIOReader::initialize(ReaderConfig desc)
{
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _loop = desc.loop();
    _shuffle = desc.shuffle();
    ret = record_reading();
    // the following code is required to make every shard the same size:: required for multi-gpu training
    if (_shard_count > 1 && _batch_count > 1) {
        int _num_batches = _file_names.size()/_batch_count;
        int max_batches_per_shard = (_file_count_all_shards + _shard_count-1)/_shard_count;
        max_batches_per_shard = (max_batches_per_shard + _batch_count-1)/_batch_count;
        if (_num_batches < max_batches_per_shard) {
            replicate_last_batch_to_pad_partial_shard();
        }
    }
    //shuffle dataset if set
    _shuffle_time.start();
    if( ret==Reader::Status::OK && _shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());
    _shuffle_time.end();

    return ret;

}

void MXNetRecordIOReader::incremenet_read_ptr()
{
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
}
size_t MXNetRecordIOReader::open()
{
    auto file_path = _file_names[_curr_file_idx]; // Get next file name
    _last_id = file_path;
    auto it = _record_properties.find(_file_names[_curr_file_idx]);
    std::tie(_current_file_size, _seek_pos, _data_size_to_read) = it->second;
    return _current_file_size;
}

size_t MXNetRecordIOReader::read_data(unsigned char *buf, size_t read_size)
{
    auto it = _record_properties.find(_file_names[_curr_file_idx]);
    std::tie(_current_file_size, _seek_pos, _data_size_to_read) = it->second;
    read_image(buf, _seek_pos, _data_size_to_read);
    incremenet_read_ptr();
    return read_size;
}

int MXNetRecordIOReader::close()
{
    return release();
}

MXNetRecordIOReader::~MXNetRecordIOReader()
{
}

int MXNetRecordIOReader::release()
{
    return 0;
}

void MXNetRecordIOReader::reset()
{
    _shuffle_time.start();
    if (_shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());
    _shuffle_time.end();
    _read_counter = 0;
    _curr_file_idx = 0;
}

Reader::Status MXNetRecordIOReader::record_reading()
{
    auto ret = Reader::Status::OK;
    if (MXNet_reader() != Reader::Status::OK)
        WRN("MXNetRecordIOReader ShardID [" + TOSTR(_shard_id) + "] MXNetRecordIOReader cannot access the storage at " + _path);

    if (_in_batch_read_count > 0 && _in_batch_read_count < _batch_count)
    {
        replicate_last_image_to_fill_last_shard();
        LOG("MXNetRecordIOReader ShardID [" << TOSTR(_shard_id) << "] Replicated " << _path + _last_file_name << " " << TOSTR((_batch_count - _in_batch_read_count)) << " times to fill the last batch")
    }
    if (!_file_names.empty())
        LOG("MXNetRecordIOReader ShardID [" << TOSTR(_shard_id) << "] Total of " << TOSTR(_file_names.size()) << " images loaded from " << _path)
    return ret;
}

void MXNetRecordIOReader::replicate_last_image_to_fill_last_shard()
{
    for (size_t i = _in_batch_read_count; i < _batch_count; i++)
    {
        _file_names.push_back(_last_file_name);
        _record_properties.insert(pair<std::string, std::tuple<unsigned int, int64_t, int64_t>>(_last_file_name, std::make_tuple(_last_file_size, _last_seek_pos, _last_data_size)));
    }
}

void MXNetRecordIOReader::replicate_last_batch_to_pad_partial_shard()
{
    if (_file_names.size() >=  _batch_count) {
        for (size_t i = 0; i < _batch_count; i++)
        {
            _file_names.push_back(_file_names[i - _batch_count]);
            auto it = _record_properties.find(_file_names[i - _batch_count]);
            std::tie(_current_file_size, _seek_pos, _data_size_to_read) = it->second;
            _record_properties.insert(pair<std::string, std::tuple<unsigned int, int64_t, int64_t>>(_file_names[i - _batch_count], std::make_tuple(_current_file_size, _seek_pos, _data_size_to_read)));
        }
    }
}

Reader::Status MXNetRecordIOReader::MXNet_reader()
{
    std::string _rec_file, _idx_file;
    if ((_src_dir = opendir (_path.c_str())) == nullptr)
        THROW("MXNetReader ShardID ["+ TOSTR(_shard_id)+ "] ERROR: Failed opening the directory at " + _path);

    while((_entity = readdir (_src_dir)) != nullptr)
    {
        std::string file_name = _path + "/" + _entity->d_name;
        filesys::path pathObj(file_name);
        if(filesys::exists(pathObj) && filesys::is_regular_file(pathObj))
        {
            auto file_extension_idx = file_name.find_last_of(".");
            if (file_extension_idx  != std::string::npos)
            {
                std::string file_extension = file_name.substr(file_extension_idx+1);
                if (file_extension == "rec")
                    _rec_file = file_name;
                else if(file_extension == "idx")
                    _idx_file = file_name;
                else
                    continue;
            }
        }
    }
    closedir(_src_dir);
    uint rec_size;
    _file_contents.open(_rec_file, ios::binary);
    if (!_file_contents)
        THROW("MXNetRecordIOReader ERROR: Failed opening the file " + _rec_file);
    _file_contents.seekg(0, ifstream::end);
    rec_size = _file_contents.tellg();
    _file_contents.seekg(0, ifstream::beg);

    ifstream index_file(_idx_file);
    if(!index_file)
        THROW("MXNetRecordIOReader ERROR: Could not open RecordIO index file. Provided path: " + _idx_file);

    std::vector<size_t> _index_list;
    size_t _index, _offset;
    while (index_file >> _index >> _offset)
        _index_list.push_back(_offset);
    if(_index_list.empty())
        THROW("MXNetRecordIOReader ERROR: RecordIO index file doesn't contain any indices. Provided path: " + _idx_file);
    _index_list.push_back(rec_size);
    std::sort(_index_list.begin(), _index_list.end());
    for (size_t i = 0; i < _index_list.size() - 1; ++i)
        _indices.emplace_back(_index_list[i], _index_list[i + 1] - _index_list[i]);
    read_image_names();
    return Reader::Status::OK;
}

size_t MXNetRecordIOReader::get_file_shard_id()
{
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    return _file_id  % _shard_count;
}

void MXNetRecordIOReader::read_image_names()
{
    for(int current_index = 0; current_index < (int)_indices.size(); current_index++ )
    {
        uint32_t _magic, _length_flag;
        std::tie(_seek_pos, _data_size_to_read) = _indices[current_index];
        _file_contents.seekg(_seek_pos, ifstream::beg);
        uint8_t* _data = new uint8_t[_data_size_to_read];
        uint8_t* _data_ptr = _data;
        auto ret = _file_contents.read((char *)_data_ptr, _data_size_to_read).gcount();
        if(ret == -1 || ret != _data_size_to_read)
            THROW("MXNetRecordIOReader ERROR:  Unable to read the data from the file ");

        _magic = *((uint32_t *) _data_ptr);
        _data_ptr += sizeof(_magic);
        if(_magic != _kMagic)
            THROW("MXNetRecordIOReader ERROR: Invalid MXNet RecordIO: wrong _magic number");
        _length_flag = *((uint32_t *) _data_ptr);
        _data_ptr += sizeof(_length_flag);
        uint32_t _clength =  DecodeLength(_length_flag);
        _hdr = *((ImageRecordIOHeader *) _data_ptr);

        if (_hdr.flag == 0)
            _image_key = to_string(_hdr.image_id[0]);
        else
        {
            WRN("\nMXNetRecordIOReader Multiple record reading has not supported");
            continue;
        }
        /* _clength - sizeof(ImageRecordIOHeader) to get the data size.
        Subtracting label size(_hdr.flag * sizeof(float)) from data size to get image size*/
        int64_t image_size = (_clength - sizeof(ImageRecordIOHeader)) - (_hdr.flag * sizeof(float));
        delete[] _data;

        if (get_file_shard_id() != _shard_id)
        {
            incremenet_file_id();
            _file_count_all_shards++;
            continue;
        }
        _in_batch_read_count++;
        _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;

        _file_names.push_back(_image_key.c_str());
        _last_file_name = _image_key.c_str();

        incremenet_file_id();
        _file_count_all_shards++;

        _last_file_size = image_size;
        _last_seek_pos = _seek_pos;
        _last_data_size = _data_size_to_read;
        //_record_properties vector used to keep track of image size, seek position and data size of the single
        _record_properties.insert(pair<std::string, std::tuple<unsigned int, int64_t, int64_t>>(_last_file_name, std::make_tuple(_last_file_size, _last_seek_pos, _last_data_size)));
    }
}

void MXNetRecordIOReader::read_image(unsigned char *buff, int64_t seek_position, int64_t _data_size_to_read)
{
    uint32_t _magic, _length_flag;
    _file_contents.seekg(seek_position, ifstream::beg);
    uint8_t* _data = new uint8_t[_data_size_to_read];
    uint8_t* _data_ptr = _data;
    auto ret = _file_contents.read((char *)_data_ptr, _data_size_to_read).gcount();
    if(ret == -1 || ret != _data_size_to_read)
        THROW("MXNetRecordIOReader ERROR:  Unable to read the data from the file ");
    _magic = *((uint32_t *) _data_ptr);
    _data_ptr += sizeof(_magic);
    if(_magic != _kMagic)
        THROW("MXNetRecordIOReader ERROR: Invalid RecordIO: wrong _magic number");
    _length_flag = *((uint32_t *) _data_ptr);
    _data_ptr += sizeof(_length_flag);
    uint32_t _cflag = DecodeFlag(_length_flag);
    uint32_t _clength =  DecodeLength(_length_flag);
    _hdr = *((ImageRecordIOHeader *) _data_ptr);
    _data_ptr += sizeof(_hdr);

    int64_t data_size = _clength - sizeof(ImageRecordIOHeader);
    int64_t label_size = _hdr.flag * sizeof(float);
    int64_t image_size = data_size - label_size;
    if (_cflag == 0)
        memcpy(buff, _data_ptr + label_size, image_size);
    else
        THROW("\nMultiple record reading has not supported");
    delete[] _data;
}