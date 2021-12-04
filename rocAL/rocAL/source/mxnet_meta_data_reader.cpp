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

#include "mxnet_meta_data_reader.h"
#include <iostream>
#include <utility>
#include <algorithm>
#include <stdint.h>
#include <google/protobuf/message_lite.h>
#include "example.pb.h"
#include "feature.pb.h"

using namespace std;

void MXNetMetaDataReader::init(const MetaDataConfig &cfg)
{
    _path = cfg.path();
    _output = new LabelBatch();
}

bool MXNetMetaDataReader::exists(const std::string& _image_name)
{
    return _map_content.find(_image_name) != _map_content.end();
}

void MXNetMetaDataReader::add(std::string image_name, int label)
{
    pMetaData info = std::make_shared<Label>(label);
    if(exists(image_name))
    {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(std::pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

void MXNetMetaDataReader::lookup(const std::vector<std::string> &_image_names)
{
    if(_image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if(_image_names.size() != (unsigned)_output->size())   
        _output->resize(_image_names.size());

    for(unsigned i = 0; i < _image_names.size(); i++)
    {
        auto _image_name = _image_names[i];
        auto it = _map_content.find(_image_name);
        if(_map_content.end() == it)
            THROW("ERROR: Given name not present in the map"+ _image_name )
        _output->get_label_batch()[i] = it->second->get_label();
    }
}

void MXNetMetaDataReader::print_map_contents()
{
    std::cerr << "\nMap contents: \n";
    for (auto& elem : _map_content) {
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_label() << std::endl;
    }
}

void MXNetMetaDataReader::read_all(const std::string &path)
{
    string rec_file = _path + "/train.rec";   
    string idx_file = _path + "/train.idx";

    uint rec_size;
    _file_contents.open(rec_file);
    if (!_file_contents)
        THROW("ERROR: Failed opening the file " + rec_file);
    _file_offsets.push_back(0);
    _file_contents.seekg(0, ifstream::end);
    rec_size = _file_contents.tellg();
    _file_contents.seekg(0, ifstream::beg);
    _file_offsets.push_back(rec_size);
    
    ifstream index_file(idx_file);
    if(!index_file)
        THROW("ERROR: Could not open RecordIO index file. Provided path: " + idx_file);

    while (index_file >> _index >> _offset)
        _temp.push_back(_offset);
    if(_temp.empty())
        THROW("ERROR: RecordIO index file doesn't contain any indices. Provided path: " + idx_file);
    
    std::sort(_temp.begin(), _temp.end());
    size_t file_offset_index = 0;
    int64_t size;
    for (size_t i = 0; i < _temp.size() - 1; ++i)
    {
        if (_temp[i] >= _file_offsets[file_offset_index + 1])
            ++file_offset_index;
        size = _temp[i + 1] - _temp[i];
        if (size)
            _indices.emplace_back(_temp[i] - _file_offsets[file_offset_index], size, file_offset_index);
    }
    size = _file_offsets.back() - _temp.back();
    if (size)
        _indices.emplace_back(_temp.back() - _file_offsets[file_offset_index], size, file_offset_index);
    read_images();
}

void MXNetMetaDataReader::release(std::string _image_name)
{
    if(!exists(_image_name))
    {
        WRN("ERROR: Given not present in the map" + _image_name);
        return;
    }
    _map_content.erase(_image_name);
}

void MXNetMetaDataReader::release() {
    _map_content.clear();
}

void MXNetMetaDataReader::read_images()
{
    for(int current_index = 0; current_index < _indices.size(); current_index++ )
    {
        std::tie(_seek_pos, _data_size_to_read, _file_index) = _indices[current_index];        
        _file_contents.seekg(_seek_pos, ifstream::beg);
        _data = (uint8_t*)malloc(_data_size_to_read);
        _file_contents.read((char *)_data, _data_size_to_read);        
        memcpy(&_magic, _data, sizeof(_magic));
        _data += sizeof(_magic);
        if(_magic != _kMagic)
            THROW("ERROR: Invalid RecordIO: wrong magic number");
        memcpy(&_length_flag, _data, sizeof(_length_flag));
        _data += sizeof(_length_flag);
        _cflag = DecodeFlag(_length_flag);
        _clength =  DecodeLength(_length_flag);
        memcpy(&_hdr, _data, sizeof(_hdr));
        _data += sizeof(_hdr);
        
        if (_hdr.flag == 0)
        {
            //std::string img_name = rec_file;
            //img_name.append("_");
            //img_name.append(to_string(_hdr.image_id[0]));
            add((to_string(_hdr.image_id[0]) + ".jpg"), _hdr.label);
        }
    }
    print_map_contents();
}

MXNetMetaDataReader::MXNetMetaDataReader()
{
}
