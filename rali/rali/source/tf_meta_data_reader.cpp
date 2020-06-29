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

#include "tf_meta_data_reader.h"
#include <iostream>
#include <utility>
#include <algorithm>
#include<fstream>
#include <string>
#include <stdint.h>
#include <google/protobuf/message_lite.h>
#include "example.pb.h"
#include "feature.pb.h"

using namespace std;

void TFMetaDataReader::init(const MetaDataConfig &cfg)
{
    _path = cfg.path();
    _output = new LabelBatch();
    _last_rec = false;
}

bool TFMetaDataReader::exists(const std::string& _image_name)
{
    return _map_content.find(_image_name) != _map_content.end();
}

void TFMetaDataReader::add(std::string _image_name, int label)
{
    pMetaData info = std::make_shared<Label>(label);
    if(exists(_image_name))
    {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(pair<std::string, std::shared_ptr<Label>>(_image_name, info));
}

void TFMetaDataReader::lookup(const std::vector<std::string> &_image_names)
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

void TFMetaDataReader::print_map_contents()
{
    std::cerr << "\nMap contents: \n";
    for (auto& elem : _map_content) {
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_label() << std::endl;
    }
}

MetaDataReader::Status TFMetaDataReader::read_record(std::ifstream &file_contents, uint file_size, std::vector<std::string> &_image_name)
{
    auto ret = MetaDataReader::Status::OK;
    uint length;
    length = file_contents.tellg();
    size_t uint64_size, uint32_size;
    uint64_t data_length;
    uint32_t length_crc, data_crc;
    uint64_size = sizeof(uint64_t); 
    uint32_size = sizeof(uint32_t); 
    char * header_length = new char [uint64_size];
    char * header_crc = new char [uint32_size];
    char * footer_crc = new char [uint32_size];
    file_contents.read(header_length, uint64_size);
    if(!file_contents)
        THROW("TFMetaDataReader: Error in reading TF records")
    file_contents.read(header_crc, uint32_size);
    if(!file_contents)
        THROW("TFMetaDataReader: Error in reading TF records")
    memcpy(&data_length, header_length, sizeof(data_length));
    memcpy(&length_crc, header_crc, sizeof(length_crc));
    if(length + data_length + 16 == file_size){
        _last_rec = true;
    }
    char *data = new char[data_length];
    file_contents.read(data,data_length);
    if(!file_contents)
        THROW("TFMetaDataReader: Error in reading TF records");
    tensorflow::Example single_example;
    single_example.ParseFromArray(data,data_length);
    tensorflow::Features features = single_example.features();
    auto feature = features.feature();
    tensorflow::Feature single_feature;
    single_feature = feature.at("image/filename");
    std::string fname = single_feature.bytes_list().value()[0];
    _image_name.push_back(fname);
    uint label;
    single_feature = feature.at("image/class/label");
    label = single_feature.int64_list().value()[0];
    add(fname, label);
    file_contents.read(footer_crc, sizeof(data_crc));
    if(!file_contents)
        THROW("TFMetaDataReader: Error in reading TF records");
    memcpy(&data_crc, footer_crc, sizeof(data_crc));
    delete[] header_length;
    delete[] header_crc;
    delete[] footer_crc;
    delete[] data;
    return ret;
}

void TFMetaDataReader::read_all(const std::string &path)
{
    read_files(path);
    auto ret = MetaDataReader::Status::OK;
    for(unsigned i = 0; i < _file_names.size(); i++)
    {
        std::string fname = path + _file_names[i];
        uint length;
        std::ifstream file_contents(fname.c_str(),std::ios::binary);
        if(!file_contents)
            THROW("TFMetaDataReader: Failed to open file "+fname);
        file_contents.seekg (0, std::ifstream::end);
        length = file_contents.tellg();
        file_contents.seekg (0, std::ifstream::beg);
        while(!_last_rec)
        {
            ret = read_record(file_contents, length, _image_name);
            if(ret != MetaDataReader::Status::OK )
                THROW("TFMetaDataReader: Error in reading TF records");
        }
        _last_rec = false;
        file_contents.close();
    }
}

void TFMetaDataReader::release(std::string _image_name)
{
    if(!exists(_image_name))
    {
        WRN("ERROR: Given not present in the map" + _image_name);
        return;
    }
    _map_content.erase(_image_name);
}

void TFMetaDataReader::release() {
    _map_content.clear();
}

void TFMetaDataReader::read_files(const std::string& _path)
{
    if ((_src_dir = opendir (_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _path);

    while((_entity = readdir (_src_dir)) != nullptr)
    {
        if(_entity->d_type != DT_REG)
            continue;

        _file_names.push_back(_entity->d_name);  
    }
    if(_file_names.empty())
        WRN("LabelReader: Could not find any file in " + _path)
    closedir(_src_dir);
}


TFMetaDataReader::TFMetaDataReader()
{
}
