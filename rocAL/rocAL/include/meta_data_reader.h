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

#include <string>
#include <memory>
#include "meta_data.h"

enum class MetaDataReaderType
{
    FOLDER_BASED_LABEL_READER = 0,// Used for imagenet-like dataset
    TEXT_FILE_META_DATA_READER,// Used when metadata is stored in a text file
    COCO_META_DATA_READER,
    CIFAR10_META_DATA_READER,    // meta_data for cifar10 data which is store as part of bin file
    TF_META_DATA_READER,
    CAFFE_META_DATA_READER,
    CAFFE_DETECTION_META_DATA_READER,
    CAFFE2_META_DATA_READER,
    CAFFE2_DETECTION_META_DATA_READER,
    TF_DETECTION_META_DATA_READER
};
enum class MetaDataType
{
    Label,
    BoundingBox
};

struct MetaDataConfig
{
private:
    MetaDataType _type;
    MetaDataReaderType _reader_type;
    std::string _path;
    std::map<std::string, std::string> _feature_key_map; 
    std::string _file_prefix;           // if we want to read only filenames with prefix (needed for cifar10 meta data)
public:
    MetaDataConfig(const MetaDataType& type, const MetaDataReaderType& reader_type, const std::string& path, const std::map<std::string, std::string> &feature_key_map=std::map<std::string, std::string>(), const std::string file_prefix=std::string())
                    :_type(type), _reader_type(reader_type),  _path(path), _feature_key_map(feature_key_map), _file_prefix(file_prefix){}
    MetaDataConfig() = delete;
    MetaDataType type() const { return _type; }
    MetaDataReaderType reader_type() const { return _reader_type; }
    std::string path() const { return  _path; }
    std::map<std::string, std::string> feature_key_map() const {return _feature_key_map; }
    std::string file_prefix() const { return  _file_prefix; }
};


class MetaDataReader
{
public:
    enum class Status
    {
        OK = 0
    };
    virtual ~MetaDataReader()= default;
    virtual void init(const MetaDataConfig& cfg) = 0;
    virtual void read_all(const std::string& path) = 0;// Reads all the meta data information
    virtual void lookup(const std::vector<std::string>& image_names) = 0;// finds meta_data info associated with given names and fills the output
    virtual void release() = 0; // Deletes the loaded information
    virtual MetaDataBatch * get_output()= 0;
    virtual bool exists(const std::string &image_name) = 0;
};

