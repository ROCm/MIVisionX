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
#include <map>
#include <dirent.h>
#include <memory>
#include <list>
#include <variant>
#include <fstream>
#include <string>
#include "commons.h"
#include "meta_data.h"
#include "meta_data_reader.h"
#include "reader.h"

class MXNetMetaDataReader: public MetaDataReader
{
public :
    void init(const MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    bool set_timestamp_mode() override { return false; }
    MetaDataBatch * get_output() override { return _output; }
    MXNetMetaDataReader();
    ~MXNetMetaDataReader() override { delete _output; }
private:
    void read_images();
    bool exists(const std::string &image_name);
    void add(std::string image_name, int label);
    uint32_t DecodeFlag(uint32_t rec) {return (rec >> 29U) & 7U; };
    uint32_t DecodeLength(uint32_t rec) {return rec & ((1U << 29U) - 1U); };
    std::vector<std::tuple<int64_t, int64_t>> _indices; // used to store seek position and record size for a particular record.
    std::ifstream _file_contents;
    ImageRecordIOHeader _hdr;
    const uint32_t _kMagic = 0xced7230a;
    std::map<std::string, std::shared_ptr<Label>> _map_content;
    std::string _path;
    DIR *_src_dir;
    struct dirent *_entity;
    LabelBatch* _output;
    std::vector<std::string> _image_name;
};