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

#pragma once
#include <map>
#include <dirent.h>
#include <memory>
#include <list>
#include <variant>
#include "commons.h"
#include "meta_data.h"
#include "meta_data_reader.h"
#include "reader.h"
#include "lmdb.h"
#include "caffe_protos.pb.h"

class CaffeMetaDataReaderDetection: public MetaDataReader
{
public :
    void init(const MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    bool set_timestamp_mode() override { return false; }
    void print_map_contents();
    std::map<std::string, std::shared_ptr<MetaData>> &get_map_content() override{ return _map_content;}
    MetaDataBatch * get_output() override { return _output; }
    CaffeMetaDataReaderDetection();
    ~CaffeMetaDataReaderDetection() override { delete _output; }
private:
    void read_files(const std::string& _path);
    bool exists(const std::string &image_name) override;
    void add(std::string image_name, BoundingBoxCords bbox, BoundingBoxLabels b_labels, ImgSize image_size);
    bool _last_rec;
    void read_lmdb_record(std::string file_name, uint file_size);
    std::map<std::string, std::shared_ptr<MetaData>> _map_content;
    std::map<std::string, std::shared_ptr<MetaData>>::iterator _itr;
    std::string _path;
    BoundingBoxBatch* _output;
    DIR *_src_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names;
    MDB_env* _mdb_env;
    MDB_dbi _mdb_dbi;
    MDB_val _mdb_key, _mdb_value;
    MDB_txn* _mdb_txn;
    MDB_cursor* _mdb_cursor;
};
