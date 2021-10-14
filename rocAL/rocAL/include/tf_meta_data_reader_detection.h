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
#include <map>
#include <dirent.h>
#include <memory>
#include <list>
#include <variant>
#include "commons.h"
#include "meta_data.h"
#include "meta_data_reader.h"


class TFMetaDataReaderDetection: public MetaDataReader
{
public :
    void init(const MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    MetaDataBatch * get_output() override { return _output; }
    TFMetaDataReaderDetection();
    ~TFMetaDataReaderDetection() override { delete _output; }
private:
    void read_files(const std::string& _path);
    bool exists(const std::string &image_name);
    // void add(std::string image_name, int label);
    //bbox add
    void add(std::string image_name, BoundingBoxCords bbox, BoundingBoxLabels b_labels, ImgSizes image_size);
    bool _last_rec;
    //std::shared_ptr<TF_Read> _TF_read = nullptr;
    void read_record(std::ifstream &file_contents, uint file_size, std::vector<std::string> &image_name, 
        std::string user_label_key, std::string user_text_key, 
        std::string user_xmin_key, std::string user_ymin_key, std::string user_xmax_key, std::string user_ymax_key,
        std::string user_filename_key);    // std::map<std::string, std::shared_ptr<Label>> _map_content;
    // std::map<std::string, std::shared_ptr<Label>>::iterator _itr;
    // //bbox map contents
    std::map<std::string, std::shared_ptr<BoundingBox>> _map_content;
    std::map<std::string, std::shared_ptr<BoundingBox>>::iterator _itr;
    std::string _path;
    BoundingBoxBatch* _output;
    DIR *_src_dir;
    struct dirent *_entity;
    std::map<std::string, std::string> _feature_key_map;
    std::vector<std::string> _file_names;
    std::vector<std::string> _subfolder_file_names;
    std::vector<std::string> _image_name;
};