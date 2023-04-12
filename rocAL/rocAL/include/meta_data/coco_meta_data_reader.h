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
#include "commons.h"
#include "meta_data.h"
#include "meta_data_reader.h"
#include "timing_debug.h"

class COCOMetaDataReader: public MetaDataReader
{
public:
    void init(const MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    bool set_timestamp_mode() override { return false; }
    MetaDataBatch * get_output() override { return _output; }
    const std::map<std::string, std::shared_ptr<MetaData>> & get_map_content() override { return _map_content;}
    COCOMetaDataReader();
    ~COCOMetaDataReader() override { delete _output; }
private:
    BoundingBoxBatch* _output;
    std::string _path;
    MetaDataType _meta_data_type;
    int meta_data_reader_type;
    void add(std::string image_name, BoundingBoxCords bbox, BoundingBoxLabels b_labels, ImgSize image_size, MaskCords mask_cords, std::vector<int> polygon_count, std::vector<std::vector<int>> vertices_count);
    void add(std::string image_name, BoundingBoxCords bbox, BoundingBoxLabels b_labels, ImgSize image_size);
    bool exists(const std::string &image_name) override;
    std::map<std::string, std::shared_ptr<MetaData>> _map_content;
    std::map<std::string, std::shared_ptr<MetaData>>::iterator _itr;
    std::map<std::string, ImgSize> _map_img_sizes;
    std::map<std::string, ImgSize> ::iterator itr;
    std::map<int, int> _label_info;
    std::map<int, int> ::iterator _it_label;
    TimingDBG _coco_metadata_read_time;
};

