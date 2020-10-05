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
#include "commons.h"
#include "meta_data.h"
#include "meta_data_reader.h"
class COCOMetaDataReader: public MetaDataReader
{
public:
    void init(const MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    MetaDataBatch * get_output() override { return _output; }
    COCOMetaDataReader();
    ~COCOMetaDataReader() override { delete _output; }
private:
    BoundingBoxBatch* _output;
    std::string _path;
    void add(std::string image_name, BoundingBoxCords bbox, BoundingBoxLabels b_labels, ImgSizes image_size);
    bool exists(const std::string &image_name);
    std::map<std::string, std::shared_ptr<BoundingBox>> _map_content;
    std::map<std::string, std::shared_ptr<BoundingBox>>::iterator _itr;
    std::map<std::string ,std::vector<ImgSize>> _map_img_sizes;
    std::map<std::string , std::vector<ImgSize> > ::iterator itr;
};

