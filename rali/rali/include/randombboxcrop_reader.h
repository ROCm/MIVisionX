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
#include <vx_ext_rpp.h>
#include <graph.h>
#include "commons.h"
#include "randombboxcrop_meta_data_reader.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"

class RandomBBoxCropReader: public RandomBBoxCrop_MetaDataReader
{
public:
    void init(const RandomBBoxCrop_MetaDataConfig& cfg) override;
    void lookup(const std::string& image_name) override;
    void read_all() override;
    void release() override;
    void print_map_contents();
    RandomBBoxCropReader();
    ~RandomBBoxCropReader() override { }
private:
    std::shared_ptr<RaliRandomCropParam> _meta_crop_param;
    int _all_boxes_overlap;
    int _no_crop;
    int _has_shape;
    int _crop_width;
    int _crop_height;
    void add(std::string image_name, CropCord bbox);
    bool exists(const std::string &image_name);
    std::map<std::string, std::shared_ptr<CropCord>> _map_content;
    std::map<std::string, std::shared_ptr<CropCord>>::iterator _itr;
};

