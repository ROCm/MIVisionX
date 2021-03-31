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
#include "meta_data_reader.h"
#include "coco_meta_data_reader.h"

class RandomBBoxCropReader: public RandomBBoxCrop_MetaDataReader
{
public:
    void init(const RandomBBoxCrop_MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all() override;
    void release() override;
    void print_map_contents();
    void update_meta_data();
    CropCordBatch * get_output() override { return _output; }
    std::shared_ptr<RaliRandomCropParam> get_crop_param() { return _crop_param; }
    float get_threshold(){return _threshold;}
    std::vector<std::pair<bool, float>> get_iou_range(){return _iou_range;}
    bool is_entire_iou(){return _entire_iou;}
    void set_meta_data(std::shared_ptr<MetaDataReader> meta_data_reader) override;
    pCropCord get_crop_cord(const std::string &image_names) override;
    RandomBBoxCropReader();
    ~RandomBBoxCropReader() override {}
private:
    std::shared_ptr<RaliRandomCropParam> _meta_crop_param;
    std::shared_ptr<COCOMetaDataReader> _meta_data_reader = nullptr;
    std::map<std::string, std::shared_ptr<BoundingBox>> _meta_bbox_map_content;
    std::vector<uint> _crop_width_val, _crop_height_val, _x1_val, _y1_val, _x2_val, _y2_val;
    std::vector<uint32_t> in_width, in_height;
    float  _threshold = 0.05;
    int _batch_size = 1;
    std::vector<std::pair<bool, float>> _iou_range;
    bool _all_boxes_overlap;
    bool _no_crop;
    bool _has_shape;
    int _crop_width;
    int _crop_height;
    int _num_of_attempts = 20;
    int _total_num_of_attempts = 0;
    bool _entire_iou = false;
    FloatParam *crop_area_factor = NULL;
    FloatParam *crop_aspect_ratio = NULL;
    constexpr static float ASPECT_RATIO_RANGE [2] = {0.5, 2.0};
    FloatParam *x_drift = NULL;
    FloatParam *y_drift = NULL;
    std::shared_ptr<RaliRandomCropParam> _crop_param;
    void add(std::string image_name, BoundingBoxCord bbox);
    bool exists(const std::string &image_name);
    std::map<std::string, std::shared_ptr<CropCord>> _map_content;
    std::map<std::string, std::shared_ptr<CropCord>>::iterator _itr;
    std::shared_ptr<Graph> _graph = nullptr;
    CropCordBatch* _output;
};

