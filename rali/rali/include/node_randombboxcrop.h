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
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"

class RandomBBoxCropNode : public Node
{
public:
    RandomBBoxCropNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    RandomBBoxCropNode() = delete;
    void init(FloatParam *crop_area_factor, FloatParam *crop_aspect_ratio, FloatParam *x_drift, FloatParam *y_drift, int num_of_attempts, int all_boxes_overlap, int no_crop, int has_shape, int crop_width, int crop_height);
    unsigned int get_dst_width() { return _outputs[0]->info().width(); }
    unsigned int get_dst_height() { return _outputs[0]->info().height_single(); }
    std::shared_ptr<RaliRandomCropParam> get_crop_param() { return _crop_param; }
    float get_threshold(){return _threshold;}
    std::vector<std::pair<float,float>> get_iou_range(){return _iou_range;}
    bool is_entire_iou(){return _entire_iou;}
    void set_meta_data_batch() {}

protected:
    void create_node() override;
    void update_node() override;

private:
    std::shared_ptr<RaliRandomCropParam> _meta_crop_param;
    vx_array _x1, _y1, _x2, _y2;
    std::vector<uint> _crop_width_val, _crop_height_val, _x1_val, _y1_val, _x2_val, _y2_val;
    // unsigned int _dst_width, _dst_height;
    std::vector<uint32_t> in_width, in_height;
    size_t _dest_width;
    size_t _dest_height;
    float  _threshold = 0.05;
    std::vector<std::pair<float,float>> _iou_range;
    int _num_of_attempts = 20;
    int _all_boxes_overlap = 1;
    int _no_crop = 1;
    int _has_shape = 0;
    int _crop_width = 500;
    int _crop_height = 500;
    bool _entire_iou = false;
    std::shared_ptr<RaliRandomCropParam> _crop_param;
};
