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
#include <set>
#include <memory>
#include "bounding_box_graph.h"
#include "meta_data.h"
#include "node.h"
#include "node_random_crop.h"
#include "parameter_vx.h"

class SSDRandomCropMetaNode : public MetaNode
{
public:
    SSDRandomCropMetaNode(){};
    void update_parameters(MetaDataBatch *input_meta_data) override;
    std::shared_ptr<RandomCropNode> _node = nullptr;
    std::vector<uint32_t> in_width, in_height;
    void set_threshold(float threshold) { _threshold = threshold; }
    Parameter<float> *area_factor;
    Parameter<float> *aspect_ratio_factor;
    Parameter<float> *x_drift_factor;
    Parameter<float> *y_drift_factor;

private:
    void initialize();
    BoundingBoxCord generate_random_crop(int img_idx);
    std::shared_ptr<RaliRandomCropParam> _meta_crop_param;
    vx_array _crop_width, _crop_height, _x1, _y1, _x2, _y2;
    std::vector<uint> _crop_width_val, _crop_height_val, _x1_val, _y1_val, _x2_val, _y2_val;
    unsigned int _dst_width, _dst_height;
    float _threshold = 0.5;
    int   _num_of_attempts = 20;
    constexpr static float ASPECT_RATIO_RANGE[2] = {0.7500, 1.333};
};
