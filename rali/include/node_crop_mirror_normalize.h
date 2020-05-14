
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
#include "parameter_vx.h"
#include "parameter_crop_resize.h"

class CropMirrorNormalizeNode : public Node
{
public:
    CropMirrorNormalizeNode(const std::vector<Image *> &inputs,
                            const std::vector<Image *> &outputs);
    CropMirrorNormalizeNode() = delete;
    void init(int crop_h, int crop_w, float start_x, float start_y, float mean, float std_dev, IntParam *mirror);
protected:
    void create_node() override ;
    void update_node() override;
private:
    vx_array _src_width_array, _src_height_array;
    std::vector<vx_uint32> _x1, _x2, _y1, _y2;
    std::vector<vx_float32> _mean_vx, _std_dev_vx;
    vx_array _mean_array, _std_dev_array;
    vx_array _x1_array, _x2_array, _y1_array, _y2_array;
    int _crop_h;
    int _crop_w;
    int _crop_d;
    float _mean; // vector of means in future
    float _std_dev; // vector of std_devs in future
    ParameterVX<int> _mirror; // Should come from int random number generator with values 1 or 0 - Coin Flip
    constexpr static int   MIRROR_RANGE [2] =  {0, 1};
};
