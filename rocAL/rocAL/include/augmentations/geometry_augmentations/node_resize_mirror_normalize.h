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
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"

class ResizeMirrorNormalizeNode : public Node
{
public:
    ResizeMirrorNormalizeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    ResizeMirrorNormalizeNode() = delete;
    void init(float mean, float std_dev, IntParam *mirror);
    vx_array get_dst_width() { return _dst_roi_width; }
    vx_array get_dst_height() { return _dst_roi_height;}
    vx_array get_src_width() { return _src_roi_width; }
    vx_array get_src_height() { return _src_roi_height; }
    vx_array return_mirror(){ return _mirror.default_array();  }
protected:
    void create_node() override;
    void update_node() override;
private:
    vx_array  _dst_roi_width , _dst_roi_height ;
    std::vector<uint> _dest_width_val, _dest_height_val;
    std::vector<vx_float32> _mean_vx, _std_dev_vx;
    vx_array _mean_array, _std_dev_array;
    float _mean; 
    float _std_dev; 
    ParameterVX<int> _mirror;
    constexpr static int   MIRROR_RANGE [2] =  {0, 1};
};