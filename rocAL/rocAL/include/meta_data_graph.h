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
#include <list>
#include "circular_buffer.h"
#include "meta_data.h"
#include "parameter_factory.h"
#include "node.h"
#include "meta_node.h"
#include "randombboxcrop_meta_data_reader.h"

class MetaDataGraph
{
public:
    virtual ~MetaDataGraph()= default;
    virtual void process(MetaDataBatch* meta_data) = 0;
    virtual void update_meta_data(MetaDataBatch* meta_data, decoded_image_info decoded_image_info) = 0;
    virtual void update_random_bbox_meta_data(CropCordBatch* random_bbox_crop_cords_data, MetaDataBatch* meta_data, decoded_image_info decoded_image_info) = 0;
    std::list<std::shared_ptr<MetaNode>> _meta_nodes;
};

