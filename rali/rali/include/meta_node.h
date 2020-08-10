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
#include "meta_data_graph.h"
#include "meta_data.h"
#include "node.h"
#include "parameter_factory.h"

class MetaNode
{
public:
    MetaNode() {}
    virtual ~MetaNode() {};
    virtual void update_parameters(MetaDataBatch* input_meta_data) = 0;
    double BBoxIntersectionOverUnion(const BoundingBoxCord &box1, const BoundingBoxCord &box2, bool is_iou) const;
    int _batch_size;
    float _iou_threshold = 0.25;
};

inline double MetaNode::BBoxIntersectionOverUnion(const BoundingBoxCord &box1, const BoundingBoxCord &box2, bool is_iou = false) const
{
    double iou;
    float xA = std::max(box1.x, box2.x);
    float yA = std::max(box1.y, box2.y);
    float xB = std::min(box1.x + box1.w, box2.x + box2.w);
    float yB = std::min(box1.y + box1.h, box2.y + box2.h);

    float intersection_area = std::max((float)0.0, xB - xA) * std::max((float)0.0, yB - yA);

    float box1_area = box1.h * box1.w;
    float box2_area = box2.h * box2.w;

    if(is_iou)
        iou = intersection_area / float(box1_area + box2_area - intersection_area);
    else
        iou = intersection_area / float(box1_area);

    return iou;
}

