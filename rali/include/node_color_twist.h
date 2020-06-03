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
#include "graph.h"

class ColorTwistBatchNode : public Node
{
public:
    ColorTwistBatchNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    ColorTwistBatchNode() = delete;
    void init(float alpha, float beta, float hue, float sat);
    void init(FloatParam *alpha, FloatParam *beta, FloatParam *hue, FloatParam *sat);

protected:
    void create_node() override;
    void update_node() override;
private:

    ParameterVX<float> _alpha;
    ParameterVX<float> _beta;
    ParameterVX<float> _hue;
    ParameterVX<float> _sat;

    constexpr static float   ALPHA_RANGE [2] = {0.1, 1.95};
    constexpr static float   BETA_RANGE [2] = {0.1, 25.0};
    constexpr static float   HUE_RANGE [2] = {5.0, 170.0};
    constexpr static float   SAT_RANGE [2] = {0.1, 0.4};
};