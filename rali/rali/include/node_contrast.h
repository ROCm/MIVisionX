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
#include "node.h"
#include "parameter_vx.h"
#include "graph.h"

class RaliContrastNode : public Node
{
public:
    RaliContrastNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    RaliContrastNode() = delete;
    void init(int min, int max);
    void init(IntParam *min, IntParam * max);

protected:
    void create_node() override ;
    void update_node() override;

private:
    ParameterVX<int> _min;
    ParameterVX<int> _max;
    constexpr static int   CONTRAST_MIN_RANGE [2] = {0, 30};
    constexpr static int   CONTRAST_MAX_RANGE [2] = {60, 90};
};