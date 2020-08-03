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
#include "parameter.h"

template <typename T>
class SimpleParameter : public Parameter<T>
{
public:
    explicit SimpleParameter(T value)
    {
        update(value);
    }
    T default_value() const override
    {
        return _val;
    }
    T get() override
    {
        return _val;
    }
    int update(T new_val)
    {
        _val = new_val;
        return 0;
    }

    ~SimpleParameter() = default;

    bool single_value() const override
    {
        return true;
    }
private:
    T _val;
};
using pIntParam = std::shared_ptr<SimpleParameter<int>>;
using pFloatParam = std::shared_ptr<SimpleParameter<float>>;

inline pIntParam create_simple_int_param(int val)
{
    return std::make_shared<SimpleParameter<int>>(val);
}

inline pFloatParam create_simple_float_param(float val)
{
    return std::make_shared<SimpleParameter<float>>(val);
}
