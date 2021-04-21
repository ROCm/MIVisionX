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

#include <cstdlib>
#include <ctime>
#include "parameter_factory.h"
#include "parameter_simple.h"
ParameterFactory* ParameterFactory::_instance = nullptr;
std::mutex ParameterFactory::_mutex;

bool validate_simple_rand_param(pParam arg)
{
    bool ret = true;
    std::visit(
            [&ret](auto&& arg)
            {
                if(arg == nullptr || arg->type != RaliParameterType::DETERMINISTIC)
                    ret = false;

            },
            arg);
    return ret;
}


bool validate_custom_rand_param(pParam arg)
{
    bool ret = true;
    std::visit(
        [&ret](auto&& arg)
        {
            if(arg == nullptr || arg->type != RaliParameterType::RANDOM_CUSTOM)
                ret = false;
            
        },
        arg);
    return ret;
}

bool validate_uniform_rand_param(pParam  rand_obj)
{
    bool ret = true;
    std::visit(
        [&ret](auto&& arg)
        {
            if(arg == nullptr || arg->type != RaliParameterType::RANDOM_UNIFORM)
                ret = false;
                           
        },
        rand_obj);
    return ret;
}

ParameterFactory::ParameterFactory()
{
    _seed = 0;
}

ParameterFactory* ParameterFactory::instance() {
    
    if(_instance == nullptr)// For performance reasons
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if(_instance == nullptr) 
        {
            _instance = new ParameterFactory();
        }
    }
    return _instance;
}

ParameterFactory::~ParameterFactory() {
    for(auto&& rand_obj : _parameters)
            std::visit(
                [](auto&& arg)
                {
                    delete arg;
                },
                rand_obj);
}

void ParameterFactory::renew_parameters()
{
    for(auto&& rand_obj : _parameters)
        std::visit(
                [](auto&& arg)
                {
                    arg->renew();
                },
                rand_obj);
}

unsigned
ParameterFactory::get_seed()
{   
    return _seed;
}

void 
ParameterFactory::set_seed(unsigned seed)
{
    _seed = seed;
}

IntParam* ParameterFactory::create_uniform_int_rand_param(int start, int end)
{
    std::random_device rd;
    _seed = rd();
    auto gen = new UniformRand<int>(start, end, _seed);
    auto ret = new IntParam(gen, RaliParameterType::RANDOM_UNIFORM);
    _parameters.insert(gen);
    return ret;
}

FloatParam* ParameterFactory::create_uniform_float_rand_param(float start, float end)
{
    std::random_device rd;
    _seed = rd();
    auto gen = new UniformRand<float>(start, end, _seed);
    auto ret = new FloatParam(gen, RaliParameterType::RANDOM_UNIFORM);
    _parameters.insert(gen);
    return ret;
}


IntParam* ParameterFactory::create_custom_int_rand_param(const int *value, const double *frequencies, size_t size)
{
    std::random_device rd;
    _seed = rd();
    auto gen = new CustomRand<int>(value, frequencies, size, _seed);
    auto ret = new IntParam(gen, RaliParameterType::RANDOM_CUSTOM);
    _parameters.insert(gen);
    return ret;
}

FloatParam* ParameterFactory::create_custom_float_rand_param(const float *value, const double *frequencies, size_t size)
{
    std::random_device rd;
    _seed = rd();
    auto gen = new CustomRand<float>(value, frequencies, size, _seed);
    auto ret = new FloatParam(gen, RaliParameterType::RANDOM_CUSTOM);
    _parameters.insert(gen);
    return ret;
}


IntParam* ParameterFactory::create_single_value_int_param(int value)
{
    auto gen = new SimpleParameter<int>(value);
    auto ret = new IntParam(gen, RaliParameterType::DETERMINISTIC);
    _parameters.insert(gen);
    return ret;
}

FloatParam* ParameterFactory::create_single_value_float_param(float value)
{
    auto gen = new SimpleParameter<float>(value);
    auto ret = new FloatParam(gen, RaliParameterType::DETERMINISTIC);
    _parameters.insert(gen);
    return ret;
}

Parameter<int>* core(IntParam* arg)
{
    if(!arg)
        return nullptr;
    return arg->core;
}

Parameter<float>* core(FloatParam* arg)
{
    if(!arg)
        return nullptr;
    return arg->core;
}
