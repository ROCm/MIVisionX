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

#include "commons.h"
#include "rali_api.h"
#include "parameter_factory.h"

void RALI_API_CALL 
raliSetSeed(unsigned seed)
{
    ParameterFactory::instance()->set_seed(seed);
}

unsigned RALI_API_CALL
raliGetSeed() 
{
    return ParameterFactory::instance()->get_seed();
}

int RALI_API_CALL
raliGetIntValue(RaliIntParam p_obj)
{
    auto obj = static_cast<IntParam*>(p_obj);
    return obj->core->get();
}

float RALI_API_CALL
raliGetFloatValue(RaliFloatParam p_obj)
{
    auto obj = static_cast<FloatParam*>(p_obj);
    return obj->core->get();
}

RaliIntParam  RALI_API_CALL
raliCreateIntUniformRand(
    int start, 
    int end) 
{
    return ParameterFactory::instance()->create_uniform_int_rand_param(start, end);
}

RaliStatus  RALI_API_CALL
raliUpdateIntUniformRand(
        int start,
        int end,
        RaliIntParam p_input_obj)
{
    auto input_obj = static_cast<IntParam*>(p_input_obj);
    if(!validate_uniform_rand_param(input_obj))  {
        ERR("raliUpdateIntUniformRand : not a UniformRand object!");
        return RALI_INVALID_PARAMETER_TYPE;
    }

    UniformRand<int>* obj;
    if((obj = dynamic_cast<UniformRand<int>*>(input_obj->core)) == nullptr)
        return RALI_INVALID_PARAMETER_TYPE;

    return (obj->update(start, end) == 0) ? RALI_OK : RALI_UPDATE_PARAMETER_FAILED ;
}    

RaliFloatParam  RALI_API_CALL
raliCreateFloatUniformRand(
    float start, 
    float end) 
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(start, end);
}

RaliStatus  RALI_API_CALL
raliUpdateFloatUniformRand(
        float start,
        float end,
        RaliFloatParam p_input_obj)
{
    auto input_obj = static_cast<FloatParam*>(p_input_obj);
    if(!validate_uniform_rand_param(input_obj)) {
        ERR("raliUpdateFloatUniformRand : not a uniform random object!");
        return RALI_INVALID_PARAMETER_TYPE;
    }

    UniformRand<float>* obj;
    if((obj = dynamic_cast<UniformRand<float>*>(input_obj->core)) == nullptr)
        return RALI_INVALID_PARAMETER_TYPE;

    return (obj->update(start, end) == 0) ? RALI_OK : RALI_UPDATE_PARAMETER_FAILED ;
                                                    
}   

RaliFloatParam  RALI_API_CALL
raliCreateFloatRand(
        const float *values,
        const double *frequencies,
        unsigned size)
{
    return ParameterFactory::instance()->create_custom_float_rand_param(values,
                                                                        frequencies,
                                                                        size);
}

RaliFloatParam  RALI_API_CALL
raliCreateFloatParameter(float val)
{
    return ParameterFactory::instance()->create_single_value_float_param(val);
}

RaliIntParam  RALI_API_CALL
raliCreateIntParameter(int val)
{
    return ParameterFactory::instance()->create_single_value_int_param(val);
}

RaliStatus  RALI_API_CALL
raliUpdateIntParameter(int new_val, RaliIntParam p_input_obj)
{
    auto input_obj = static_cast<IntParam*>(p_input_obj);
    if(!validate_simple_rand_param(input_obj)) {
        ERR("raliUpdateIntParameter : not a custom random object!");
        return RALI_INVALID_PARAMETER_TYPE;
    }

    SimpleParameter<int>* obj;
    if((obj = dynamic_cast<SimpleParameter<int>*>(input_obj->core)) == nullptr)
        return RALI_INVALID_PARAMETER_TYPE;
    return (obj->update(new_val) == 0) ? RALI_OK : RALI_UPDATE_PARAMETER_FAILED;
}

RaliStatus  RALI_API_CALL
raliUpdateFloatParameter(float new_val, RaliFloatParam p_input_obj)
{
    auto input_obj = static_cast<FloatParam*>(p_input_obj);
    if(!validate_simple_rand_param(input_obj)) {
        ERR("raliUpdateFloatParameter : not a custom random object!");
        return RALI_INVALID_PARAMETER_TYPE;
    }

    SimpleParameter<float>* obj;
    if((obj = dynamic_cast<SimpleParameter<float>*>(input_obj->core)) == nullptr)
        return RALI_INVALID_PARAMETER_TYPE;
    return (obj->update(new_val) == 0) ? RALI_OK : RALI_UPDATE_PARAMETER_FAILED;
}

RaliStatus  RALI_API_CALL
raliUpdateFloatRand(
        const float *values,
        const double *frequencies,
        unsigned size,
        RaliFloatParam p_updating_obj)
{
    auto updating_obj = static_cast<FloatParam*>(p_updating_obj);
    if(!validate_custom_rand_param(updating_obj)) {
        ERR("raliUpdateFloatRand : not a custom random object!");
        return RALI_INVALID_PARAMETER_TYPE;
    }
    
    CustomRand<float>* obj;
    if((obj = dynamic_cast<CustomRand<float>*>(updating_obj->core)) == nullptr)
        return RALI_INVALID_PARAMETER_TYPE;

    return (obj->update(values, frequencies, size) == 0) ? RALI_OK : RALI_UPDATE_PARAMETER_FAILED ;
} 

RaliIntParam  RALI_API_CALL
raliCreateIntRand(
        const int *values,
        const double *frequencies,
        unsigned size)
{
    return ParameterFactory::instance()->create_custom_int_rand_param(values,
                                                                      frequencies,
                                                                      size);
}

RaliStatus  RALI_API_CALL
raliUpdateIntRand(
        const int *values,
        const double *frequencies,
        unsigned size,
        RaliIntParam p_updating_obj)
{
    auto updating_obj = static_cast<IntParam*>(p_updating_obj);
    if(!validate_custom_rand_param(updating_obj)) {
        ERR("raliUpdateIntRand : not a CustomRand object!");
        return RALI_INVALID_PARAMETER_TYPE;
    }

    CustomRand<int>* obj;
    if((obj = dynamic_cast<CustomRand<int>*>(updating_obj->core)) == nullptr)
        return RALI_INVALID_PARAMETER_TYPE;

    return (obj->update( values, frequencies, size) == 0) ? RALI_OK : RALI_UPDATE_PARAMETER_FAILED ;
} 
