#pragma once
#include <VX/vx_types.h>
#include <VX/vx_compatibility.h>
#include "parameter_factory.h"

template<typename T>
class ParameterVX
{
public:
    ParameterVX(unsigned ovx_param_idx, T default_range_start, T default_range_end):
            OVX_PARAM_IDX(ovx_param_idx),
            _DEFAULT_RANGE_START(default_range_start),
            _DEFAULT_RANGE_END(default_range_end)
    {
        _param = ParameterFactory::instance()->create_uniform_rand_param<T>(_DEFAULT_RANGE_START,
                                                                            _DEFAULT_RANGE_END);
    }
    void create(vx_node node)
    {
        vx_status status;
        auto ref = vxGetParameterByIndex(node, OVX_PARAM_IDX);
        if( (status = vxQueryParameter(ref,VX_PARAMETER_ATTRIBUTE_REF, &_scalar, sizeof(vx_scalar))) != VX_SUCCESS ||
            (status = vxGetStatus((vx_reference)node)) != VX_SUCCESS)
            THROW("Getting vx scalar from the vx node failed" + TOSTR(status));
        if( (status = vxReadScalarValue(_scalar, &_val)) != VX_SUCCESS)
            THROW("Reading vx scalar failed" + TOSTR(status));
    }
    void set_param(Parameter<T>* param)
    {
        if(!param)
            return;

        ParameterFactory::instance()->destroy_param(_param);
        _param = param;
    }
    void set_param(T val)
    {
        ParameterFactory::instance()->destroy_param(_param);
        _param = ParameterFactory::instance()->create_single_value_param(val);
    }
    T default_value()
    {
        return _param->default_value();
    }
    T get()
    {
        return _val;
    }
    void update()
    {
        vx_status status;

        T val = _param->get();

        if(_val == val)
            return;

        if((status = vxWriteScalarValue(_scalar, &val))!= VX_SUCCESS)
            WRN("Updating vx scalar failed")

    }
private:
    vx_scalar _scalar;
    Parameter<T>* _param;
    T _val;
    const unsigned OVX_PARAM_IDX;
    const T _DEFAULT_RANGE_START;
    const T _DEFAULT_RANGE_END;
};
