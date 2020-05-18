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
