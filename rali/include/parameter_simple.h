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