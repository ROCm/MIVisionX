#pragma once

template <typename T>
class Parameter
{
public:
    virtual T default_value() const = 0;
    ///
    /// \return returns the updated value of the parameter
    virtual T get()  = 0;
    ///
    /// \return most recent value returned by calling the get function, does not return the updated value
    virtual T most_recent_used() const = 0;
    virtual ~Parameter() {}
    ///
    /// \return returns if this parameter takes a single value (vs a range of values or many values)
    virtual bool single_value() const = 0;
};

