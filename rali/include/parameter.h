#pragma once

template <typename T>
class Parameter
{
public:
    virtual T default_value() const = 0;
    ///
    /// \return returns the updated value of the parameter
    virtual T get()  = 0;

    /// used to internally renew state of the parameter if needed (for random parameters)
    virtual void renew() {};

    virtual ~Parameter() {}
    ///
    /// \return returns if this parameter takes a single value (vs a range of values or many values)
    virtual bool single_value() const = 0;
};

