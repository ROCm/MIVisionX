#pragma once
#include <variant>
#include <cstdlib>
#include <stdexcept>
#include <memory>
#include <ctime>
#include <numeric>// std::inner_product, std::accumulate
#include <algorithm> // std::remove_if
#include <vector>
#include <thread>
#include "parameter.h"

template <typename T>
class UniformRand: public Parameter<T>
{
public:

    UniformRand(T start, T end)
    {
        update(start, end);
        renew();
    }

    explicit UniformRand(T start):
            UniformRand(start, start) {}

    T default_value() const override
    {
        return static_cast<T>((_start+_end)/static_cast<T>(2));
    }

    T get() override
    {
        return _updated_val;
    };
    void renew() override
    {
        if(single_value())
        {
            // If there is only a single value possible for the random variable
            // don't waste time on calling the rand function , just return it.
            _updated_val = _start;
        } else {
            _updated_val = static_cast<T>(
                    ((double) std::rand() / (double) RAND_MAX) * ((double) _end - (double) _start) + (double) _start);
        }
    }
    int update(T start, T end) {
        if(end < start)
            end = start;

        _start = start;
        _end = end;
        return 0;
    }
    bool single_value() const override
    {
        return (_start == _end);
    }
private:
    T _start;
    T _end;
    T _updated_val;

};



template <typename T>
struct CustomRand: public Parameter<T>
{

    CustomRand
    (
        const T values[],
        const double frequencies[],
        size_t size)
    {
        update(values, frequencies, size);
        renew();
    }
    int update
    (
        const T values[],
        const double frequencies[],
        size_t size
    )
    {
        if(size == 0)
            return -1;

        _values.assign(values, values+size);
        _frequencies.assign(frequencies, frequencies+size);
        _comltv_dist.resize(size, 0);
        double sum = 0;
        // filter out negative values if any, and sum it up
        std::copy_if(
            _frequencies.begin(),
            _frequencies.end(),
            _frequencies.begin(),
            [&](double in) -> double { if(in >= 0) { sum += in; return in;} return 0; });

        // NOTE: If there remains values with probability zero it may cause issues with sampling
        // TODO: Remove values associated with probabilities equal to 0 from the _frequencies and _values

        // Normalize the frequencies , so that the sum is equal to 1.0
        std::transform (
            _frequencies.begin(),
            _frequencies.end(),
            _frequencies.begin(),
            [&](double in) { return (double)in/sum;});

        //Compute the expected value by performing inner product of probs and values
        _mean = std::inner_product(
            _values.begin(),
            _values.end(),
            _frequencies.begin(), 0);

        // Create the partial sum of the probability distribution function (PDF), is used for random generation
        std::partial_sum(
            _frequencies.begin(),
            _frequencies.end(),
            _comltv_dist.begin(),
            std::plus<double>());

        return 0;
    }
    T default_value() const override
    {
        return static_cast<T>(_mean);
    }
    void renew() override
    {
        if(single_value())
        {
            // If there is only a single value possible for the random variable
            // don't waste time on calling the rand function , just return it.
            _updated_val =  _values[0];
        }
        else {
            // Generate a value between [0 1]
            double rand_val = (double) std::rand() / (double) RAND_MAX;

            // Find the iterators pointing to the first element bigger than idx
            auto it = std::upper_bound(_comltv_dist.begin(), _comltv_dist.end(), rand_val);

            // Get the index and return the associated value
            unsigned idx = std::distance(_comltv_dist.begin(), it);

            _updated_val = _values[idx];
        }
    }
    T get() override
    {
        return _updated_val;
    };

    bool single_value() const override
    {
        return (_values.size() == 1);
    }
private:
    std::vector<T> _values;//!< Values
    std::vector<double> _frequencies;//!< Probabilities
    std::vector<double> _comltv_dist;//!< commulative probabilities
    double _mean;
    T _updated_val;
};