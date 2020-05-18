#pragma once
#include <string>
#include <iostream>
#include <chrono>
#include <utility>
#include "commons.h"


#define DEFAULT_DBG_TIMING 1
/*! \brief Debugging RaliDbgTiming class
* 
* Can be used anywhere in the code for adding RaliDbgTiming for debugging and profiling 
*/
class TimingDBG {
public:
    //! Constrcutor
    /*!
    \param name Name of the timer, 
    \param enable enables the timer module, if not set, timer is disabled
    */
    explicit TimingDBG(std::string  name, bool enable = DEFAULT_DBG_TIMING):
            _accumulated_time(_t_start - _t_start),
            _count(0),
            _enable(enable),
            _name(std::move(name))
    {}

    //! Starts the timer
    inline
    void start()
    {
        if(!_enable)
            return;

        _t_start = std::chrono::high_resolution_clock::now();
    }

    //! Stops the timer
    inline
    void end()
    {
        if(!_enable)
            return;

        std::chrono::high_resolution_clock::time_point t_end =
                std::chrono::high_resolution_clock::now();

        if(_t_start < t_end)
        {
            _instantaneous_time = t_end - _t_start;
            _accumulated_time = _accumulated_time + _instantaneous_time;
            _count++;
        }
    }

    //! Prints total elapsed time
    unsigned long long get_timing()
    {
        if(!_enable)
            return 0;

        auto dur = static_cast<long long unsigned> (std::chrono::duration_cast<std::chrono::microseconds>(_accumulated_time).count());
        if(_count > 0)
            LOG (_name + " ran " + TOSTR(_count) + " times with average duration " + TOSTR((dur / _count) / 1000000 ) + " sec " + TOSTR((dur / _count) % 1000000) + " us ")
        _count = 0;
        _accumulated_time = _t_start - _t_start;
        return dur;
    }

    //! Returns last timing
    unsigned long long get_last_timing()
    {
        if(!_enable)
            return 0;
        auto dur = static_cast<long long unsigned> (std::chrono::duration_cast<std::chrono::microseconds>(_accumulated_time).count());
        return dur;
    }


    unsigned count()
    {
        return _count;
    }
private:
    std::chrono::high_resolution_clock::time_point _t_start;
    std::chrono::duration<double, std::micro>  _accumulated_time = _t_start - _t_start;
    std::chrono::duration<double, std::micro> _instantaneous_time = _t_start - _t_start;
    unsigned _count;
    const bool _enable;
    std::string _name;


};