#include <commons.h>
#include "timing_debug.h"

TimingDBG::TimingDBG(const std::string& name, bool enable):
        _accumulated_time(_t_start - _t_start),
        _count(0),
        _enable(enable),
        _name(name)
{}

void TimingDBG::start()
{
    if(!_enable)
        return;

    _t_start = std::chrono::high_resolution_clock::now();
}

void TimingDBG::end()
{
    if(!_enable)
        return;

    std::chrono::high_resolution_clock::time_point t_end =
    std::chrono::high_resolution_clock::now();

    if(_t_start < t_end)
    {
        _accumulated_time = _accumulated_time + (t_end - _t_start);
        _count++;
    }
}

unsigned TimingDBG::count()
{
    return _count;
}

unsigned long long TimingDBG::get_timing()
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
