#pragma once
#include <string>
#include <iostream>
#include <chrono>


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
    TimingDBG(const std::string& name, bool enable = DEFAULT_DBG_TIMING);

    //! Starts the timer
    void start();

    //! Stops the timer
    void end();

    //! Prints total elapsed time
    unsigned long long get_timing();

    unsigned count();
private:
    std::chrono::high_resolution_clock::time_point _t_start;
    std::chrono::duration<double, std::micro>  _accumulated_time;
    unsigned _count;
    const bool _enable;
    std::string _name;


};
