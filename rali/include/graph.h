#pragma once
#include "commons.h"
#include <VX/vx.h>
#include <VX/vx_types.h>


class Graph
{
public:
    enum class Status { OK = 0 };
    Graph(vx_context context, RaliAffinity affinity, int cpu_id = 0, int gpu_id = 0 );
    Status verify();
    Status process();
    Status release();
    vx_graph get() { return _graph; }
private:
    RaliMemType _mem_type;
    vx_context  _context = nullptr;
    vx_graph    _graph = nullptr;
    RaliAffinity _affinity;
    int _gpu_id;
    int _cpu_id;
};