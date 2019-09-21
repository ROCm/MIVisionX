#pragma once

#include "commons.h"
#include "graph.h"
#include "master_graph.h"


struct Context
{
    explicit Context(size_t batch_size, RaliAffinity affinity, int gpu_id , size_t cpu_thread_count ):
    batch_size(batch_size),
    affinity(affinity)
    {
        LOG("Processing on " + STR(((affinity == RaliAffinity::CPU)?" CPU": " GPU")))
        master_graph = std::make_shared<MasterGraph>(batch_size, affinity, gpu_id, cpu_thread_count);
    }
    ~Context()
    {
        clear_errors();
    };
    std::shared_ptr<MasterGraph> master_graph;
    size_t batch_size;
    RaliAffinity affinity;
    bool no_error() { return error.empty(); }
    const char* error_msg() { return error.c_str(); }
    void capture_error(const std::string& err_msg) { error = err_msg; }
    std::vector<long long unsigned> timing()
    {
        return master_graph->timing();
    }
private:
    void clear_errors() { error = "";}
    std::string error;
};