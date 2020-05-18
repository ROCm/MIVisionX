#pragma once

#include "commons.h"
#include "graph.h"
#include "master_graph.h"


struct Context
{
    explicit Context(size_t batch_size, RaliAffinity affinity, int gpu_id , size_t cpu_thread_count ):
    affinity(affinity),
    _user_batch_size(batch_size)
    {
        LOG("Processing on " + STR(((affinity == RaliAffinity::CPU)?" CPU": " GPU")))
        master_graph = std::make_shared<MasterGraph>(batch_size, affinity, gpu_id, cpu_thread_count);
        _internal_batch_size = master_graph->internal_batch_size();
    }
    ~Context()
    {
        clear_errors();
    };
    std::shared_ptr<MasterGraph> master_graph;

    RaliAffinity affinity;
    bool no_error() { return error.empty(); }
    const char* error_msg() { return error.c_str(); }
    void capture_error(const std::string& err_msg) { error = err_msg; }
    Timing timing()
    {
        return master_graph->timing();
    }
    size_t user_batch_size() { return _user_batch_size; }
    size_t internal_batch_size() { return _internal_batch_size; }
private:
    void clear_errors() { error = "";}
    std::string error;
    size_t _user_batch_size;
    size_t _internal_batch_size;
};