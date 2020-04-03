#pragma once
#include "node.h"
#include "image_loader_multi_thread.h"
#include "graph.h"

class JpegFileNode: public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override  { if(!_graph) _graph = graph; }

    JpegFileNode(Image *output, DeviceResources device_resources, RaliMemType mem_type,
                 unsigned batch_size);
    ~JpegFileNode() override
    {
        _loader_module = nullptr;
    }
    JpegFileNode() = delete;
    void init(size_t num_threads, const std::string &source_path, bool loop);
    void update_parameters() override  {};
    std::shared_ptr<ImageLoaderMultiThread> get_loader_module() { return _loader_module; }
private:
    std::shared_ptr<ImageLoaderMultiThread> _loader_module;
    RaliMemType _mem_type;
    unsigned _batch_size;
    constexpr static size_t NUM_THREADS = 1;
};