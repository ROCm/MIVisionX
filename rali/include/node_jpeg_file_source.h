#pragma once
#include "node.h"
#include "image_loader_multi_thread.h"
#include "image_loader_configs.h"
#include "graph.h"

class JpegFileNode: public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override  { if(!_graph) _graph = graph; }
    JpegFileNode(Image *output, std::shared_ptr<ImageLoaderMultiThread> loader_module,
                          JpegFileLoaderConfig loader_config);
    JpegFileNode() = delete;
    void init(const std::string& source_path, size_t num_threads);
    void update_parameters() override  {};
private:
    std::shared_ptr<ImageLoaderMultiThread> _loader_module;
    JpegFileLoaderConfig _loader_config;
    constexpr static size_t NUM_THREADS = 1;
};