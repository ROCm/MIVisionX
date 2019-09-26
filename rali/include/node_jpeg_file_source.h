#pragma once
#include "node.h"
#include "image_loader_multi_thread.h"
#include "graph.h"

class JpegFileNode: public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override  { if(!_graph) _graph = graph; }

    JpegFileNode(Image *output, std::shared_ptr<ImageLoaderMultiThread> loader_module,
                 RaliMemType mem_type, unsigned batch_size);

    JpegFileNode() = delete;
    void init(const std::string& source_path, size_t num_threads = NUM_THREADS);
    void update_parameters() override  {};
private:
    std::shared_ptr<ImageLoaderMultiThread> _loader_module;
    std::string  _image_dir_path;
    RaliMemType _mem_type;
    unsigned _batch_size;
    constexpr static size_t NUM_THREADS = 1;
};