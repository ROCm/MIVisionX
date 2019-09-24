#pragma once
#include <memory>
#include <list>
#include <variant>
#include <map>
#include "graph.h"
#include "timing_debug.h"
#include "node.h"
#include "node_jpeg_file_source.h"
class MasterGraph
{
public:
    enum class Status { OK = 0, GRAPH_NOT_VERIFIED = 1 };
    MasterGraph(size_t batch_size, RaliAffinity affinity, int gpu_id, size_t cpu_threads);
    ~MasterGraph();
    Status reset_loaders();
    size_t remaining_images_count();
    MasterGraph::Status copy_output(unsigned char *out_ptr);
    MasterGraph::Status
    copy_out_tensor(float *out_ptr, RaliTensorFormat format, float multiplier, float offset,
                    bool reverse_channels);
    Status copy_output(cl_mem out_ptr, size_t out_size);
    size_t output_width();
    size_t output_height();
    size_t output_image_count();
    RaliColorFormat output_color_format();
    Status build();
    Status run();
    std::vector<long long unsigned> timing();
    RaliMemType mem_type();
    void release();
    template <typename T>
    std::shared_ptr<T> add_node(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    Image *create_image(const ImageInfo &info, bool is_output);
    Image *create_loader_output_image(const ImageInfo &info, bool is_output);

private:
    Status update_parameters();
    Status allocate_output_tensor();
    Status deallocate_output_tensor();
    DeviceManager   _device;
    ImageInfo _output_image_info;
    std::list<Image*> _output_images;//!< Keeps the ovx images that are used to store the augmented output (there is an image per augmentation branch)
    std::list<Image*> _internal_images;//!< Keeps all the ovx images (virtual/non-virtual) either intermediate images, or input images that feed the graph
    std::list<std::shared_ptr<Node>> _nodes;
    std::list<std::shared_ptr<Node>> _root_nodes;
    std::map<Image*, std::shared_ptr<Node>> _image_map;
    cl_mem _output_tensor;
    std::shared_ptr<Graph> _graph = nullptr;
    RaliAffinity _affinity;
    int _gpu_id;
    std::list<pLoaderModule> _loader_modules; //<! Keeps the loader modules used to feed the input the images of the graph
    TimingDBG _convert_time;
    size_t _batch_size;
    size_t _cpu_threads;
    vx_context _context;
    RaliMemType _mem_type;
    TimingDBG _process_time;
    bool _graph_verfied = false;
    void create_single_graph();
};

template <typename T>
std::shared_ptr<T> MasterGraph::add_node(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs)
{
    auto node = std::make_shared<T>(inputs, outputs);
    _nodes.push_back(node);

    for(auto& input: inputs)
    {
        if (_image_map.find(input) == _image_map.end())
            THROW("Input image is invalid, cannot be found among output of previously created nodes")

        auto parent_node = _image_map.find(input)->second;
        parent_node->add_next(node);
        node->add_previous(parent_node);
    }

    for(auto& output: outputs)
        _image_map.insert(make_pair(output, node));

    return node;
}

/*
 * Explicit specialization for JpegFileNode
 */
template<> inline std::shared_ptr<JpegFileNode> MasterGraph::add_node(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs)
{
    auto loader_module = std::make_shared<ImageLoaderMultiThread>(_device.resources());
    auto node = std::make_shared<JpegFileNode>(outputs[0], loader_module, JpegFileLoaderConfig(_batch_size, _mem_type));

    _loader_modules.push_back(loader_module);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _image_map.insert(make_pair(output, node));

    return node;
}