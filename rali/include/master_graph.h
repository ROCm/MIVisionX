#pragma once
#include <memory>
#include <list>
#include <variant>
#include <map>
#include "graph.h"
#include "ring_buffer.h"
#include "timing_debug.h"
#include "node.h"
#include "node_jpeg_file_source.h"
class MasterGraph
{
public:
    enum class Status { OK = 0,  NOT_IMPLEMENTED };
    MasterGraph(size_t batch_size, RaliAffinity affinity, int gpu_id, size_t cpu_threads);
    ~MasterGraph();
    Status reset_loaders();
    size_t remaining_images_count();
    MasterGraph::Status copy_output(unsigned char *out_ptr);
    MasterGraph::Status
    copy_out_tensor(float *out_ptr, RaliTensorFormat format, float multiplier0, float multiplier1, float multiplier2,
                    float offset0, float offset1, float offset2, bool reverse_channels);
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
    Image *create_loader_output_image(const ImageInfo &info);

private:
    Status update_node_parameters();
    Status allocate_output_tensor();
    Status deallocate_output_tensor();
    void create_single_graph();
    void start_processing();
    void stop_processing();
    void output_routine();
    RingBuffer _ring_buffer;
    std::thread _output_thread;
    DeviceManager   _device;
    ImageInfo _output_image_info;
    std::vector<Image*> _output_images;//!< Keeps the ovx images that are used to store the augmented output (there is an image per augmentation branch)
    std::list<Image*> _internal_images;//!< Keeps all the ovx images (virtual/non-virtual) either intermediate images, or input images that feed the graph
    std::list<Image*> _loader_image;//!< keeps images that used in the loader modules to update the input to the graph
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
    bool _first_run = true;
    bool _processing;
    const static unsigned OUTPUT_RING_BUFFER_DEPTH = 3;
    std::mutex _count_lock;
    unsigned _in_process_count;
    size_t internal_image_count();
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
    auto node = std::make_shared<JpegFileNode>(outputs[0], _device.resources(),  _mem_type, _batch_size);
    _loader_image.push_back(outputs[0]);
    _loader_modules.push_back(node->get_loader_module());
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _image_map.insert(make_pair(output, node));

    return node;
}