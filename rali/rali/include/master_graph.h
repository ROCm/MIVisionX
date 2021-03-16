/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include <memory>
#include <list>
#include <variant>
#include <map>
#include "graph.h"
#include "ring_buffer.h"
#include "timing_debug.h"
#include "node.h"
#include "node_image_loader.h"
#include "node_image_loader_single_shard.h"
#include "node_fused_jpeg_crop.h"
#include "node_fused_jpeg_crop_single_shard.h"
#include "node_cifar10_loader.h"
#include "meta_data_reader.h"
#include "meta_data_graph.h"
#include "randombboxcrop_meta_data_reader.h"

class MasterGraph
{
public:
    enum class Status { OK = 0,  NOT_RUNNING = 1, NO_MORE_DATA = 2, NOT_IMPLEMENTED };
    MasterGraph(size_t batch_size, RaliAffinity affinity, int gpu_id, size_t cpu_threads);
    ~MasterGraph();
    Status reset();
    size_t remaining_images_count();
    MasterGraph::Status copy_output(unsigned char *out_ptr);
    MasterGraph::Status
    copy_out_tensor(void *out_ptr, RaliTensorFormat format, float multiplier0, float multiplier1, float multiplier2,
                    float offset0, float offset1, float offset2, bool reverse_channels, RaliTensorDataType output_data_type);
    Status copy_output(cl_mem out_ptr, size_t out_size);
    Status copy_out_tensor_planar(void *out_ptr, RaliTensorFormat format, float multiplier0, float multiplier1, float multiplier2,
                    float offset0, float offset1, float offset2, bool reverse_channels, RaliTensorDataType output_data_type);
    size_t output_width();
    size_t output_height();
    size_t output_byte_size();
    size_t output_depth();
    size_t augmentation_branch_count();
    size_t output_sample_size();
    RaliColorFormat output_color_format();
    Status build();
    Status run();
    Timing timing();
    RaliMemType mem_type();
    void release();
    template <typename T>
    std::shared_ptr<T> add_node(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    template <typename T, typename M> std::shared_ptr<T> meta_add_node(std::shared_ptr<M> node);
    Image *create_image(const ImageInfo &info, bool is_output);
    Image *create_loader_output_image(const ImageInfo &info);
    MetaDataBatch *create_label_reader(const char *source_path, MetaDataReaderType reader_type);
    MetaDataBatch *create_coco_meta_data_reader(const char *source_path, bool is_output);
    MetaDataBatch *create_tf_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type,  MetaDataType label_type, const std::map<std::string, std::string> feature_key_map);   
    MetaDataBatch *create_caffe_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type,  MetaDataType label_type);
    MetaDataBatch *create_caffe2_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type,  MetaDataType label_type);
    MetaDataBatch* create_cifar10_label_reader(const char *source_path, const char *file_prefix);
    void create_randombboxcrop_reader(RandomBBoxCrop_MetaDataReaderType reader_type, RandomBBoxCrop_MetaDataType label_type, bool all_boxes_overlap, bool no_crop, FloatParam* aspect_ratio, bool has_shape, int crop_width, int crop_height, int num_attempts, FloatParam* scaling, int total_num_attempts);
    const std::pair<ImageNameBatch,pMetaDataBatch>& meta_data();
    void set_loop(bool val) { _loop = val; }
    bool empty() { return (remaining_images_count() < _user_batch_size); }
    size_t internal_batch_size() { return _internal_batch_size; }
    std::shared_ptr<MetaDataGraph> meta_data_graph() { return _meta_data_graph; }
    std::shared_ptr<MetaDataReader> meta_data_reader() { return _meta_data_reader; }
    bool is_random_bbox_crop() {return _is_random_bbox_crop; }
private:
    Status update_node_parameters();
    Status allocate_output_tensor();
    Status deallocate_output_tensor();
    void create_single_graph();
    void start_processing();
    void stop_processing();
    void output_routine();
    void decrease_image_count();
    bool processing_on_device() { return _output_image_info.mem_type() == RaliMemType::OCL; };
    /// notify_user_thread() is called when the internal processing thread is done with processing all available images
    void notify_user_thread();
    /// no_more_processed_data() is logically linked to the notify_user_thread() and is used to tell the user they've already consumed all the processed images
    bool no_more_processed_data();
    RingBuffer _ring_buffer;//!< The queue that keeps the images that have benn processed by the internal thread (_output_thread) asynchronous to the user's thread
    MetaDataBatch* _augmented_meta_data = nullptr;//!< The output of the meta_data_graph,
    CropCordBatch* _random_bbox_crop_cords_data = nullptr;
    std::thread _output_thread;
    DeviceManager   _device;//!< Keeps the device related constructs needed for running on GPU
    ImageInfo _output_image_info;//!< Keeps the information about RALI's output image , it includes all images of a batch stacked on top of each other
    std::vector<Image*> _output_images;//!< Keeps the ovx images that are used to store the augmented output (there is an image per augmentation branch)
    std::list<Image*> _internal_images;//!< Keeps all the ovx images (virtual/non-virtual) either intermediate images, or input images that feed the graph
    std::list<std::shared_ptr<Node>> _nodes;//!< List of all the nodes
    std::list<std::shared_ptr<Node>> _root_nodes;//!< List of all root nodes (image/video loaders)
    std::list<std::shared_ptr<Node>> _meta_data_nodes;//!< List of nodes where meta data has to be updated after augmentation
    std::map<Image*, std::shared_ptr<Node>> _image_map;//!< key: image, value : Parent node
    cl_mem _output_tensor;//!< In the GPU processing case , is used to convert the U8 samples to float32 before they are being transfered back to host
    std::shared_ptr<Graph> _graph = nullptr;
    const RaliAffinity _affinity;
    const int _gpu_id;//!< Defines the device id used for processing
    pLoaderModule _loader_module; //!< Keeps the loader module used to feed the input the images of the graph
    TimingDBG _convert_time;
    const size_t _user_batch_size;//!< Batch size provided by the user
    const size_t _cpu_threads;//!< Not in use
    vx_context _context;
    const RaliMemType _mem_type;//!< Is set according to the _affinity, if GPU, is set to CL, otherwise host
    TimingDBG _process_time;
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    std::shared_ptr<MetaDataGraph> _meta_data_graph = nullptr;
    std::shared_ptr<RandomBBoxCrop_MetaDataReader> _randombboxcrop_meta_data_reader = nullptr;
    bool _first_run = true;
    bool _processing;//!< Indicates if internal processing thread should keep processing or not
    const static unsigned OUTPUT_RING_BUFFER_DEPTH = 3;
    const static unsigned SAMPLE_SIZE = sizeof(unsigned char);
    int _remaining_images_count;//!< Keeps the count of remaining images yet to be processed for the user,
    bool _loop;//!< Indicates if user wants to indefinitely loops through images or not
    static size_t compute_optimum_internal_batch_size(size_t user_batch_size, RaliAffinity affinity);
    const size_t _internal_batch_size;//!< In the host processing case , internal batch size can be different than _user_batch_size. This batch size used internally throughout.
    const size_t _user_to_internal_batch_ratio;
    bool _output_routine_finished_processing = false;
    bool _is_random_bbox_crop = false;
};

template <typename T>
std::shared_ptr<T> MasterGraph::add_node(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs)
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

template <typename T, typename M>
std::shared_ptr<T> MasterGraph::meta_add_node(std::shared_ptr<M> node)
{
    auto meta_node = std::make_shared<T>();
    _meta_data_graph->_meta_nodes.push_back(meta_node);
    meta_node->_node = node;
    meta_node->_batch_size = _user_batch_size;
    return meta_node;
}


/*
 * Explicit specialization for ImageLoaderNode
 */
template<> inline std::shared_ptr<ImageLoaderNode> MasterGraph::add_node(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<ImageLoaderNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _image_map.insert(make_pair(output, node));

    return node;
}
template<> inline std::shared_ptr<ImageLoaderSingleShardNode> MasterGraph::add_node(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<ImageLoaderSingleShardNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _image_map.insert(make_pair(output, node));

    return node;
}
template<> inline std::shared_ptr<FusedJpegCropNode> MasterGraph::add_node(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<FusedJpegCropNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _loader_module->set_random_bbox_data_reader(_randombboxcrop_meta_data_reader);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _image_map.insert(make_pair(output, node));

    return node;
}

template<> inline std::shared_ptr<FusedJpegCropSingleShardNode> MasterGraph::add_node(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<FusedJpegCropSingleShardNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _loader_module->set_random_bbox_data_reader(_randombboxcrop_meta_data_reader);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _image_map.insert(make_pair(output, node));

    return node;
}

/*
 * Explicit specialization for Cifar10LoaderNode
 */
template<> inline std::shared_ptr<Cifar10LoaderNode> MasterGraph::add_node(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<Cifar10LoaderNode>(outputs[0], _device.resources());
    _loader_module = node->get_loader_module();
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _image_map.insert(make_pair(output, node));

    return node;
}

#ifdef RALI_VIDEO
/*
 * Explicit specialization for VideoFileNode
 */
template<> inline std::shared_ptr<VideoFileNode> MasterGraph::add_node(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs, const size_t batch_size)
{
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
    auto node = std::make_shared<VideoFileNode>(inputs,outputs);
    _nodes.push_back(node);
    auto loader = std::make_shared<VideoLoaderModule>(node);
    _loader_module = loader;
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _image_map.insert(make_pair(output, node));

    return node;
}
#endif
