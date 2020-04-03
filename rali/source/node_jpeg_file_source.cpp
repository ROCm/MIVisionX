#include "node_jpeg_file_source.h"
#include "exception.h"


JpegFileNode::JpegFileNode(Image *output, DeviceResources device_resources, RaliMemType mem_type,
                           unsigned batch_size) :
Node({}, {output}),
_mem_type(mem_type),
_batch_size(batch_size)
{
    _loader_module = std::make_shared<ImageLoaderMultiThread>(device_resources);
}

void JpegFileNode::init(size_t num_threads, const std::string &source_path, bool loop)
{
    if(!_loader_module)
        THROW("ERROR: loader module is not set for JpegFileNode, cannot initialize")

    _loader_module->set_thread_count(num_threads);
    _loader_module->set_output_image(_outputs[0]);
    // Set reader and decoder config accordingly for the JpegFileNode
     _loader_module->initialize(ReaderConfig(StorageType::FILE_SYSTEM, source_path, loop), DecoderConfig(DecoderType::TURBO_JPEG),
             _mem_type,
             _batch_size);
    _loader_module->start_loading();
}