#include "node_jpeg_file_source.h"
#include "exception.h"


JpegFileNode::JpegFileNode(
        Image *output,
        DeviceResources device_resources,
        RaliMemType mem_type,
        unsigned batch_size):
Node({}, {output}),
_mem_type(mem_type),
_batch_size(batch_size)
{
    _loader_module = std::make_shared<ImageLoaderMultiThread>(device_resources);
}

void JpegFileNode::init( const std::string& source_path, size_t num_threads)
{
    if(!_loader_module)
        THROW("ERROR: loader module is not set for JpegFileNode, cannot initialize")

    _loader_module->set_thread_count(num_threads);
    _loader_module->set_path(source_path);
    _loader_module->set_output_image(_outputs[0]);
    //auto reader_config = FileSourceReaderConfig(source_path, );
     _loader_module->initialize(StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG,
             _mem_type,
             _batch_size);
    _loader_module->start_loading();
}