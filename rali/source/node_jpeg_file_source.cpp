#include "node_jpeg_file_source.h"
#include "exception.h"


JpegFileNode::JpegFileNode(
        Image *output,
        std::shared_ptr<ImageLoaderMultiThread> loader_module,
        RaliMemType mem_type,
        unsigned batch_size):
Node({}, {output}),
_loader_module(loader_module),
_mem_type(mem_type),
_batch_size(batch_size)
{
}

void JpegFileNode::init( const std::string& source_path, size_t num_threads)
{
    if(!_loader_module)
        THROW("ERROR: loader module is not set for JpegFileNode, cannot initialize")

    _image_dir_path = source_path;

    LoaderModuleStatus status;

    _loader_module->set_thread_count(num_threads);
    _loader_module->set_path(_image_dir_path);
    _loader_module->set_output_image(_outputs[0]);
    if((status = _loader_module->create(StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, _mem_type, _batch_size)) != LoaderModuleStatus::OK)
        THROW("Adding file source input failed " + TOSTR(status))
    _loader_module->start_loading();
    // There is no reason to keep the resource remove the reference so that resource can be deallocated if needed later
    _loader_module = nullptr;
}