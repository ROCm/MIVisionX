
#include "image_loader_configs.h"
#include "node_jpeg_file_source.h"
#include "exception.h"


JpegFileNode::JpegFileNode(
        Image *output,
        std::shared_ptr<ImageLoaderMultiThread> loader_module,
        JpegFileLoaderConfig loader_config):
Node({}, {output}),
_loader_module(loader_module),
_loader_config(loader_config)
{
}

void JpegFileNode::init( const std::string& source_path, size_t num_threads)
{
    if(!_loader_module)
        THROW("ERROR: loader module is not set for JpegFileNode, cannot initialize")

    _loader_config.path = source_path;

    LoaderModuleStatus status;
    _loader_module->set_thread_count(num_threads);

    if((status = _loader_module->create(&_loader_config)) !=  LoaderModuleStatus::OK)
        THROW("Adding file source input failed " + TOSTR(status))

    if( (status = _loader_module->set_output_image(_outputs[0])) !=  LoaderModuleStatus::OK)
        THROW("ERROR: Adding Jpeg file source input failed "+TOSTR(status));
    // There is no reason to keep the resourcesm remove the reference so that resource can be deallocated if needed later
    _loader_module = nullptr;
}
