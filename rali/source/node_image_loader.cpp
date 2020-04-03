#include "node_image_loader.h"
#include "exception.h"


ImageLoaderNode::ImageLoaderNode(Image *output, DeviceResources device_resources):
        Node({}, {output})
{
    _loader_module = std::make_shared<ImageLoaderSharded>(device_resources);
}

void ImageLoaderNode::init(unsigned internal_shard_count, const std::string &source_path, StorageType storage_type,
                           DecoderType decoder_type, bool loop, size_t load_batch_count, RaliMemType mem_type)
{
    if(!_loader_module)
        THROW("ERROR: loader module is not set for ImageLoaderNode, cannot initialize")
    if(internal_shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    _loader_module->set_output_image(_outputs[0]);
    // Set reader and decoder config accordingly for the ImageLoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, loop);
    reader_cfg.set_shard_count(internal_shard_count);
    reader_cfg.set_batch_count(load_batch_count);
    _loader_module->initialize(reader_cfg, DecoderConfig(decoder_type),
             mem_type,
             _batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> ImageLoaderNode::get_loader_module()
{
    if(!_loader_module)
        WRN("ImageLoaderNode's loader module is null, not initialized")
    return _loader_module;
}

ImageLoaderNode::~ImageLoaderNode()
{
    _loader_module = nullptr;
}