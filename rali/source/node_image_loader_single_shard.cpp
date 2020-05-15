#include "node_image_loader_single_shard.h"
#include "exception.h"


ImageLoaderSingleShardNode::ImageLoaderSingleShardNode(Image *output, DeviceResources device_resources):
        Node({}, {output})
{
    _loader_module = std::make_shared<ImageLoader>(device_resources);
}

void
ImageLoaderSingleShardNode::init(unsigned shard_id, unsigned shard_count, const std::string &source_path,
                                 StorageType storage_type, DecoderType decoder_type, bool loop,
                                 size_t load_batch_count, RaliMemType mem_type, bool decoder_keep_orig, bool shuffle)
{
    if(!_loader_module)
        THROW("ERROR: loader module is not set for ImageLoaderNode, cannot initialize")
    if(shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    if(shard_id >= shard_count)
        THROW("Shard is should be smaller than shard count")
    _loader_module->set_output_image(_outputs[0]);
    // Set reader and decoder config accordingly for the ImageLoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, loop);
    reader_cfg.set_shard_count(shard_count);
    reader_cfg.set_shard_id(shard_id);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_shuffle(shuffle);
    _loader_module->initialize(reader_cfg, DecoderConfig(decoder_type),
                               mem_type,
                               _batch_size, decoder_keep_orig);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> ImageLoaderSingleShardNode::get_loader_module()
{
    if(!_loader_module)
        WRN("ImageLoaderSingleShardNode's loader module is null, not initialized")
    return _loader_module;
}

ImageLoaderSingleShardNode::~ImageLoaderSingleShardNode()
{
    _loader_module = nullptr;
}