#include "node_jpeg_file_source_single_shard.h"
#include "exception.h"


JpegFileSingleShardNode::JpegFileSingleShardNode(Image *output, DeviceResources device_resources):
        Node({}, {output})
{
    _loader_module = std::make_shared<ImageLoader>(device_resources);
}

void
JpegFileSingleShardNode::init(unsigned shard_id, unsigned shard_count, const std::string &source_path, StorageType storage_type, bool loop,
                              size_t load_batch_count, RaliMemType mem_type)
{
    if(!_loader_module)
        THROW("ERROR: loader module is not set for JpegFileNode, cannot initialize")
    if(shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    if(shard_id >= shard_count)
        THROW("Shard is should be smaller than shard count")
    _loader_module->set_output_image(_outputs[0]);
    // Set reader and decoder config accordingly for the JpegFileNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, loop);
    reader_cfg.set_shard_count(shard_count);
    reader_cfg.set_shard_id(shard_id);
    reader_cfg.set_batch_count(load_batch_count);
    _loader_module->initialize(reader_cfg, DecoderConfig(DecoderType::TURBO_JPEG),
                               mem_type,
                               _batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> JpegFileSingleShardNode::get_loader_module()
{
    if(!_loader_module)
        WRN("JpegFileSingleShardNode's loader module is null, not initialized")
    return _loader_module;
}

JpegFileSingleShardNode::~JpegFileSingleShardNode()
{
    _loader_module = nullptr;
}