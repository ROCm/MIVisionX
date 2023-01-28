/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "node_video_loader_single_shard.h"
#include "exception.h"
#ifdef ROCAL_VIDEO

VideoLoaderSingleShardNode::VideoLoaderSingleShardNode(Image *output, void *device_resources):
	Node({}, {output})
{
    _loader_module = std::make_shared<VideoLoader>(device_resources);
}

void VideoLoaderSingleShardNode::init(unsigned shard_id, unsigned shard_count, const std::string &source_path, VideoStorageType storage_type, VideoDecoderType decoder_type, DecodeMode decoder_mode,
                                      unsigned sequence_length, unsigned step, unsigned stride, VideoProperties &video_prop, bool shuffle, bool loop, size_t load_batch_count, RocalMemType mem_type)
{
    _decode_mode = decoder_mode; // for future use
    if (!_loader_module)
        THROW("ERROR: loader module is not set for VideoLoaderNode, cannot initialize")
    if (shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    if (shard_id >= shard_count)
        THROW("Shard is should be smaller than shard count")
    _loader_module->set_output_image(_outputs[0]);
    // Set reader and decoder config accordingly for the ImageLoaderNode
    auto reader_cfg = VideoReaderConfig(storage_type, source_path, shuffle, loop);
    reader_cfg.set_shard_count(shard_count);
    reader_cfg.set_shard_id(shard_id);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_sequence_length(sequence_length);
    reader_cfg.set_frame_step(step);
    reader_cfg.set_frame_stride(stride);
    reader_cfg.set_video_properties(video_prop);
    _loader_module->initialize(reader_cfg, VideoDecoderConfig(decoder_type), mem_type, _batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<VideoLoaderModule> VideoLoaderSingleShardNode::get_loader_module()
{
    if (!_loader_module)
        WRN("VideoLoaderSingleShardNode's loader module is null, not initialized")
    return _loader_module;
}

VideoLoaderSingleShardNode::~VideoLoaderSingleShardNode()
{
    _loader_module = nullptr;
}
#endif
