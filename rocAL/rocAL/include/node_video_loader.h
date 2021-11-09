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
#include "node.h"
#include "video_loader_sharded.h"
#include "graph.h"
#include <tuple>

#ifdef RALI_VIDEO
class VideoLoaderNode : public Node
{
public:
#if ENABLE_HIP
    VideoLoaderNode(Image *output, DeviceResourcesHip device_resources);
#else
    VideoLoaderNode(Image *output, DeviceResources device_resources);
#endif
    ~VideoLoaderNode() override;
    VideoLoaderNode() = delete;
    ///
    /// \param internal_shard_count Defines the amount of parallelism user wants for the load and decode process to be handled internally.
    /// \param source_path Defines the path that includes the video dataset
    /// \param load_batch_count Defines the quantum count of the sequences to be loaded. It's usually equal to the user's batch size.
    /// The loader will repeat sequences if necessary to be able to have sequences in multiples of the load_batch_count,
    /// for example if there are 10 sequences in the dataset and load_batch_count is 3, the loader repeats 2 sequences as if there are 12 sequences available.
    void init(unsigned internal_shard_count, const std::string &source_path, VideoStorageType storage_type, VideoDecoderType decoder_type, DecodeMode decoder_mode,
              unsigned sequence_length, unsigned step, unsigned stride, VideoProperties &video_prop, bool shuffle, bool loop, size_t load_batch_count, RaliMemType mem_type);
    std::shared_ptr<VideoLoaderModule> get_loader_module();
protected:
    void create_node() override{};
    void update_node() override{};
private:
    DecodeMode _decode_mode = DecodeMode::CPU;
    std::shared_ptr<VideoLoaderSharded> _loader_module = nullptr;
};
#endif
