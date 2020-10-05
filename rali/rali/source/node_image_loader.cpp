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

#include "node_image_loader.h"
#include "exception.h"


ImageLoaderNode::ImageLoaderNode(Image *output, DeviceResources device_resources):
        Node({}, {output})
{
    _loader_module = std::make_shared<ImageLoaderSharded>(device_resources);
}

void ImageLoaderNode::init(unsigned internal_shard_count, const std::string &source_path, const std::string &json_path, const std::map<std::string, std::string> feature_key_map, StorageType storage_type,
                           DecoderType decoder_type, bool shuffle, bool loop, size_t load_batch_count, RaliMemType mem_type, bool decoder_keep_orig, const char* file_prefix)
{
    if(!_loader_module)
        THROW("ERROR: loader module is not set for ImageLoaderNode, cannot initialize")
    if(internal_shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    _loader_module->set_output_image(_outputs[0]);
    // Set reader and decoder config accordingly for the ImageLoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, json_path, feature_key_map, shuffle, loop);
    reader_cfg.set_shard_count(internal_shard_count);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_file_prefix(file_prefix);
    _loader_module->initialize(reader_cfg, DecoderConfig(decoder_type),
             mem_type,
             _batch_size, decoder_keep_orig);
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