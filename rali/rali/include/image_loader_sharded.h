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
#include <vector>
#include "image_loader.h"
//
// ImageLoaderSharded Can be used to run load and decode in multiple shards, each shard by a single loader instance,
// It improves load and decode performance since each loader loads the images in parallel using an internal thread
//
class ImageLoaderSharded : public LoaderModule
{
public:
    explicit ImageLoaderSharded(DeviceResources dev_resources);
    ~ImageLoaderSharded() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size, bool keep_orig_size=false) override;
    void set_output_image (Image* output_image) override;
    size_t remaining_count() override;
    void reset() override;
    void start_loading() override;
    std::vector<std::string> get_id() override;
    Timing timing() override;
private:
    void increment_loader_idx();
    const DeviceResources _dev_resources;
    bool _initialized = false;
    std::vector<std::shared_ptr<ImageLoader>> _loaders;
    size_t _loader_idx;
    size_t _shard_count = 1;
    void fast_forward_through_empty_loaders();

    Image *_output_image;
};