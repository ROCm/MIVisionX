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


#include "video_loader_module.h"
#ifdef RALI_VIDEO
VideoLoaderModule::VideoLoaderModule(std::shared_ptr<VideoFileNode> video_node):_video_node(std::move(video_node))
{
}

LoaderModuleStatus 
VideoLoaderModule::load_next()
{
    // Do nothing since call to process graph suffices (done externally)
    return LoaderModuleStatus::OK;
}

void
VideoLoaderModule::set_output_image (Image* output_image)
{
}

void
VideoLoaderModule::initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size)
{
}

size_t VideoLoaderModule::count()
{
    // TODO: use FFMPEG to find the total number of frames and keep counting 
    // how many times laod_next() is called successfully, subtract them and 
    // that would be the count of frames remained to be decoded
    return 9999999;
}

void VideoLoaderModule::reset()
{
    // Functionality not there yet in the OpenVX API
}
#endif