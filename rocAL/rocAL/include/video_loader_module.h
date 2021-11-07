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
#include <memory>
#include "video_reader.h"
#include "video_decoder.h"
#include "commons.h"
#include "image.h"
#include "circular_buffer.h"
#include "meta_data_reader.h"
#include "meta_data_graph.h"

enum class VideoLoaderModuleStatus
{
    OK = 0,
    DEVICE_BUFFER_SWAP_FAILED,
    HOST_BUFFER_SWAP_FAILED,
    NO_FILES_TO_READ,
    DECODE_FAILED,
    NO_MORE_DATA_TO_READ,
    NOT_INITIALIZED
};

/*! \class VideoLoaderModule The interface defining the API and requirements of loader modules*/
class VideoLoaderModule
{
public:
    virtual void initialize(VideoReaderConfig reader_config, VideoDecoderConfig decoder_config, RaliMemType mem_type, unsigned batch_size, bool keep_orig_size) = 0;
    virtual void set_output_image(Image *output_image) = 0;
    virtual VideoLoaderModuleStatus load_next() = 0; // Loads the next sequence of frames into the buffer set by calling into the set_output_image
    virtual void reset() = 0;                        // Resets the loader to load from the beginning of the video files
    virtual size_t remaining_count() = 0;            // Returns the number of available frames to be loaded
    virtual ~VideoLoaderModule() = default;
    virtual Timing timing() = 0;                   // Returns timing info
    virtual std::vector<std::string> get_id() = 0; // returns the id of the last batch of images/frames loaded
    virtual void start_loading() = 0;              // starts internal loading thread
    virtual decoded_image_info get_decode_image_info() = 0;
    virtual void set_prefetch_queue_depth(size_t prefetch_queue_depth) = 0;
    virtual std::vector<size_t> get_sequence_start_frame_number() = 0;
    virtual std::vector<std::vector<float>> get_sequence_frame_timestamps() = 0;
};

using pVideoLoaderModule = std::shared_ptr<VideoLoaderModule>;
