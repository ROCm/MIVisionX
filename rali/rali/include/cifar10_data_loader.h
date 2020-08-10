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
#include "reader_factory.h"
#include "timing_debug.h"
#include "cifar10_data_reader.h"

class CIFAR10DataLoader : public LoaderModule
{
public:
    explicit CIFAR10DataLoader(DeviceResources dev_resources);
    ~CIFAR10DataLoader() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size, bool keep_orig_size=true) override;
    void set_output_image (Image* output_image) override;
    size_t remaining_count() override;
    void reset() override;
    void start_loading() override;
    std::vector<std::string> get_id() override;
    decoded_image_info get_decode_image_info() override;
    Timing timing() override;
private:
    void increment_loader_idx();
    bool is_out_of_data();
    void de_init();
    void stop_internal_thread();
    LoaderModuleStatus update_output_image();
    LoaderModuleStatus load_routine();
    std::shared_ptr<Reader> _reader;

    const DeviceResources _dev_resources;
    decoded_image_info _raw_img_info;       // image info to store the names. In this case the ID of image is stored in _roi_width field 
    decoded_image_info _output_decoded_img_info;
    bool _initialized = false;
    RaliMemType _mem_type;
    size_t _output_mem_size;
    bool _internal_thread_running;
    size_t _batch_size;
    size_t _image_size;
    std::thread _load_thread;
    std::vector<unsigned char *> _load_buff;
    std::vector<size_t> _actual_read_size;
    std::vector<std::string> _output_names;
    CircularBuffer _circ_buff;
    const static size_t CIRC_BUFFER_DEPTH = 3; // Used for circular buffer's internal buffer
    TimingDBG _file_load_time, _swap_handle_time;
    size_t _loader_idx;
    size_t _shard_count = 1;
    void fast_forward_through_empty_loaders();
    bool _is_initialized;
    bool _stopped = false;
    bool _loop;//<! If true the reader will wrap around at the end of the media (files/images/...) and wouldn't stop
    size_t _image_counter = 0;//!< How many images have been loaded already
    size_t _remaining_image_count;//!< How many images are there yet to be loaded
    Image *_output_image;
};