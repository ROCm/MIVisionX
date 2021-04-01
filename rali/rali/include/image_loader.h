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

#include <string>
#include <thread>
#include <vector>
#include "commons.h"
#include "circular_buffer.h"
#include "image_read_and_decode.h"
#include "meta_data_reader.h"
//
// ImageLoader runs an internal thread for loading an decoding of images asynchronously
// it uses a circular buffer to store decoded frames and images for the user
class ImageLoader : public LoaderModule {
public:
    explicit ImageLoader(DeviceResources dev_resources);
    ~ImageLoader() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size, bool keep_orig_size=false) override;
    void set_output_image (Image* output_image) override;
    void set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader) override;
    size_t remaining_count() override; // returns number of remaining items to be loaded
    void reset() override; // Resets the loader to load from the beginning of the media
    Timing timing() override;
    void start_loading() override;
    LoaderModuleStatus set_cpu_affinity(cpu_set_t cpu_mask);
    LoaderModuleStatus set_cpu_sched_policy(struct sched_param sched_policy);
    std::vector<std::string> get_id() override;
    decoded_image_info get_decode_image_info() override;
private:
    bool is_out_of_data();
    void de_init();
    void stop_internal_thread();
    std::shared_ptr<ImageReadAndDecode> _image_loader;
    LoaderModuleStatus update_output_image();
    LoaderModuleStatus load_routine();
    Image* _output_image;
    std::vector<std::string> _output_names;//!< image name/ids that are stores in the _output_image
    size_t _output_mem_size;
    MetaDataBatch* _meta_data = nullptr;//!< The output of the meta_data_graph,
    std::vector<std::vector <float>> _bbox_coords;
    bool _internal_thread_running;
    size_t _batch_size;
    std::thread _load_thread;
    RaliMemType _mem_type;
    decoded_image_info _decoded_img_info;
    decoded_image_info _output_decoded_img_info;
    CircularBuffer _circ_buff;
    TimingDBG _swap_handle_time;
    bool _is_initialized;
    bool _stopped = false;
    bool _loop;//<! If true the reader will wrap around at the end of the media (files/images/...) and wouldn't stop
    const static size_t CIRC_BUFFER_DEPTH = 3; // Used for circular buffer's internal buffer
    size_t _image_counter = 0;//!< How many images have been loaded already
    size_t _remaining_image_count;//!< How many images are there yet to be loaded
    bool _decoder_keep_original = false;
    std::shared_ptr<RandomBBoxCrop_MetaDataReader> _randombboxcrop_meta_data_reader = nullptr; 
};

