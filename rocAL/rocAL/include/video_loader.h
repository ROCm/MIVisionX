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
#include "video_read_and_decode.h"

#ifdef RALI_VIDEO

//
// VideoLoader runs an internal thread for loading an decoding of sequences asynchronously
// it uses a circular buffer to store decoded sequence of frames for the user
class VideoLoader : public VideoLoaderModule
{
public:
#if ENABLE_HIP
    explicit VideoLoader(DeviceResourcesHip dev_resources);
#else
    explicit VideoLoader(DeviceResources dev_resources);
#endif
    ~VideoLoader() override;
    VideoLoaderModuleStatus load_next() override;
    void initialize(VideoReaderConfig reader_cfg, VideoDecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size, bool keep_orig_size = false) override;
    void set_output_image(Image *output_image) override;
    size_t remaining_count() override; // returns number of remaining items to be loaded
    void reset() override;             // Resets the loader to load from the beginning
    Timing timing() override;
    void start_loading() override;
    VideoLoaderModuleStatus set_cpu_affinity(cpu_set_t cpu_mask);
    VideoLoaderModuleStatus set_cpu_sched_policy(struct sched_param sched_policy);
    std::vector<std::string> get_id() override;
    decoded_image_info get_decode_image_info() override;
    void set_prefetch_queue_depth(size_t prefetch_queue_depth) override;
    std::vector<size_t> get_sequence_start_frame_number();
    std::vector<std::vector<float>> get_sequence_frame_timestamps();
private:
    bool is_out_of_data();
    void de_init();
    void stop_internal_thread();
    std::shared_ptr<VideoReadAndDecode> _video_loader;
    VideoLoaderModuleStatus update_output_image();
    VideoLoaderModuleStatus load_routine();
    Image *_output_image;
    std::vector<std::string> _output_names; //!< frame name/ids that are stored in the _output_image
    size_t _output_mem_size;
    bool _internal_thread_running;
    size_t _batch_size;
    size_t _sequence_count;
    size_t _sequence_length;
    std::thread _load_thread;
    RaliMemType _mem_type;
    decoded_image_info _decoded_img_info;
    decoded_image_info _output_decoded_img_info;
    CircularBuffer _circ_buff;
    TimingDBG _swap_handle_time;
    bool _is_initialized;
    bool _stopped = false;
    bool _loop;                    //<! If true the reader will wrap around at the end of the media (files/images/...) and wouldn't stop
    size_t _prefetch_queue_depth;  // Used for circular buffer's internal buffer
    size_t _image_counter = 0;     //!< How many frames have been loaded already
    size_t _remaining_sequences_count; //!< How many frames are there yet to be loaded
    bool _decoder_keep_original = false;
    std::vector<std::vector<size_t>> _sequence_start_framenum_vec;
    std::vector<std::vector<std::vector<float>>> _sequence_frame_timestamps_vec;
};
#endif
