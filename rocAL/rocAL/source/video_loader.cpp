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

#include <thread>
#include <chrono>
#include "video_loader.h"
#include "video_read_and_decode.h"
#include "vx_ext_amd.h"

#ifdef RALI_VIDEO
#if ENABLE_HIP
VideoLoader::VideoLoader(DeviceResourcesHip dev_resources):
#else
VideoLoader::VideoLoader(DeviceResources dev_resources):
#endif
_circ_buff(dev_resources),
_swap_handle_time("Swap_handle_time", DBG_TIMING)
{
    _output_image = nullptr;
    _mem_type = RaliMemType::HOST;
    _internal_thread_running = false;
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
    _remaining_sequences_count = 0;
}

VideoLoader::~VideoLoader()
{
    de_init();
}

void VideoLoader::set_prefetch_queue_depth(size_t prefetch_queue_depth)
{
    if (prefetch_queue_depth <= 0)
        THROW("Prefetch quque depth value cannot be zero or negative");
    _prefetch_queue_depth = prefetch_queue_depth;
}

size_t
VideoLoader::remaining_count()
{
    return _remaining_sequences_count;
}

void VideoLoader::reset()
{
    // stop the writer thread and empty the internal circular buffer
    _internal_thread_running = false;
    _circ_buff.unblock_writer();
    if (_load_thread.joinable())
        _load_thread.join();

    // Emptying the internal circular buffer
    _circ_buff.reset();

    // Clearing the frame_num and timestamp vectors
    _sequence_start_framenum_vec.clear();
    _sequence_frame_timestamps_vec.clear();

    // resetting the reader thread to the start of the media
    _image_counter = 0;
    _video_loader->reset();

    // Start loading (writer thread) again
    start_loading();
}

void VideoLoader::de_init()
{
    // Set running to 0 and wait for the internal thread to join
    stop_internal_thread();
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
}

VideoLoaderModuleStatus
VideoLoader::load_next()
{
    return update_output_image();
}

void VideoLoader::set_output_image(Image *output_image)
{
    _output_image = output_image;
    _output_mem_size = _output_image->info().data_size();
}

void VideoLoader::stop_internal_thread()
{
    _internal_thread_running = false;
    _stopped = true;
    _circ_buff.unblock_reader();
    _circ_buff.unblock_writer();
    _circ_buff.reset();
    if (_load_thread.joinable())
        _load_thread.join();
}

void VideoLoader::initialize(VideoReaderConfig reader_cfg, VideoDecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size, bool decoder_keep_original)
{
    if (_is_initialized)
        WRN("initialize() function is already called and loader module is initialized")
    if (_output_mem_size == 0)
        THROW("output image size is 0, set_output_image() should be called before initialize for loader modules")
    _mem_type = mem_type;
    _batch_size = batch_size;
    _loop = reader_cfg.loop();
    _sequence_length = reader_cfg.get_sequence_length();
    _sequence_count = _batch_size / _sequence_length;
    _decoder_keep_original = decoder_keep_original;
    _video_loader = std::make_shared<VideoReadAndDecode>();
    try
    {
        _video_loader->create(reader_cfg, decoder_cfg, _batch_size);
    }
    catch (const std::exception &e)
    {
        de_init();
        throw;
    }
    _decoded_img_info._image_names.resize(_sequence_count);
    _decoded_img_info._roi_height.resize(_batch_size);
    _decoded_img_info._roi_width.resize(_batch_size);
    _decoded_img_info._original_height.resize(_batch_size);
    _decoded_img_info._original_width.resize(_batch_size);
    _circ_buff.init(_mem_type, _output_mem_size, _prefetch_queue_depth);
    _is_initialized = true;
    LOG("Loader module initialized");
}

void VideoLoader::start_loading()
{
    if (!_is_initialized)
        THROW("start_loading() should be called after initialize() function is called")
    _remaining_sequences_count = _video_loader->count();
    _internal_thread_running = true;
    _load_thread = std::thread(&VideoLoader::load_routine, this);
}

VideoLoaderModuleStatus
VideoLoader::load_routine()
{
    LOG("Started the internal loader thread");
    VideoLoaderModuleStatus last_load_status = VideoLoaderModuleStatus::OK;

    // Initially record number of all the frames that are going to be loaded, this is used to know how many still there
    while (_internal_thread_running)
    {
        auto data = _circ_buff.get_write_buffer();
        if (!_internal_thread_running)
            break;

        auto load_status = VideoLoaderModuleStatus::NO_MORE_DATA_TO_READ;
        {
            load_status = _video_loader->load(data,
                                              _decoded_img_info._image_names,
                                              _output_image->info().width(),
                                              _output_image->info().height_single(),
                                              _decoded_img_info._roi_width,
                                              _decoded_img_info._roi_height,
                                              _decoded_img_info._original_width,
                                              _decoded_img_info._original_height,
                                              _sequence_start_framenum_vec,
                                              _sequence_frame_timestamps_vec,
                                              _output_image->info().color_format());

            if (load_status == VideoLoaderModuleStatus::OK)
            {
                _circ_buff.set_image_info(_decoded_img_info);
                _circ_buff.push();
                _image_counter += _output_image->info().batch_size();
            }
        }
        if (load_status != VideoLoaderModuleStatus::OK)
        {
            if (last_load_status != load_status)
            {
                if (load_status == VideoLoaderModuleStatus::NO_MORE_DATA_TO_READ ||
                    load_status == VideoLoaderModuleStatus::NO_FILES_TO_READ)
                {
                    LOG("Cycled through all images, count " + TOSTR(_image_counter));
                }
                else
                {
                    ERR("ERROR: Detected error in reading the images");
                }
                last_load_status = load_status;
            }

            // Here it sets the out-of-data flag and signal the circular buffer's internal
            // read semaphore using release() call
            // , and calls the release() allows the reader thread to wake up and handle
            // the out-of-data case properly
            // It also slows down the reader thread since there is no more data to read,
            // till program ends or till reset is called
            _circ_buff.unblock_reader();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    return VideoLoaderModuleStatus::OK;
}

bool VideoLoader::is_out_of_data()
{
    return (remaining_count() < _sequence_count);
}

VideoLoaderModuleStatus
VideoLoader::update_output_image()
{
    VideoLoaderModuleStatus status = VideoLoaderModuleStatus::OK;
    if (is_out_of_data())
        return VideoLoaderModuleStatus::NO_MORE_DATA_TO_READ;
    if (_stopped)
        return VideoLoaderModuleStatus::OK;

    // _circ_buff.get_read_buffer_x() is blocking and puts the caller on sleep until new images are written to the _circ_buff
    //if (_mem_type == RaliMemType::OCL)
    if((_mem_type== RaliMemType::OCL) || (_mem_type== RaliMemType::HIP))
    {
        auto data_buffer = _circ_buff.get_read_buffer_dev();
        _swap_handle_time.start();
        if (_output_image->swap_handle(data_buffer) != 0)
            return VideoLoaderModuleStatus ::DEVICE_BUFFER_SWAP_FAILED;
        _swap_handle_time.end();
    }
    else
    {
        auto data_buffer = _circ_buff.get_read_buffer_host();
        _swap_handle_time.start();
        if (_output_image->swap_handle(data_buffer) != 0)
            return VideoLoaderModuleStatus::HOST_BUFFER_SWAP_FAILED;
        _swap_handle_time.end();
    }
    if (_stopped)
        return VideoLoaderModuleStatus::OK;
    _output_decoded_img_info = _circ_buff.get_image_info();
    _output_names = _output_decoded_img_info._image_names;
    _output_image->update_image_roi(_output_decoded_img_info._roi_width, _output_decoded_img_info._roi_height);
    _circ_buff.pop();
    if (!_loop)
        _remaining_sequences_count -= _sequence_count;
    return status;
}

Timing VideoLoader::timing()
{
    auto t = _video_loader->timing();
    t.video_process_time = _swap_handle_time.get_timing();
    return t;
}

VideoLoaderModuleStatus VideoLoader::set_cpu_affinity(cpu_set_t cpu_mask)
{
    if (!_internal_thread_running)
        THROW("set_cpu_affinity() should be called after start_loading function is called")
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
    int ret = pthread_setaffinity_np(_load_thread.native_handle(),
                                     sizeof(cpu_set_t), &cpu_mask);
    if (ret != 0)
        WRN("Error calling pthread_setaffinity_np: " + TOSTR(ret));
#endif
    return VideoLoaderModuleStatus::OK;
}

VideoLoaderModuleStatus VideoLoader::set_cpu_sched_policy(struct sched_param sched_policy)
{
    if (!_internal_thread_running)
        THROW("set_cpu_sched_policy() should be called after start_loading function is called")
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
    auto ret = pthread_setschedparam(_load_thread.native_handle(), SCHED_FIFO, &sched_policy);
    if (ret != 0)
        WRN("Unsuccessful in setting thread realtime priority for loader thread err = " + TOSTR(ret))
#endif
    return VideoLoaderModuleStatus::OK;
}

std::vector<std::string> VideoLoader::get_id()
{
    return _output_names;
}

decoded_image_info VideoLoader::get_decode_image_info()
{
    return _output_decoded_img_info;
}

std::vector<size_t> VideoLoader::get_sequence_start_frame_number()
{
    std::vector<size_t> sequence_start_framenum;
    sequence_start_framenum = _sequence_start_framenum_vec.back();
    _sequence_start_framenum_vec.pop_back();
    return sequence_start_framenum;
}

std::vector<std::vector<float>> VideoLoader::get_sequence_frame_timestamps()
{
    std::vector<std::vector<float>> sequence_frame_timestamp;
    sequence_frame_timestamp = _sequence_frame_timestamps_vec.back();
    _sequence_frame_timestamps_vec.pop_back();
    return sequence_frame_timestamp;
}
#endif
