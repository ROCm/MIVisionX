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