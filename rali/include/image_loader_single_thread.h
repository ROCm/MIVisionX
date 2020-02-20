#pragma once 
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "commons.h"
#include "circular_buffer.h"
#include "image_read_and_decode.h"

class ImageLoaderSingleThread : public LoaderModule {
public:
    explicit ImageLoaderSingleThread(DeviceResources dev_resources);
    ~ImageLoaderSingleThread() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size) override;
    void set_output_image (Image* output_image) override;
    size_t count() override; // returns number of remaining items to be loaded
    void reset() override; // Resets the loader to load from the beginning of the media
    std::vector<long long unsigned> timing() override;
    LoaderModuleStatus start_loading();
    LoaderModuleStatus set_cpu_affinity(cpu_set_t cpu_mask);
    LoaderModuleStatus set_cpu_sched_policy(struct sched_param sched_policy);
    void stop() override;
private:
    bool is_out_of_data();
    void de_init();
    std::shared_ptr<ImageReadAndDecode> _image_loader;
    LoaderModuleStatus update_output_image();
    LoaderModuleStatus load_routine();
    Image* _output_image;
    size_t _output_mem_size;
    bool _internal_thread_running;
    size_t _batch_size;
    std::mutex _names_buff_lock;
    std::thread _load_thread;
    RaliMemType _mem_type;
    std::queue<std::vector<std::string>> _circ_buff_names;//!< Stores the loaded images names (data is stored in the _circ_buff)
    std::vector<std::string> _image_names;//!< Name/ID of all the images in tha batch loaded most recently
    CircularBuffer _circ_buff;
    bool _is_initialized;
    bool _ready;
    bool _loop;//<! If true the reader will wrap around at the end of the media (files/images/...) and wouldn't stop
    const static size_t CIRC_BUFFER_DEPTH = 3; // Used for circular buffer's internal buffer
    size_t _image_counter = 0;//!< How many images have been loaded already
    size_t _remaining_image_count;//!< How many images are there yet to be loaded
};

