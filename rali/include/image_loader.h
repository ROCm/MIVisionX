#pragma once 

#include <string>
#include <thread>
#include <vector>
#include "commons.h"
#include "circular_buffer.h"
#include "image_read_and_decode.h"
//
// ImageLoader runs an internal thread for loading an decoding of images asynchronously
// it uses a circular buffer to store decoded frames and images for the user
class ImageLoader : public LoaderModule {
public:
    explicit ImageLoader(DeviceResources dev_resources);
    ~ImageLoader() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size) override;
    void set_output_image (Image* output_image) override;
    size_t remaining_count() override; // returns number of remaining items to be loaded
    void reset() override; // Resets the loader to load from the beginning of the media
    Timing timing() override;
    void start_loading() override;
    LoaderModuleStatus set_cpu_affinity(cpu_set_t cpu_mask);
    LoaderModuleStatus set_cpu_sched_policy(struct sched_param sched_policy);
    std::vector<std::string> get_id() override;
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
    bool _internal_thread_running;
    size_t _batch_size;
    std::thread _load_thread;
    RaliMemType _mem_type;
    decoded_image_info _decoded_img_info;
    CircularBuffer _circ_buff;
    TimingDBG _swap_handle_time;
    bool _is_initialized;
    bool _stopped = false;
    bool _loop;//<! If true the reader will wrap around at the end of the media (files/images/...) and wouldn't stop
    const static size_t CIRC_BUFFER_DEPTH = 3; // Used for circular buffer's internal buffer
    size_t _image_counter = 0;//!< How many images have been loaded already
    size_t _remaining_image_count;//!< How many images are there yet to be loaded
};

