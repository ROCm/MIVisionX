#pragma once 
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "image_loader_configs.h"
#include "commons.h"
#include "circular_buffer.h"
#include "image_loader_factory.h"

class ImageLoaderSingleThread : public LoaderModule {
public:
    explicit ImageLoaderSingleThread(OCLResources ocl);
    ~ImageLoaderSingleThread() override;
    LoaderModuleStatus load_next() override;
    LoaderModuleStatus create( LoaderModuleConfig* desc) override;
    LoaderModuleStatus set_output_image (Image* output_image) override;
    size_t count() override; // returns number of remaining items to be loaded
    void reset() override; // Resets the loader to load from the beginning of the media
    void set_load_offset(size_t offset);
    void set_load_interval(size_t interval);
    std::vector<long long unsigned> timing() override;
private:
    bool is_out_of_data();
    void de_init();
    std::shared_ptr<ImageLoaderFactory> _image_loader;
    LoaderModuleStatus update_output_image();
    LoaderModuleStatus load_routine();
    void start_loading();
    Image* _output_image;
    int _output_mem_size;
    int _running;
    int _batch_size;
    std::mutex _lock;
    std::thread _load_thread;
    RaliMemType _mem_type;
    std::queue<std::vector<std::string>> _circ_buff_names;//!< Stores the loaded images names (data is stored in the _circ_buff)
    std::vector<std::string> _image_names;
    CircularBuffer _circ_buff;
    bool _is_initialized;
    bool _ready;
    size_t _load_offset = 0;
    size_t _load_interval = 1;
    const static size_t CIRC_BUFFER_DEPTH = 3; // Used for circular buffer's internal buffer
    size_t _image_counter = 0;
};

