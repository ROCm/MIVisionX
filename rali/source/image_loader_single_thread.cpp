#include <thread> 
#include <chrono> 
#include <CL/cl_ext.h>
#include "image_loader_single_thread.h"
#include "image_read_and_decode.h"
#include "vx_ext_amd.h"

ImageLoaderSingleThread::ImageLoaderSingleThread(DeviceResources dev_resources):
_circ_buff(dev_resources, CIRC_BUFFER_DEPTH)
{
    _output_image = nullptr;
    _mem_type = RaliMemType::HOST;
    _internal_thread_running = false;
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
    _ready = false;
    _remaining_image_count = 0;
}

ImageLoaderSingleThread::~ImageLoaderSingleThread()
{
    de_init();
}

size_t
ImageLoaderSingleThread::count()
{
    return _remaining_image_count;
}

void 
ImageLoaderSingleThread::reset()
{
    if(!_ready) return;
    // stop the writer thread and empty the internal circular buffer
    _internal_thread_running = false;
    _circ_buff.unblock_writer();

    if(_load_thread.joinable())
        _load_thread.join();

    // Emptying the internal circular buffer
    _circ_buff.reset();
    while(!_circ_buff_names.empty())
        _circ_buff_names.pop();

    // resetting the reader thread to the start of the media
    _image_counter = 0;
    _image_loader->reset();

    // Start loading (writer thread) again
    start_loading();
}

void 
ImageLoaderSingleThread::de_init()
{
    if(!_ready) return;
    reset();
    // Set running to 0 and wait for the internal thread to join
    stop();
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
    _ready = false;
}

LoaderModuleStatus
ImageLoaderSingleThread::load_next()
{
    if(!_ready)
        return LoaderModuleStatus::NOT_INITIALIZED;

    return update_output_image();
}

void
ImageLoaderSingleThread::set_output_image (Image* output_image)
{
    _output_image = output_image;
    _output_mem_size = _output_image->info().data_size();
}

void
ImageLoaderSingleThread::stop()
{
    _internal_thread_running = false;
    _circ_buff.unblock_reader();
    _circ_buff.unblock_writer();
    _circ_buff.reset();
    if(_load_thread.joinable())
        _load_thread.join();
}

void
ImageLoaderSingleThread::initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size)
{
    if(_is_initialized)
        WRN("Create function is already called and loader module is initialized")

    if(_output_mem_size == 0)
        THROW("output image size is 0, set_output_image() should be called before initialize for loader modules")

    _mem_type = mem_type;
    _batch_size = batch_size;
    _loop = reader_cfg.loop();
    _image_loader = std::make_shared<ImageReadAndDecode>();
    try
    {
        _image_loader->create(reader_cfg, decoder_cfg);
    }
    catch(const std::exception& e)
    {
        de_init();
        throw;
    }
    _image_names.resize(_batch_size);
    _circ_buff.init(_mem_type, _output_mem_size);
    _is_initialized = true;
    LOG("Loader module initialized");
}

LoaderModuleStatus
ImageLoaderSingleThread::start_loading()
{
    if(!_is_initialized)
        THROW("start_loading() should be called after create function is called")

    _ready = true;
    _internal_thread_running = true;
    _load_thread = std::thread(&ImageLoaderSingleThread::load_routine, this);
    return LoaderModuleStatus::OK;
}


LoaderModuleStatus 
ImageLoaderSingleThread::load_routine()
{
    LOG("Started the internal loader thread");
    LoaderModuleStatus last_load_status = LoaderModuleStatus::OK;
    // Initially record number of all the images that are going to be loaded, this is used to know how many still there
    _remaining_image_count = _image_loader->count();
    while(_internal_thread_running)
    {

        auto data = _circ_buff.get_write_buffer();
        if(!_internal_thread_running)
            break;

        auto load_status = LoaderModuleStatus::NO_MORE_DATA_TO_READ;
        {
            load_status = _image_loader->load(data,
                                              _image_names,
                                              _output_image->info().batch_size(),
                                              _output_image->info().width(),
                                              _output_image->info().height_batch(),
                                              _output_image->info().color_format() );

            if(load_status == LoaderModuleStatus::OK)
            {
                std::unique_lock<std::mutex> lock(_names_buff_lock);
                // Pushing to the _circ_buff and _circ_buff_names must happen all at the same time
                _circ_buff.push();
                 _image_counter += _output_image->info().batch_size();
                _circ_buff_names.push(_image_names);
            }

        }
        if(load_status != LoaderModuleStatus::OK)
        {
            if(last_load_status != load_status )
            {
                if (load_status == LoaderModuleStatus::NO_MORE_DATA_TO_READ ||
                    load_status == LoaderModuleStatus::NO_FILES_TO_READ)
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
    return LoaderModuleStatus::OK;
}

bool 
ImageLoaderSingleThread::is_out_of_data()
{
    return (count() == 0) ;
}
LoaderModuleStatus 
ImageLoaderSingleThread::update_output_image()
{
    LoaderModuleStatus status = LoaderModuleStatus::OK;

    if(is_out_of_data())
        return LoaderModuleStatus::NO_MORE_DATA_TO_READ;

    // _circ_buff.get_read_buffer_x() is blocking and puts the caller on sleep until new images are written to the _circ_buff
    if(_mem_type== RaliMemType::OCL)
    {
        auto data_buffer =  _circ_buff.get_read_buffer_dev();
        if(_output_image->swap_handle(data_buffer)!= 0)
            return LoaderModuleStatus ::DEVICE_BUFFER_SWAP_FAILED;
    } 
    else 
    {
        auto data_buffer = _circ_buff.get_read_buffer_host();
        if(_output_image->swap_handle(data_buffer) != 0)
            return LoaderModuleStatus::HOST_BUFFER_SWAP_FAILED;
    }
    {
        // Reason for the mutex here: Pop from _circ_buff and _circ_buff_names happens at the same time so that
        // image and it's id stay matched
        std::unique_lock<std::mutex> lock(_names_buff_lock);
        _output_image->set_names(_circ_buff_names.front());
        _circ_buff_names.pop();
        _circ_buff.pop();
        if(!_loop)
            _remaining_image_count -= _batch_size;
    }
    return status;
}

std::vector<long long unsigned> ImageLoaderSingleThread::timing()
{
    return _image_loader->timing();
}

LoaderModuleStatus ImageLoaderSingleThread::set_cpu_affinity(cpu_set_t cpu_mask)
{
    if(!_internal_thread_running)
        THROW("set_cpu_affinity() should be called after start_loading function is called")
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
    int ret = pthread_setaffinity_np(_load_thread.native_handle(),
                                    sizeof(cpu_set_t), &cpu_mask);
    if (ret != 0)
        WRN("Error calling pthread_setaffinity_np: " + TOSTR(ret));
#endif
    return LoaderModuleStatus::OK;
}

LoaderModuleStatus ImageLoaderSingleThread::set_cpu_sched_policy(struct sched_param sched_policy)
{
    if(!_internal_thread_running)
        THROW("set_cpu_sched_policy() should be called after start_loading function is called")
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
    auto ret = pthread_setschedparam(_load_thread.native_handle(), SCHED_FIFO, &sched_policy);
    if (ret != 0)
        WRN("Unsuccessful in setting thread realtime priority for loader thread err = "+TOSTR(ret))
#endif
    return LoaderModuleStatus::OK;
}
