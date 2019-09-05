#include <thread> 
#include <chrono> 
#include <CL/cl_ext.h>
#include "image_loader_single_thread.h"
#include "image_loader_factory.h"
#include "vx_ext_amd.h"

ImageLoaderSingleThread::ImageLoaderSingleThread(OCLResources ocl):
_circ_buff(ocl, CIRC_BUFFER_DEPTH)
{
    _running = 0;
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
    _ready = false; 
}

void ImageLoaderSingleThread::set_load_offset(size_t offset)
{
    _load_offset = offset;
}
void ImageLoaderSingleThread::set_load_interval(size_t interval)
{ 
    _load_interval = interval;  
}
ImageLoaderSingleThread::~ImageLoaderSingleThread()
{
    de_init();
}

size_t
ImageLoaderSingleThread::count()
{
    // see load_routine() function for details on the need for the mutex used here
    std::unique_lock<std::mutex> lock(_lock);
    return _image_loader->count() + _circ_buff.level();
}

void 
ImageLoaderSingleThread::reset()
{
    if(!_ready) return;
    _image_counter = 0;
    _image_loader->reset();
}

void 
ImageLoaderSingleThread::de_init()
{
    if(!_ready) return;
    reset();
    _running = 0;
    _circ_buff.cancel_reading();
    _circ_buff.cancel_writing();
    _load_thread.join();
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

    return swap_buffers();
}

LoaderModuleStatus 
ImageLoaderSingleThread::set_output_image (Image* output_image)
{
    if(!_is_initialized)
        THROW("set_output_image should be called after create function is called")

    _output_image = output_image;
    _output_mem_size = _output_image->info().data_size;
    if(_circ_buff.init(_mem_type, _output_mem_size) != CIRCULAR_BUFFER_STATUS::OK)
        return LoaderModuleStatus::INTERNAL_BUFFER_INITIALIZATION_FAILED;

    _ready = true;
    start_loading();
    
    return LoaderModuleStatus::OK;

}

LoaderModuleStatus 
ImageLoaderSingleThread::create(LoaderModuleConfig* desc)
{
    if(_is_initialized)
        WRN("Create function is already called and loader module is initialized")

    _mem_type = desc->_mem_type;
    _batch_size = desc->_batch_size;

    LoaderModuleStatus status = LoaderModuleStatus::OK;
    _image_loader = std::make_shared<ImageLoaderFactory>();
    if((status= _image_loader->create(desc, _load_offset,  _load_interval)) != LoaderModuleStatus::OK) 
    {
        de_init();
        THROW("ERROR, couldn't initialize the loader module");
    }
    
    _is_initialized = true;
    LOG("Loader module initialized");
    return status;
}

void 
ImageLoaderSingleThread::start_loading()
{
     _running = 1;
     _load_thread = std::thread(&ImageLoaderSingleThread::load_routine, this);
}


LoaderModuleStatus 
ImageLoaderSingleThread::load_routine()
{
    LOG("Started the internal loader thread");
    LoaderModuleStatus last_load_status = LoaderModuleStatus::OK;
    while(_running)
    {
        auto data = _circ_buff.get_write_buffer();
        if(!_running)
            break;

        auto load_status = LoaderModuleStatus::NO_MORE_DATA_TO_READ;
        {   // load from image loader and calling done_writing() on the circular buffer
            // should be atomic with respect to call to the count() function, since
            // count function return the summation of the level of the circular buffer
            // and the image loader.
            std::unique_lock<std::mutex> lock(_lock);
            load_status = _image_loader->load(data,
                                               _output_image->info().batch_size,
                                               _output_image->info().width(),
                                               _output_image->info().height_batch(),
                                               _output_image->info().color_fmt );

            if(load_status == LoaderModuleStatus::OK)
            {
                _circ_buff.done_writing();
                _image_counter += _output_image->info().batch_size;
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
            _circ_buff.cancel_reading();
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
ImageLoaderSingleThread::swap_buffers()
{
    LoaderModuleStatus status = LoaderModuleStatus::OK;
    vx_status vxstatus;

    if(_mem_type== RaliMemType::OCL) 
    {
        void*  ptr_in[] = { _circ_buff.get_read_buffer_dev()};

        if(is_out_of_data())
            return LoaderModuleStatus::NO_MORE_DATA_TO_READ;

        if((vxstatus= vxSwapImageHandle(_output_image->img,ptr_in,nullptr, 1)) != VX_SUCCESS)
        {
            ERR("Swap handles failed "+TOSTR(vxstatus));
            status= LoaderModuleStatus::OCL_BUFFER_SWAP_FAILED;
        }

        // Updating the buffer pointer as well,
        // user might want to copy directly using it
        _output_image->buf = _circ_buff.get_read_buffer_dev();
    } 
    else 
    {
        void*  ptr_in[] = {_circ_buff.get_read_buffer_host()}; 

        if(is_out_of_data())
            return LoaderModuleStatus::NO_MORE_DATA_TO_READ;

        if((vxstatus= vxSwapImageHandle(_output_image->img,ptr_in,nullptr, 1)) != VX_SUCCESS)
        {
            ERR("Swap handles failed "+TOSTR(vxstatus));
            status= LoaderModuleStatus::HOST_BUFFER_SWAP_FAILED;
        }

        // Updating the buffer pointer as well,
        // user might want to copy directly using it
        _output_image->buf = _circ_buff.get_read_buffer_host();
    }

    _circ_buff.done_reading();

    return status;
}

std::vector<long long unsigned> ImageLoaderSingleThread::timing()
{
    return _image_loader->timing();
}