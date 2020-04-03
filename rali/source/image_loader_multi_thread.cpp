#include "image_loader_multi_thread.h"

ImageLoaderMultiThread::ImageLoaderMultiThread(DeviceResources dev_resources):
        _dev_resources(dev_resources)
{
    _loader_idx = 0;
    for(size_t i = 0; i< MIN_NUM_THREADS; i++)
    {
        auto loader = std::make_shared<ImageLoaderSingleThread>(_dev_resources);
        _loaders.push_back(loader);
    }
}

void ImageLoaderMultiThread::stop()
{
    for(auto& loader: _loaders)
        loader->stop();
}

ImageLoaderMultiThread::~ImageLoaderMultiThread()
{
    _loaders.clear();
}
LoaderModuleStatus ImageLoaderMultiThread::load_next()
{
    auto ret = _loaders[_loader_idx]->load_next();
    increment_loader_idx();
    return ret;
}
void
ImageLoaderMultiThread::initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type,
                                   unsigned batch_size)
{
    if(_created)
        return;

    for(size_t idx = 0; idx < THREAD_COUNT; idx++)
    {
        reader_cfg.set_load_interval(THREAD_COUNT);
        reader_cfg.set_load_offset(idx);
        _loaders[idx]->initialize(reader_cfg, decoder_cfg, mem_type, batch_size);
    }
    _created = true;
}
void ImageLoaderMultiThread::start_loading()
{
    for(unsigned i = 0; i < _loaders.size(); i++)
    {
        _loaders[i]->start_loading();
    //  Changing thread scheduling policy and it's priority does not help on latest Ubuntu builds
    //  and needs tweaking the Linux security settings , can be turned on for experimentation
#if 0
        // Set thread scheduling policy
        struct sched_param params;
        params.sched_priority = sched_get_priority_max(SCHED_FIFO);
        _loaders[i]->set_cpu_sched_policy(params);
#endif
        // Setting cpu affinity for threads works and can be activated below for experimentation
#if 0
        // Set thread affinity thread 0 to core 0 , 1 toc core 1 , ...
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        _loaders[i]->set_cpu_affinity(cpuset);
#endif
    }

}
void ImageLoaderMultiThread::set_output_image (Image* output_image)
{
    for(auto& loader: _loaders)
        loader->set_output_image(output_image);
}
size_t ImageLoaderMultiThread::count()
{
    int sum = 0;
    for(auto& loader: _loaders)
        sum += loader->count();
    return sum;
}
void ImageLoaderMultiThread::reset()
{
    for(auto& loader: _loaders)
        loader->reset();
}
void ImageLoaderMultiThread::increment_loader_idx()
{
    _loader_idx = (_loader_idx + 1)%THREAD_COUNT;
}

void ImageLoaderMultiThread::set_thread_count(size_t num_threads)
{
    // If the threads have been created already, thread count cannot be updated
    if(_created)
        return;

    // The new thread count cannot be less than previously requested
    if(num_threads < THREAD_COUNT)
        return;

    // Add new loader modules to increase their count to num_threads
    for(size_t i = THREAD_COUNT; i< num_threads; i++)
    {
        auto loader = std::make_shared<ImageLoaderSingleThread>(_dev_resources);
        _loaders.push_back(loader);
    }
    THREAD_COUNT = num_threads;
}

std::vector<long long unsigned> ImageLoaderMultiThread::timing()
{
    std::vector<long long unsigned > ret(2,0);

    for(auto& loader: _loaders)
    {
        auto info = loader->timing();
        ret[0] = info[0] > ret[0] ? info[0] : ret[0];
        ret[1] = info[1] > ret[1] ? info[1] : ret[1];
    }
    return ret;
}
