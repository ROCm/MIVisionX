#include "image_loader_multi_thread.h"

ImageLoaderMultiThread::ImageLoaderMultiThread(const OCLResources& ocl):
_ocl(ocl)
{
    _loader_idx = 0;
    for(size_t i = 0; i< MIN_NUM_THREADS; i++)
    {
        auto loader = std::make_shared<ImageLoaderSingleThread>(ocl);
        _loaders.push_back(loader);
    }
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
LoaderModuleStatus ImageLoaderMultiThread::create(LoaderModuleConfig* desc)
{
    auto ret = LoaderModuleStatus::OK;

    if(_created)
        return ret;

    for(size_t idx = 0; idx < THREAD_COUNT; idx++)
    {
        _loaders[idx]->set_load_interval(THREAD_COUNT);
        _loaders[idx]->set_load_offset(idx);
        ret = _loaders[idx]->create(desc);
        if(ret != LoaderModuleStatus::OK)
            return ret;
    }
    _created = true;
    return ret;
}
LoaderModuleStatus ImageLoaderMultiThread::set_output_image (Image* output_image)
{
    auto ret = LoaderModuleStatus::OK;
    for(auto& loader: _loaders)
    {
        ret = loader->set_output_image(output_image);
        if(ret != LoaderModuleStatus::OK)
            return ret;
    }
    return ret;
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
        auto loader = std::make_shared<ImageLoaderSingleThread>(_ocl);
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
