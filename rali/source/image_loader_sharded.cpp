#include "image_loader_sharded.h"

ImageLoaderSharded::ImageLoaderSharded(DeviceResources dev_resources):
        _dev_resources(dev_resources)
{
    _loader_idx = 0;
}

std::vector<std::string> ImageLoaderSharded::get_id()
{
    if(!_initialized)
        THROW("get_id() should be called after initialize() function");
    return _loaders[_loader_idx]->get_id();
}

ImageLoaderSharded::~ImageLoaderSharded()
{
    _loaders.clear();
}

void
ImageLoaderSharded::fast_forward_through_empty_loaders()
{
    int loaders_count = _loaders.size();
    // reject empty loaders and get to a loader that still has images to play
    while (_loaders[_loader_idx]->remaining_count() == 0 && loaders_count-- > 0)
        increment_loader_idx();
}

LoaderModuleStatus ImageLoaderSharded::load_next()
{
    if(!_initialized)
        return LoaderModuleStatus::NOT_INITIALIZED;

    increment_loader_idx();

    // Since loaders may have different number of images loaded, some run out earlier than other.
    // Fast forward through loaders that are empty to get to a loader that is not empty.
    fast_forward_through_empty_loaders();

    auto ret= _loaders[_loader_idx]->load_next();

    return ret;
}
void
ImageLoaderSharded::initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type,
                               unsigned batch_size)
{
    if(_initialized)
        return;
    _shard_count = reader_cfg.get_shard_count();
    // Create loader modules
    for(size_t i = 0; i < _shard_count; i++)
    {
        auto loader = std::make_shared<ImageLoader>(_dev_resources);
        _loaders.push_back(loader);
    }
    // Initialize loader modules
    for(size_t idx = 0; idx < _shard_count; idx++)
    {
        _loaders[idx]->set_output_image(_output_image);
        reader_cfg.set_shard_count(_shard_count);
        reader_cfg.set_shard_id(idx);
        _loaders[idx]->initialize(reader_cfg, decoder_cfg, mem_type, batch_size);
    }
    _initialized = true;
}
void ImageLoaderSharded::start_loading()
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
void ImageLoaderSharded::set_output_image (Image* output_image)
{
    _output_image = output_image;
}
size_t ImageLoaderSharded::remaining_count()
{
    int sum = 0;
    for(auto& loader: _loaders)
        sum += loader->remaining_count();
    return sum;
}
void ImageLoaderSharded::reset()
{
    for(auto& loader: _loaders)
        loader->reset();
}
void ImageLoaderSharded::increment_loader_idx()
{
    _loader_idx = (_loader_idx + 1)%_shard_count;
}

Timing ImageLoaderSharded::timing()
{
    Timing t;
    long long unsigned  max_decode_time = 0;
    long long unsigned  max_read_time = 0;
    long long unsigned  swap_handle_time = 0;

    // image read and decode runs in parallel using multiple loaders, and the observable latency that the ImageLoaderSharded user
    // is experiences on the load_next() call due to read and decode time is the maximum of all
    for(auto& loader: _loaders)
    {
        auto info = loader->timing();
        max_read_time = (info.image_read_time > max_read_time) ?  info.image_read_time : max_read_time;
        max_decode_time = (info.image_decode_time > max_decode_time) ? info.image_decode_time : max_decode_time;
        swap_handle_time += info.image_process_time;
    }
    t.image_decode_time = max_decode_time;
    t.image_read_time = max_read_time;
    t.image_process_time = swap_handle_time;
    return t;
}
