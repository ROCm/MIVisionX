#pragma once
#include <vector>
#include "image_loader_single_thread.h"

class ImageLoaderMultiThread : public LoaderModule
{
public:
    explicit ImageLoaderMultiThread(const OCLResources& ocl);
    ~ImageLoaderMultiThread() override;
    LoaderModuleStatus load_next() override;
    LoaderModuleStatus create( LoaderModuleConfig* desc) override;
    LoaderModuleStatus set_output_image (Image* output_image) override;
    size_t count() override;
    void reset() override;
    /*!
     *  This function is only effective if called before the create function is called
     */
    void set_thread_count(size_t num_threads);
    std::vector<long long unsigned> timing() override;
private:
    void increment_loader_idx();
    bool _created = false;
    std::vector<std::shared_ptr<ImageLoaderSingleThread>> _loaders;
    size_t _loader_idx;
    constexpr static size_t MIN_NUM_THREADS = 1;
    size_t THREAD_COUNT = MIN_NUM_THREADS;
    const OCLResources _ocl;
};