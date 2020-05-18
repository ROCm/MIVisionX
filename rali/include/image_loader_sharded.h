#pragma once
#include <vector>
#include "image_loader.h"
//
// ImageLoaderSharded Can be used to run load and decode in multiple shards, each shard by a single loader instance,
// It improves load and decode performance since each loader loads the images in parallel using an internal thread
//
class ImageLoaderSharded : public LoaderModule
{
public:
    explicit ImageLoaderSharded(DeviceResources dev_resources);
    ~ImageLoaderSharded() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size, bool keep_orig_size=false) override;
    void set_output_image (Image* output_image) override;
    size_t remaining_count() override;
    void reset() override;
    void start_loading() override;
    std::vector<std::string> get_id() override;
    Timing timing() override;
private:
    void increment_loader_idx();
    const DeviceResources _dev_resources;
    bool _initialized = false;
    std::vector<std::shared_ptr<ImageLoader>> _loaders;
    size_t _loader_idx;
    size_t _shard_count = 1;
    void fast_forward_through_empty_loaders();

    Image *_output_image;
};
