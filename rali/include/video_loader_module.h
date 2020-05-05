#pragma once
#include <string>
#include <utility>
#include "commons.h"
#include "loader_module.h"
#include "node_video_file_source.h"

#ifdef RALI_VIDEO
class VideoLoaderModule : public LoaderModule
{
public:
    explicit VideoLoaderModule(std::shared_ptr<VideoFileNode> sharedPtr);

    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size) override;
    void set_output_image (Image* output_image) override;
    size_t count() override; // returns number of remaining items to be loaded
    void reset() override; // Resets the loader to load from the beginning of the media
    std::vector<long long unsigned> timing() override {return {0}; }
    void stop() override  {}
    void get_id() { return 0; }
private:
    std::shared_ptr<VideoFileNode> _video_node;
};
#endif