#pragma once
#include <string>
#include <VX/vx.h>
#include "commons.h"
#include "video_loader_configs.h"


class VideoLoaderModule : public LoaderModule 
{
public:
    VideoLoaderModule(std::shared_ptr<Graph> ovx_graph);
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size) override;
    void set_output_image (Image* output_image) override;
    size_t count() override; // returns number of remaining items to be loaded
    void reset() override; // Resets the loader to load from the beginning of the media
    std::vector<long long unsigned> timing() override {return {0}; }
    void stop() override  {}
private:
    vx_graph _graph;
    DecodeMode _decode_mode  = DecodeMode::USE_SW;
    unsigned _video_stream_count;
    std::vector<std::string> _path_to_videos;
    size_t _video_width, _video_height;
    const static unsigned MAXIMUM_VIDEO_CONCURRENT_DECODE = 4;
    RaliMemType _mem_type;
    Image* _output_image = nullptr;
    unsigned _loop = 0;
};
