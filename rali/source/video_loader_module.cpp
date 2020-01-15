
#include "video_loader_module.h"

VideoLoaderModule::VideoLoaderModule(std::shared_ptr<VideoFileNode> video_node):_video_node(std::move(video_node))
{
}

LoaderModuleStatus 
VideoLoaderModule::load_next()
{
    // Do nothing since call to process graph suffices (done externally)
    return LoaderModuleStatus::OK;
}

void
VideoLoaderModule::set_output_image (Image* output_image)
{
}

void
VideoLoaderModule::initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size)
{
}

size_t VideoLoaderModule::count()
{
    // TODO: use FFMPEG to find the total number of frames and keep counting 
    // how many times laod_next() is called successfully, subtract them and 
    // that would be the count of frames remained to be decoded
    return 9999999;
}

void VideoLoaderModule::reset()
{
    // Functionality not there yet in the OpenVX API
}