#include <sstream>
#include <vx_amd_media.h>
#include <vx_ext_amd.h>
#include <graph.h>
#include "video_loader_module.h"

VideoLoaderModule::VideoLoaderModule(std::shared_ptr<Graph> ovx_graph): _graph(ovx_graph->get())
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
    _output_image = output_image;
}

void
VideoLoaderModule::initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RaliMemType mem_type, unsigned batch_size)
{
    if(batch_size > MAXIMUM_VIDEO_CONCURRENT_DECODE)
        THROW("Video batch size " + TOSTR(batch_size)+" is bigger than " + TOSTR(MAXIMUM_VIDEO_CONCURRENT_DECODE))

    // Currently only files as storage and FFMEG OpenVX decoding is supported
    if (decoder_cfg.type() != DecoderType::OVX_FFMPEG)
        THROW("Unsupported video decoder type " + TOSTR(decoder_cfg.type()));

    if (reader_cfg.type() != StorageType::FILE_SYSTEM)
        THROW("Unsupported video storage type " + TOSTR(DecoderType::OVX_FFMPEG));

    _mem_type = mem_type;

    std::ostringstream iss;

    // The format for the input string to the OVX decoder API is as follows:
    // <video_stream_count>,<path_to_video_0>:0|1,<path_to_video_0>:0|1,<path_to_video_0>:0|1 ...
    // after each path to videos 0 stands for sw decoding, while 1 stands for hw decoding
    _video_stream_count = batch_size;
    iss << _video_stream_count << ",";

    auto video_path = reader_cfg.path();
    std::string path;
    size_t new_pos = 0, prev_pos = 0, vid_count = 0;
    _path_to_videos.resize(batch_size);
    while((new_pos = video_path.find(":", prev_pos)) != std::string::npos)
    {
        path = video_path.substr(prev_pos, new_pos);
        if(vid_count >= batch_size)
            break;
        prev_pos = new_pos;
        _path_to_videos[vid_count++] = path;
    }
    LOG("Total of "+TOSTR(vid_count)+ " videos going to be loaded ")
    for(size_t i = 0; i < _video_stream_count; i++)
        iss << (_path_to_videos[i])<< ":" << ((_decode_mode == DecodeMode::USE_HW) ? "1":"0");

    LOG("Total of "+TOSTR(vid_count)+ " videos going to be loaded " + iss.str())
    
    // TODO: use _loop and enbale_open_cl flag as the last arguments passed to the amdMediaDecoderNode 
    vx_node ret  = amdMediaDecoderNode(_graph, iss.str().c_str(), _output_image->handle(), NULL);
    vx_status res = VX_SUCCESS;
    if((res = vxGetStatus((vx_reference)ret)) != VX_SUCCESS)
        THROW("Failed to add video decoder node")

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