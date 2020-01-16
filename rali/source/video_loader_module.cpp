#include <sstream>
#include <vx_amd_media.h>
#include <vx_ext_amd.h>
#include "video_loader_module.h"

VideoLoaderModule::VideoLoaderModule(vx_graph ovx_graph):_graph(ovx_graph)
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
#if 0    
    // Currently only files as storage and FFMEG OpenVX decoding is supported
    if (desc->decoder_type() != DecoderType::OVX_FFMPEG)
        return LoaderModuleStatus::UNSUPPORTED_DECODER_TYPE;

    if (desc->storage_type() != StorageType::FILE_SYSTEM)
        return LoaderModuleStatus::UNSUPPORTED_STORAGE_TYPE;
    
    
    std::ostringstream iss;
    auto vid_desc = dynamic_cast<H264FileDecoderConfig*>(desc);

    // The format for the input string to the OVX decoder API is as follows:
    // <video_stream_count>,<path_to_video_0>:0|1,<path_to_video_0>:0|1,<path_to_video_0>:0|1 ...
    // after each path to videos 0 stands for sw decoding, while 1 stands for hw decoding
    _video_stream_count = vid_desc->_batch_size;
    iss << _video_stream_count << ",";

    for(size_t i = 0; i < _video_stream_count; i++)
        iss << (vid_desc->_path_to_videos[i])<< ":" << ((_decode_mode == DecodeMode::USE_HW) ? "1":"0");
    
    
    size_t video_width, video_height;
    // TODO: Find out the video size from the file itself using FFMPEG API

    vx_image interm_image = vxCreateVirtualImage(_graph, video_width , video_height*_video_stream_count, VX_DF_IMAGE_VIRT);
    
    
    unsigned enbale_open_cl = (vid_desc->_mem_type == RaliMemType::OCL) ? 1 : 0;
    
    // TODO: use _loop and enbale_open_cl flag as the last arguments passed to the amdMediaDecoderNode 
    // vx_node ret  = amdMediaDecoderNode(_graph, iss.str().c_str(), interm_image, NULL);
    // vx_status res = VX_SUCCESS;
    // if((res = vxGetStatus((vx_reference)ret)) != VX_SUCCESS)
       // return LoaderModuleStatus::ADDING_OVX_VIDEO_DECODE_FAILED;

    //
#endif     
    return ;
    
}

size_t VideoLoaderModule::count()
{
    // TODO: use FFMPEG to find the total number of frames and keep counting 
    // how many times laod_next() is called successfully, subtract them and 
    // that would be the count of frames remained to be decoded
    return 999999999; 
}

void VideoLoaderModule::reset()
{
    // Functionality not there yet in the OpenVX API
}