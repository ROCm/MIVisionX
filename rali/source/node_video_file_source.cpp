#include <memory>
#include <sstream>
#include <vx_amd_media.h>
#include <vx_ext_amd.h>
#include <vx_ext_rpp.h>
#include "node_video_file_source.h"
#include "video_loader_module.h"
#ifdef RALI_VIDEO
#include "video_loader_module.h"
void VideoFileNode::create(std::shared_ptr<Graph> graph)
{
    if(!_graph)
        _graph = graph;

    std::ostringstream iss;

    // The format for the input string to the OVX decoder API is as follows:
    // <video_stream_count>,<path_to_video_0>:0|1,<path_to_video_0>:0|1,<path_to_video_0>:0|1 ...
    // after each path to videos 0 stands for sw decoding, while 1 stands for hw decoding
    _video_stream_count = _batch_size;
    iss << _video_stream_count << ",";

    auto video_path = _source_path+":";
    std::string path;
    size_t new_pos = 0, prev_pos = 0, vid_count = 0;
    _path_to_videos.resize(_batch_size);
    while((new_pos = video_path.find(":", prev_pos)) != std::string::npos)
    {
        path = video_path.substr(prev_pos, new_pos);
        if(vid_count >= _batch_size)
            break;
        prev_pos = new_pos;
        _path_to_videos[vid_count++] = path;
    }
    LOG("Total of "+TOSTR(vid_count)+ " videos going to be loaded , using "+((_decode_mode == DecodeMode::USE_HW) ? " hardware decoder ":" software decoder"))
    for(size_t i = 0; i < _video_stream_count; i++)
        iss << (_path_to_videos[i])<< ":" << ((_decode_mode == DecodeMode::USE_HW) ? "1":"0");

    LOG("Total of "+TOSTR(vid_count)+ " videos going to be loaded " + iss.str())
    _interm_output = std::make_unique<Image>(_outputs[0]->info());
    _interm_output->create(_outputs[0]->context());
    _node = amdMediaDecoderNode(_graph->get(), iss.str().c_str(), _interm_output->handle(), NULL);
    _copy_node = vxExtrppNode_Copy(_graph->get(), _interm_output->handle(), _outputs[0]->handle());
    vx_status res = VX_SUCCESS;
    if((res = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Failed to add video decoder node")
    if((res = vxGetStatus((vx_reference)_copy_node)) != VX_SUCCESS)
        THROW("Failed to add video copy decoder node")
}
void VideoFileNode::init(const std::string &source_path, DecodeMode decoder_mode, bool loop)
{
    _decode_mode = decoder_mode;
    _source_path = source_path;
    _loop = loop;
    //_loader_module->set_output_image(_outputs[0]);
    // Set reader and decoder config accordingly for the JpegFileNode
    //_loader_module->initialize(ReaderConfig(StorageType::FILE_SYSTEM, _source_path, _loop), DecoderConfig(DecoderType::OVX_FFMPEG),
    //                           _mem_type,
    //                          _batch_size);
}

VideoFileNode::VideoFileNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs)
{
    _batch_size = outputs[0]->info().batch_size();
    if(_batch_size > MAXIMUM_VIDEO_CONCURRENT_DECODE)
        THROW("Video batch size " + TOSTR(_batch_size)+" is bigger than " + TOSTR(MAXIMUM_VIDEO_CONCURRENT_DECODE))
}
#endif