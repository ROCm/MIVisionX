#pragma once
#include <vector>
#include "loader_module.h"

enum class DecodeMode {
    USE_HW = 0,
    USE_SW = 1
};

/*! \brief Defines the description for FFMPEG enabled H264 video decoing  
 *
 * 
*/
class H264FileDecoderConfig : public LoaderModuleConfig {
public:	
    /*!  
    * 
    * @param path_to_videos Path to maximum of 4 videos, these videos will be stacked on top of each other in the output image
    * @param decode_type Video decoding mode, can be either software or hardware, 
    */
   const static unsigned MAX_DECODING_STREAM_COUNT = 4;//!< This is defined by the hardware capability and the functionality exposed throuh OpenVX
    H264FileDecoderConfig(std::vector<std::string> path_to_videos, RaliMemType mem_type, DecodeMode decode_type, bool loop):
    LoaderModuleConfig(1, mem_type),
    _decode_mode(decode_type),
    _loop(loop) 
    {
        // Limit the number of videos to be decoded to MAX_DECODING_STREAM_COUNT,the rest is ignored
        _batch_size = (_path_to_videos.size() > MAX_DECODING_STREAM_COUNT) ? MAX_DECODING_STREAM_COUNT :
                                                                             path_to_videos.size();

        for(int i = 0; i < _batch_size; i++)
            _path_to_videos.push_back(path_to_videos[i]);
    }
    StorageType storage_type() override { return StorageType::FILE_SYSTEM;}
    DecoderType decoder_type() override { return DecoderType::OVX_FFMPEG;}
    std::vector<std::string> _path_to_videos;
    DecodeMode _decode_mode;
    bool _loop;
};