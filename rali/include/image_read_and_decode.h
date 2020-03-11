#pragma once
#include <dirent.h>
#include <vector>
#include <memory>
#include "commons.h"
#include "turbo_jpeg_decoder.h"
#include "reader_factory.h"
#include "timing_debug.h"
#include "loader_module.h"


class ImageReadAndDecode
{
public:
    ImageReadAndDecode();
    ~ImageReadAndDecode();
    size_t count();
    void reset();
    void create(ReaderConfig reader_config, DecoderConfig decoder_config);

    //! Loads a decompressed batch of images into the buffer indicated by buff
    LoaderModuleStatus load(
        unsigned char* buff,
        std::vector<std::string>& names,
        unsigned batch_size,
        unsigned output_width,
        unsigned output_height,
        RaliColorFormat output_color_format );

    //! returns timing info or other status information
    std::vector<long long unsigned> timing();

private:
    //! Decodes an image to the desired width, height and color format.
    /*!
     * Depending on the decoder capability it might resize to the exact 
     * or closest desired size
    */ 
    LoaderModuleStatus decode(
            unsigned char* input_buff,
            size_t size,
            unsigned char *output_buff,
            unsigned int output_width,
            unsigned int output_height,
            Decoder::ColorFormat color_format,
            unsigned int output_planes);

    std::shared_ptr<Decoder> _decoder;
    std::shared_ptr<Reader> _reader;
    std::vector<unsigned char> _compressed_buff;
    static const size_t MAX_COMPRESSED_SIZE = 2*1024*1024; // 2 Meg
    TimingDBG _file_load_time, _decode_time;
};

