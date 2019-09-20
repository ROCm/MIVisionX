/*
 * folderSourceInputModule.h
 *
 *  Created on: Jan 21, 2019
 *      Author: root
 */

#pragma once
#include <dirent.h>
#include <vector>
#include <memory>
#include "commons.h"
#include "turbo_jpeg_decoder.h"
#include "reader_factory.h"
#include "timing_debug.h"
#include "loader_module.h"


class ImageLoaderFactory
{
public:
    ImageLoaderFactory();
    ~ImageLoaderFactory();
    size_t count();
    void reset();
    LoaderModuleStatus create(LoaderModuleConfig* desc, size_t load_interval = 1, size_t load_offset = 0);

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
        int output_width, 
        int output_height, 
        Decoder::ColorFormat color_format, 
        int output_planes);

    std::shared_ptr<Decoder> _decoder;
    std::shared_ptr<Reader> _reader;
    std::vector<unsigned char> _compressed_buff;
    unsigned _compressed_size;
    static const size_t MAX_COMPRESSED_SIZE = 2*1024*1024; // 2 Meg
    TimingDBG _file_load_time, _decode_time;
};

