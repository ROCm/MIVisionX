#pragma once
#include <dirent.h>
#include <vector>
#include <memory>
#include "commons.h"
#include "turbo_jpeg_decoder.h"
#include "reader_factory.h"
#include "timing_debug.h"
#include "loader_module.h"

/**
 * Compute the scaled value of <tt>dimension</tt> using the given scaling
 * factor.  This macro performs the integer equivalent of <tt>ceil(dimension *
 * scalingFactor)</tt>.
 */
#define TJSCALED(dimension, scalingFactor) \
  ((dimension * scalingFactor.num + scalingFactor.denom - 1) / \
   scalingFactor.denom)

class ImageReadAndDecode
{
public:
    ImageReadAndDecode();
    ~ImageReadAndDecode();
    size_t count();
    void reset();
    void create(ReaderConfig reader_config, DecoderConfig decoder_config, int batch_size);

    //! Loads a decompressed batch of images into the buffer indicated by buff
    /// \param buff User's buffer provided to be filled with decoded image samples
    /// \param names User's buffer provided to be filled with name of the images decoded
    /// \param max_decoded_width User's buffer maximum width per decoded image. User expects the decoder to downscale the image if image's original width is bigger than max_width
    /// \param max_decoded_height user's buffer maximum height per decoded image. User expects the decoder to downscale the image if image's original height is bigger than max_height
    /// \param actual_decoded_width User's buffer to be filled with actual decoded image width. decoded_width is lower than max_width and is either equal to the original image width if smaller than max_width or downscaled if necessary to fit the max_width criterion.
    /// \param actual_decoded_height User's buffer to be filled with actual decoded image height. decoded_height is lower than max_height and is either equal to the original image height if smaller than max_height or downscaled if necessary to fit the max_height criterion.
    /// \param output_color_format defines what color format user expects decoder to decode images into if capable of doing so supported is
    LoaderModuleStatus load(
            unsigned char* buff,
            std::vector<std::string>& names,
            const size_t  max_decoded_width,
            const size_t max_decoded_height,
            std::vector<uint>& actual_decoded_width,
            std::vector<uint>& actual_decoded_height,
            RaliColorFormat output_color_format );

    //! returns timing info or other status information
    Timing timing();

private:
    std::vector<std::shared_ptr<Decoder>> _decoder;
    std::shared_ptr<Reader> _reader;
    std::vector<std::vector<unsigned char>> _compressed_buff;
    std::vector<size_t> _actual_read_size;
    std::vector<std::string> _image_names;
    std::vector<size_t> _compressed_image_size;
    std::vector<unsigned char*> _decompressed_buff_ptrs;
    std::vector<size_t> _actual_decoded_width;
    std::vector<size_t> _actual_decoded_height;
    static const size_t MAX_COMPRESSED_SIZE = 1*1024*1024; // 1 Meg
    TimingDBG _file_load_time, _decode_time;
    size_t _batch_size;
};

