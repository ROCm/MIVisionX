#pragma once

#include <cstddef>

enum class DecoderType
{
    TURBO_JPEG = 0,//!< Can only decode
    OVX_FFMPEG,//!< Uses FFMPEG to decode video streams, can decode up to 4 video streams simultaneously
};



class DecoderConfig
{
public:
    explicit DecoderConfig(DecoderType type):_type(type){}
    virtual DecoderType type() {return _type; };
    DecoderType _type = DecoderType::TURBO_JPEG;
};


class Decoder
{
public:

    enum class Status {
        OK = 0,
        HEADER_DECODE_FAILED,
        CONTENT_DECODE_FAILED,
        UNSUPPORTED
    };

    enum class ColorFormat {
        GRAY = 0,
        RGB, 
        BGR
    };
    //! Decodes the header of the Jpeg compressed data and returns basic info about the compressed image
    /*!
     \param input_buffer  User provided buffer containig the encoded image
     \param input_size Size of the compressed data provided in the input_buffer
     \param width pointer to the user's buffer to write the width of the compressed image to 
     \param height pointer to the user's buffer to write the height of the compressed image to 
     \param color_comps pointer to the user's buffer to write the number of color components of the compressed image to 
    */
    virtual Status decode_info(unsigned char* input_buffer,
                                size_t input_size,
                                int* width, 
                                int* height, 
                                int* color_comps) = 0;
    
    // TODO: Extend the decode API if needed, color format and order can be passed to the function
    //! Decodes the actual image data
    /*! 
      \param input_buffer  User provided buffer containig the encoded image
      \param output_buffer User provided buffer used to write the decoded image into
      \param input_size Size of the compressed data provided in the input_buffer
      \param max_decoded_width The maximum width user wants the decoded image to be
      \param max_decoded_height The maximum height user wants the decoded image to be.

    */
    virtual Decoder::Status decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                   size_t max_decoded_width, size_t max_decoded_height,
                                   size_t original_image_width, size_t original_image_height,
                                   size_t &actual_decoded_width, size_t &actual_decoded_height,
                                   Decoder::ColorFormat desired_decoded_color_format) = 0;

    virtual ~Decoder() = default;
};
