#include <stdio.h>
#include <commons.h>
#include "turbo_jpeg_decoder.h"

TJDecoder::TJDecoder(){
    m_jpegDecompressor = tjInitDecompress();

#if 0
    int num_avail_scalings = 0;
    auto scaling_factors = tjGetScalingFactors	(&num_avail_scalings);	
    for(int i = 0; i < num_avail_scalings; i++) {
        if(scaling_factors[i].num < scaling_factors[i].denom) {

            printf("%d / %d  - ",scaling_factors[i].num, scaling_factors[i].denom );
        }
    }
#endif
};


Decoder::Status TJDecoder::decode_info(unsigned char* input_buffer, size_t input_size, int* width, int* height, int* color_comps) 
{
    //TODO : Use the most recent TurboJpeg API tjDecompressHeader3 which returns the color components
    if(tjDecompressHeader2(m_jpegDecompressor,
                            input_buffer, 
                            input_size, 
                            width, 
                            height, 
                            color_comps) != 0)
    {
        WRN("Jpeg header decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
        return Status::HEADER_DECODE_FAILED;
    }
    return Status::OK;
}

Decoder::Status TJDecoder::decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                  size_t max_decoded_width, size_t max_decoded_height,
                                  size_t original_image_width, size_t original_image_height,
                                  size_t &actual_decoded_width, size_t &actual_decoded_height,
                                  Decoder::ColorFormat desired_decoded_color_format, bool keep_original_size)
{
    int tjpf = TJPF_RGB;
    int planes = 1;
    switch (desired_decoded_color_format) {
        case Decoder::ColorFormat::GRAY:
            tjpf = TJPF_GRAY;
            planes= 1;
        break;
        case Decoder::ColorFormat::RGB:
            tjpf = TJPF_RGB;
            planes = 3;
        break;
        case Decoder::ColorFormat::BGR:
            tjpf = TJPF_BGR;
            planes = 3;
        break;
    };
    if (!keep_original_size) {
        actual_decoded_width = max_decoded_width;
        actual_decoded_height = max_decoded_height;

        //TODO : Turbo Jpeg supports multiple color packing and color formats, add more as an option to the API TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_RGBA, TJPF_GRAY, TJPF_CMYK , ...
        if (tjDecompress2(m_jpegDecompressor,
                          input_buffer,
                          input_size,
                          output_buffer,
                          max_decoded_width,
                          max_decoded_width * planes,
                          max_decoded_height,
                          tjpf,
                          TJFLAG_FASTDCT) != 0) {
            WRN("Jpeg image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
            return Status::CONTENT_DECODE_FAILED;
        }
        // Find the decoded image size using the predefined scaling factors in the turbo jpeg decoder
        uint scaledw = max_decoded_width, scaledh = max_decoded_height;
        for (auto scaling_factor : SCALING_FACTORS) {
            scaledw = TJSCALED(original_image_width, scaling_factor);
            scaledh = TJSCALED(original_image_height, scaling_factor);
            if (scaledw <= max_decoded_width && scaledh <= max_decoded_height)
                break;
        }
        actual_decoded_width = scaledw;
        actual_decoded_height = scaledh;
    } else {
        if (original_image_width < max_decoded_width)
            actual_decoded_width = original_image_width;
        else
            actual_decoded_width = max_decoded_width;
        if (original_image_height < max_decoded_height)
            actual_decoded_height = original_image_height;
        else
            actual_decoded_height = max_decoded_height;

        //TODO : Turbo Jpeg supports multiple color packing and color formats, add more as an option to the API TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_RGBA, TJPF_GRAY, TJPF_CMYK , ...
        if (tjDecompress2(m_jpegDecompressor,
                          input_buffer,
                          input_size,
                          output_buffer,
                          actual_decoded_width,
                          max_decoded_width * planes,
                          actual_decoded_height,
                          tjpf,
                          TJFLAG_FASTDCT) != 0) {
            WRN("Jpeg image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
            return Status::CONTENT_DECODE_FAILED;
        }
        // Find the decoded image size using the predefined scaling factors in the turbo jpeg decoder
        if ((actual_decoded_width != original_image_width) || (actual_decoded_height != original_image_height))
        {
            uint scaledw = actual_decoded_width, scaledh = actual_decoded_height;
            for (auto scaling_factor : SCALING_FACTORS) {
                scaledw = TJSCALED(original_image_width, scaling_factor);
                scaledh = TJSCALED(original_image_height, scaling_factor);
                if (scaledw <= max_decoded_width && scaledh <= max_decoded_height)
                    break;
            }
            actual_decoded_width = scaledw;
            actual_decoded_height = scaledh;
        }
    }
    return Status::OK;
}

TJDecoder::~TJDecoder() {
    tjDestroy(m_jpegDecompressor);
}