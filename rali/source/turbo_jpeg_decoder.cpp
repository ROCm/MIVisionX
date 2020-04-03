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
    int jpegSubsamp;
    if(tjDecompressHeader2(m_jpegDecompressor,
                            input_buffer, 
                            input_size, 
                            width, 
                            height, 
                            &jpegSubsamp) != 0) 
    {
        WRN("Jpeg header decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
        return Status::HEADER_DECODE_FAILED;
    }
    return Status::OK;
}

Decoder::Status TJDecoder::decode(unsigned char* input_buffer, 
                                    size_t input_size, 
                                    unsigned char* output_buffer, 
                                    int desired_width, 
                                    int desired_height, 
                                    Decoder::ColorFormat desired_color) 
{
    int tjpf = TJPF_RGB;
    int planes = 1;
    switch (desired_color) {
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
    
    //TODO : Turbo Jpeg supports multiple color packing and color formats, add more as an option to the API TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_RGBA, TJPF_GRAY, TJPF_CMYK , ...
    if( tjDecompress2(m_jpegDecompressor,  
                        input_buffer, 
                        input_size, 
                        output_buffer, 
                        desired_width, 
                        desired_width*planes, 
                        desired_height, 
                        tjpf,
                        TJFLAG_FASTDCT) != 0)
    {
        WRN("Jpeg image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
        return Status::CONTENT_DECODE_FAILED;
    }
    return Status::OK;
}

TJDecoder::~TJDecoder() {
    tjDestroy(m_jpegDecompressor);
}