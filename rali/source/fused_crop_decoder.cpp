#include <stdio.h>
#include <commons.h>
#include <string.h>
#include "fused_crop_decoder.h"

FusedCropTJDecoder::FusedCropTJDecoder(){
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

Decoder::Status FusedCropTJDecoder::decode_info(unsigned char* input_buffer, size_t input_size, int* width, int* height, int* color_comps) 
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

Decoder::Status FusedCropTJDecoder::decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                  size_t max_decoded_width, size_t max_decoded_height,
                                  size_t original_image_width, size_t original_image_height,
                                  size_t &actual_decoded_width, size_t &actual_decoded_height,
                                  Decoder::ColorFormat desired_decoded_color_format, DecoderConfig decoder_config, bool keep_original_size)
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
    actual_decoded_width = max_decoded_width;
    actual_decoded_height = max_decoded_height;
    
    std::vector<float> crop_mul_param =  decoder_config.get_crop_param();
    // On the assumption decoded_width and decoded_height is not less than max_decoded_size/2
    auto is_valid_crop = [](uint h, uint w, uint height, uint width)
    {
        return (h < height && w < width); 
    };
    int num_of_attempts = 5;
    double target_area;
    float in_ratio;
    unsigned int crop_width, crop_height, x1, y1;
    for(int i = 0; i < num_of_attempts; i++)
    {
        //renew_params();// Should write a member function
        target_area   = crop_mul_param[0] * original_image_width * original_image_height;
        crop_width  = static_cast<size_t>(std::sqrt(target_area * crop_mul_param[1]));
        crop_height = static_cast<size_t>(std::sqrt(target_area * (1 / crop_mul_param[1])));  
        if(is_valid_crop(crop_height, crop_width, original_image_height, original_image_height)) 
        {
            x1 = static_cast<size_t>(crop_mul_param[2] * (original_image_width  - crop_width));
            y1 = static_cast<size_t>(crop_mul_param[2] * (original_image_height - crop_height));
        }
    }
    constexpr static float ASPECT_RATIO_RANGE[2] = {0.75, 1.33};
    // Fallback on Central Crop
    if( !is_valid_crop(crop_height, crop_width, original_image_height, original_image_height)) 
    {
        in_ratio = static_cast<float>(original_image_width) / original_image_height;
        if(in_ratio < ASPECT_RATIO_RANGE[0])
        {
            crop_width =  original_image_width;
            crop_height = crop_width / ASPECT_RATIO_RANGE[0];
        }
        else if(in_ratio > ASPECT_RATIO_RANGE[1])
        {
            crop_height = original_image_height;
            crop_width  = crop_height  * ASPECT_RATIO_RANGE[1];
        } 
        else
        {
            crop_height = original_image_height;
            crop_width  = original_image_width;
        }
        x1 =  (original_image_width - crop_width) / 2;
        y1 =  (original_image_height - crop_height) / 2;
    }
    
    
    //TODO : Turbo Jpeg supports multiple color packing and color formats, add more as an option to the API TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_RGBA, TJPF_GRAY, TJPF_CMYK , ...
    if( tjDecompress2_partial(m_jpegDecompressor,
                      input_buffer,
                      input_size,
                      output_buffer,
                      max_decoded_width,
                      max_decoded_width * planes,
                      max_decoded_height,
                      tjpf,
                      TJFLAG_FASTDCT,
		      x1, y1, crop_width, crop_height) != 0)

    {
        WRN("Jpeg image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
        return Status::CONTENT_DECODE_FAILED;
    }    

    unsigned char *srcPtrTemp, *dstPtrTemp;

    unsigned int elementsInRow = max_decoded_width * planes;
    unsigned int elementsInCropRow = crop_width * planes;
    unsigned int remainingElements = elementsInRow - elementsInCropRow;

    srcPtrTemp = output_buffer + (y1 * elementsInRow);
    dstPtrTemp = output_buffer;

    unsigned int i = 0;
    for (; i < crop_height; i++)
    {
        memcpy(dstPtrTemp, srcPtrTemp, elementsInCropRow * sizeof(unsigned char));
        memset(dstPtrTemp + elementsInCropRow, 0, remainingElements * sizeof(unsigned char));
        srcPtrTemp += elementsInRow;
        dstPtrTemp += elementsInRow;
    }
    for (; i < max_decoded_height; i++)
    {
        memset(dstPtrTemp, 0, elementsInRow * sizeof(unsigned char));
        dstPtrTemp += elementsInRow;
    }

    actual_decoded_width = crop_width;
    actual_decoded_height = crop_height;
    return Status::OK;
}

FusedCropTJDecoder::~FusedCropTJDecoder() {
    tjDestroy(m_jpegDecompressor);
}
