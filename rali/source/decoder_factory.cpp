
#include <decoder.h>
#include <turbo_jpeg_decoder.h>
#include <fused_crop_decoder.h>
#include "decoder_factory.h"
#include "commons.h"

std::shared_ptr<Decoder> create_decoder(DecoderConfig config) {
    switch(config.type())
    {
        case DecoderType::TURBO_JPEG:
            return std::make_shared<TJDecoder>();
            break;
        case DecoderType::FUSED_TURBO_JPEG:
            return std::make_shared<FusedCropTJDecoder>();
            break;
        default:
            THROW("Unsupported decoder type "+ TOSTR(config.type()));
    }
}
