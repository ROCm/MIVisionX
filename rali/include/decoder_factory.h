#pragma once
#include <memory>
#include "decoder.h"
std::shared_ptr<Decoder> create_decoder(DecoderConfig config);