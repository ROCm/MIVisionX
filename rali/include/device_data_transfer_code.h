#pragma once

#include "device_code.h"

const static std::string data_transfer_program_name = "utility";

const static std::vector<std::string> data_transfer_kernel_names = {"copyInt8ToNHWC","copyInt8ToNCHW"};

const static std::string data_transfer_source =
"__kernel void copyInt8ToNHWC(__global const unsigned char* in, __global float* out, unsigned out_offset, unsigned w, unsigned h, unsigned c, float multiplier, float offset, unsigned reverse_channels) {"
"	int i = get_global_id(0);"
"   unsigned size = h*w*c;"
"   if(i >= size) return; "
"   out[i+out_offset] = multiplier*((float)in[i])+offset; }"
""
"__kernel void copyInt8ToNCHW(__global const unsigned char* in, __global float* out, unsigned out_offset, unsigned w, unsigned h, unsigned c, float multiplier, float offset, unsigned reverse_channels) {"
"	int i = get_global_id(0);"
"   unsigned channel_size = h*w;"
"   unsigned size = channel_size*c;"
"   if(i >= size) return; "
"   unsigned channel_idx = i % c;"
"   unsigned pixel_idx = i % channel_size;"
"   float out_val = 0;"
"   if(reverse_channels) {"
"       out_val = multiplier*((float)(in[c*pixel_idx+c-channel_idx-1]))+offset; "
"   } else {"
"       out_val = multiplier*((float)(in[c*pixel_idx+channel_idx]))+offset; "
"   }"
"   out [out_offset + channel_idx*channel_size + pixel_idx] = out_val;}";

class OCLUtility : public DeviceCode {
    public:
    OCLUtility(): DeviceCode(data_transfer_source, data_transfer_program_name, data_transfer_kernel_names){}
    // TODO : delete other implicit constructors
};


