#pragma once

#include "device_code.h"

const static std::string data_transfer_program_name = "utility";

const static std::vector<std::string> data_transfer_kernel_names = {"copyInt8ToNHWC","copyInt8ToNCHW"};

const static std::string data_transfer_source =
"__kernel void copyInt8ToNHWC(__global const unsigned char* in, __global float* out, unsigned out_offset, unsigned w, unsigned h, unsigned c, float multiplier, float offset) {"
"	int i = get_global_id(0);"
"   unsigned size = h*w*c;"
"   if(i >= size) return; "
"   out[i+out_offset] = multiplier*((float)in[i])+offset; }"
""
"__kernel void copyInt8ToNCHW(__global const unsigned char* in, __global float* out, unsigned out_offset, unsigned w, unsigned h, unsigned c, float multiplier, float offset) {"
"	int i = get_global_id(0);"
"   unsigned plane_size = h*w;"
"   unsigned size = plane_size*c;"
"   if(i >= size) return; "
"   unsigned plane_idx = i % c;"
"   unsigned plane_offset = plane_idx * plane_size;"
"   out[(i%plane_size)+plane_offset+out_offset] = multiplier*((float)in[i])+offset; }";

class OCLUtility : public DeviceCode {
    public:
    OCLUtility(): DeviceCode(data_transfer_source, data_transfer_program_name, data_transfer_kernel_names){}
    // TODO : delete other implicit constructors
};


