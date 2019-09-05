#pragma once

#include "device_code.h"

const static std::string utility_program_name = "utility";

const static std::vector<std::string> utility_kernel_names = { "copyInt8ToFloat" };

const static std::string utility_source = 
"__kernel void copyInt8ToFloat(__global const unsigned char* in, __global float* out, unsigned out_offset, unsigned size) {"
"	int i = get_global_id(0);"
"   if(i >= size) return; "
"   out[i+out_offset] = 2*(((float)(in[i]))/256.0 - 0.5); }";

class OCLUtility : public DeviceCode {
    public:
    OCLUtility(): DeviceCode(utility_source, utility_program_name, utility_kernel_names){}
    // TODO : delete other implicit constructors
};


