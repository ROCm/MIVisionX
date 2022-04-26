/* 
Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc. All rights reserved.
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#ifndef __AMD_MIGRAPHX_H__
#define __AMD_MIGRAPHX_H__

#include "vxParameter.h"
#include "vxParamHelper.h"
#include "vxUtils.h"
#include "vx_amd_migraphx.h"

class CVxParamMIGraphX : public CVxParameter {
public:
	CVxParamMIGraphX();
	virtual ~CVxParamMIGraphX();
	virtual int Initialize(vx_context context, vx_graph graph, const char * desc);
	virtual int InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params);
	virtual int Finalize();
    virtual int ReadFrame(int frameNumber);
    virtual int WriteFrame(int frameNumber);
    virtual int CompareFrame(int frameNumber);
	virtual int Shutdown();

private:
    //vx object
	vx_scalar migraphx_prog_scalar;
	// I/O configuration
	std::string modelFileName;
    vx_enum migraphx_prog_e;
    migraphx::program prog;
    migraphx::program_parameters prog_params;
    vx_scalar migraphx_prog;
    vx_tensor input_tensor, output_tensor;
    vx_size input_num_of_dims;
    vx_enum input_data_format;
    vx_size input_dims[4];
    vx_size output_num_of_dims;
    vx_enum output_data_format;
    vx_size output_dims[4];
    vx_status status = 0;
};

#endif /* __AMD_MIGRAPHX_H__ */