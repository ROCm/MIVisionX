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

#define _CRT_SECURE_NO_WARNINGS
#include "amdMIGraphX.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamMIGraphX
//
CVxParamMIGraphX::CVxParamMIGraphX() {
    // vx configuration
    modelFileName = "";
    migraphx_prog_scalar = nullptr;
}

CVxParamMIGraphX::~CVxParamMIGraphX() {
    Shutdown();
}

int CVxParamMIGraphX::Shutdown(void) {
    if (migraphx_prog_scalar) {
        vxReleaseScalar(&migraphx_prog_scalar);
    }
    return 0;
}

int CVxParamMIGraphX::Initialize(vx_context context, vx_graph graph, const char * desc) {
    // get object parameters and create object
    const char * ioParams = desc;
    char objType[64], inputType[64], fileName[256];
    ioParams = ScanParameters(desc, "migraphx:<input-type>,<file-name>", "s:s,S", objType, inputType, fileName);
    if (!_stricmp(objType, "migraphx")) {
        if (!_stricmp(inputType, "onnx")) {
            m_fileNameRead.assign(RootDirUpdated(fileName));
            modelFileName = m_fileNameRead;
            status = amdMIGraphXcompile(modelFileName.c_str(), &prog,
                    &input_num_of_dims, input_dims, &input_data_format,
                    &output_num_of_dims, output_dims, &output_data_format, false, false);
            if (status) {
                printf("ERROR: amdMIGraphXcompile failed => %d (%s)\n", status, ovxEnum2Name(status));
                return status;
            }
            migraphx_prog_e = vxRegisterUserStruct(context, sizeof(prog));
            migraphx_prog_scalar = vxCreateScalarWithSize(context, migraphx_prog_e, &prog, sizeof(prog));
        }
        else if (inputType == "json") {
            ReportError("ERROR: unsupported migraphx input type: %s\n", inputType);
        }
    }
    else ReportError("ERROR: unsupported migraphx type: %s\n", desc);

    vx_status ovxStatus = vxGetStatus((vx_reference)migraphx_prog_scalar);
    if (ovxStatus != VX_SUCCESS) {
        printf("ERROR: scalar creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
        if (migraphx_prog_scalar) vxReleaseScalar(&migraphx_prog_scalar);
        throw - 1;
    }
    m_vxObjRef = (vx_reference)migraphx_prog_scalar;

    // io initialize
    return InitializeIO(context, graph, m_vxObjRef, ioParams);
}

int CVxParamMIGraphX::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params) {
    return 0;
}

int CVxParamMIGraphX::Finalize() {
    return 0;
}

int CVxParamMIGraphX::ReadFrame(int frameNumber) {
    return 0;
}

int CVxParamMIGraphX::WriteFrame(int frameNumber) {
    return 0;
}

int CVxParamMIGraphX::CompareFrame(int frameNumber) {
    return 0;
}
