/*
Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

// Small C-API probes for MIVisionX gaps; one "PROBE <name> <values...>" line per check.
// Nothing here is expected to crash (the crash reproducers live in roi_release.c).
// The target comes from AGO_DEFAULT_TARGET, set by the caller.
#include <VX/vx.h>
#include <VX/vx_compatibility.h>

#include <cstdio>
#include <cstring>
#include <vector>

static const char* st(vx_status s) {
    switch (s) {
        case VX_SUCCESS: return "VX_SUCCESS";
        case VX_FAILURE: return "VX_FAILURE";
        case VX_ERROR_NOT_IMPLEMENTED: return "VX_ERROR_NOT_IMPLEMENTED";
        case VX_ERROR_NOT_SUPPORTED: return "VX_ERROR_NOT_SUPPORTED";
        case VX_ERROR_INVALID_FORMAT: return "VX_ERROR_INVALID_FORMAT";
        case VX_ERROR_INVALID_PARAMETERS: return "VX_ERROR_INVALID_PARAMETERS";
        case VX_ERROR_INVALID_VALUE: return "VX_ERROR_INVALID_VALUE";
        case VX_ERROR_INVALID_DIMENSION: return "VX_ERROR_INVALID_DIMENSION";
        case VX_ERROR_NO_RESOURCES: return "VX_ERROR_NO_RESOURCES";
        case VX_ERROR_INVALID_REFERENCE: return "VX_ERROR_INVALID_REFERENCE";
        case VX_ERROR_INVALID_NODE: return "VX_ERROR_INVALID_NODE";
        case VX_ERROR_INVALID_GRAPH: return "VX_ERROR_INVALID_GRAPH";
        default: return "other";
    }
}

int main() {
    vx_context ctx = vxCreateContext();
    vx_status cs = vxGetStatus((vx_reference)ctx);
    printf("PROBE context %s %d\n", st(cs), cs);
    if (cs != VX_SUCCESS) return 1;

    // VX_DF_IMAGE_RGBA is declared in the shipped vx_types.h
    vx_image rgba = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_RGBA);
    vx_status s = vxGetStatus((vx_reference)rgba);
    printf("PROBE rgba-create %s %d\n", st(s), s);

    {
        vx_graph g = vxCreateGraph(ctx);
        vx_image a = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8), b = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
        vx_node n = vxCopyNode(g, (vx_reference)a, (vx_reference)b);
        vx_status ns = vxGetStatus((vx_reference)n);
        vx_status vs = ns == VX_SUCCESS ? vxVerifyGraph(g) : ns;
        vx_status ps = vs == VX_SUCCESS ? vxProcessGraph(g) : vs;
        printf("PROBE copy-node %s %s %s\n", st(ns), st(vs), st(ps));
        vxReleaseGraph(&g);
    }

    {
        vx_threshold t = vxCreateThresholdForImage(ctx, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
        vx_enum type = VX_THRESHOLD_TYPE_RANGE;
        vx_status ts = vxSetThresholdAttribute(t, VX_THRESHOLD_TYPE, &type, sizeof(type));
        vx_df_image fmt = VX_DF_IMAGE_S16;
        vx_status fs = vxSetThresholdAttribute(t, VX_THRESHOLD_INPUT_FORMAT, &fmt, sizeof(fmt));
        printf("PROBE threshold-readonly-attribute %s %s\n", st(ts), st(fs));
        vxReleaseThreshold(&t);
    }

    {
        const vx_uint32 W = 64, H = 64;
        vx_graph g = vxCreateGraph(ctx);
        vx_image in = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8), out = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8);
        std::vector<unsigned char> buf(W * H);
        for (size_t i = 0; i < buf.size(); i++) buf[i] = (unsigned char)(i * 7);
        vx_rectangle_t rect = {0, 0, W, H};
        vx_imagepatch_addressing_t addr;
        memset(&addr, 0, sizeof(addr));
        addr.dim_x = W; addr.dim_y = H; addr.stride_x = 1; addr.stride_y = W;
        vxCopyImagePatch(in, &rect, 0, &addr, buf.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        vx_convolution conv = vxCreateConvolution(ctx, 9, 7);
        std::vector<vx_int16> coef(9 * 7, 1);
        vx_status cps = vxCopyConvolutionCoefficients(conv, coef.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        vx_uint32 scale = 64;
        vxSetConvolutionAttribute(conv, VX_CONVOLUTION_SCALE, &scale, sizeof(scale));
        vx_node n = vxConvolveNode(g, in, conv, out);
        vx_status ns = vxGetStatus((vx_reference)n);
        vx_status vs = ns == VX_SUCCESS ? vxVerifyGraph(g) : ns;
        vx_status ps = vs == VX_SUCCESS ? vxProcessGraph(g) : vs;
        printf("PROBE convolution-9x7 %s %s %s %s\n", st(cps), st(ns), st(vs), st(ps));
        vxReleaseGraph(&g);
    }
    vxReleaseContext(&ctx);
    return 0;
}
