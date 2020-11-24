/*
Copyright (c) 2015 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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


#ifndef MIVISIONX_HIP_KERNELS_H
#define MIVISIONX_HIP_KERNELS_H
#include <VX/vx.h>
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"

typedef struct AgoConfigScaleMatrix ago_scale_matrix_t;

// arithmetic_kernels

int HipExec_AbsDiff_U8_U8U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes, 
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_AbsDiff_S16_S16S16_Sat(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_U8_U8U8_Wrap(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_U8_U8U8_Sat(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_S16_U8U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_S16_S16U8_Wrap(
        vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_S16_S16U8_Sat(
        vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_S16_S16S16_Wrap(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_S16_S16S16_Sat(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_U8_U8U8_Wrap(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_U8_U8U8_Sat(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_U8U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_S16U8_Wrap(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_S16U8_Sat(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_U8S16_Wrap(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_U8S16_Sat(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_S16S16_Wrap(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_S16S16_Sat(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Mul_U8_U8U8_Wrap_Trunc(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_U8_U8U8_Wrap_Round(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_U8_U8U8_Sat_Trunc(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_U8_U8U8_Sat_Round(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_U8U8_Wrap_Trunc(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_U8U8_Wrap_Round(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_U8U8_Sat_Trunc(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_U8U8_Sat_Round(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16U8_Wrap_Trunc(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16U8_Wrap_Round(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16U8_Sat_Trunc(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16U8_Sat_Round(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16S16_Wrap_Trunc(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16S16_Wrap_Round(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16S16_Sat_Trunc(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16S16_Sat_Round(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Magnitude_S16_S16S16(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Phase_U8_S16S16(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_WeightedAverage_U8_U8U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 alpha
        );


// logical_kernels

int HipExec_And_U8_U8U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U8_U8U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U8_U1U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U8_U1U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U1_U8U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U1_U8U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U1_U1U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U1_U1U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U8_U8U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U8_U8U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U8_U1U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U8_U1U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U1_U8U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U1_U8U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U1_U1U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U1_U1U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U8_U8U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U8_U8U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U8_U1U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U8_U1U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U1_U8U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U1_U8U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U1_U1U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U1_U1U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Not_U8_U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_Not_U8_U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_Not_U1_U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_Not_U1_U1(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );

// statistical_kernels

int HipExec_Threshold_U8_U8_Binary(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdValue
        );
int HipExec_Threshold_U8_U8_Range(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdLower, vx_int32 thresholdUpper
        );
int HipExec_Threshold_U1_U8_Binary(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdValue
        );
int HipExec_Threshold_U1_U8_Range(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdLower, vx_int32 thresholdUpper
        );
int HipExec_ThresholdNot_U8_U8_Binary(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdValue
        );
int HipExec_ThresholdNot_U8_U8_Range(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdLower, vx_int32 thresholdUpper
        );
int HipExec_ThresholdNot_U1_U8_Binary(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdValue
        );
int HipExec_ThresholdNot_U1_U8_Range(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdLower, vx_int32 thresholdUpper
        );

// color_kernels

int HipExec_Lut_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_uint8 *lut
    );

int HipExec_ColorDepth_U8_S16_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift
    );
int HipExec_ColorDepth_U8_S16_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift
    );
int HipExec_ColorDepth_S16_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift
    );

int HipExec_ChannelExtract_U8_U16_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    );
int HipExec_ChannelExtract_U8_U16_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    );
int HipExec_ChannelExtract_U8_U24_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    );
int HipExec_ChannelExtract_U8_U24_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    );
int HipExec_ChannelExtract_U8_U24_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    );
int HipExec_ChannelExtract_U8_U32_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    );
int HipExec_ChannelExtract_U8_U32_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    );
int HipExec_ChannelExtract_U8_U32_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    );
int HipExec_ChannelExtract_U8_U32_Pos3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    );
int HipExec_ChannelCombine_U16_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    );
int HipExec_ChannelCombine_U24_U8U8U8_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
    );
int HipExec_ChannelCombine_U32_U8U8U8_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
    );
int HipExec_ChannelCombine_U32_U8U8U8_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
    );
int HipExec_ChannelCombine_U32_U8U8U8U8_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes,
    const vx_uint8 *pHipSrcImage4, vx_uint32 srcImage4StrideInBytes
    );

int HipExec_ColorConvert_RGBX_RGB(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGB_RGBX(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGB_YUYV(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGB_UYVY(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGBX_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    );
int HipExec_ColorConvert_RGBX_YUYV(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGBX_UYVY(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGBX_IYUV(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcYImage, vx_uint32 srcYImageStrideInBytes,
        const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
        const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes
        );
int HipExec_ColorConvert_RGBX_NV12(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
    );
int HipExec_ColorConvert_RGBX_NV21(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
    );
int HipExec_ColorConvert_IYUV_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    );
int HipExec_ColorConvert_IYUV_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    );
// filter_kernels

int HipExec_Box_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    );
int HipExec_Dilate_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    );
int HipExec_Erode_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    );
int HipExec_Median_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    );
int HipExec_Gaussian_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    );
int HipExec_Convolve_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int16 *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight
    );
int HipExec_Convolve_S16_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int16 *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight
    );
int HipExec_Sobel_S16S16_U8_3x3_GXY(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_int16 *pHipDstImage1, vx_uint32 dstImage1StrideInBytes,
    vx_int16 *pHipDstImage2, vx_uint32 dstImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    );

// geometric_kernels

int HipExec_ScaleImage_U8_U8_Nearest(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight, 
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const ago_scale_matrix_t *matrix
    );
int HipExec_ScaleImage_U8_U8_Bilinear(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight, 
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const ago_scale_matrix_t *matrix
    );
int HipExec_ScaleImage_U8_U8_Area(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight, 
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const ago_scale_matrix_t *matrix
    );
int HipExec_ScaleGaussianHalf_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight, 
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    );
int HipExec_ScaleGaussianHalf_U8_U8_5x5(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight, 
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    );

// vision_kernels

// miscellaneous_kernels

int HipExec_ChannelCopy_U8_U8(
        vx_uint32 dstWidth, vx_uint32 dstHeight, 
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );


int HipExec_ChannelCopy
(
        hipStream_t  stream,
        vx_uint32     dstWidth,
        vx_uint32     dstHeight,
        vx_uint8     * pHipDstImage,
        vx_uint32     dstImageStrideInBytes,
        const vx_uint8    * pHipSrcImage,
        vx_uint32     srcImageStrideInBytes
);

#endif //MIVISIONX_HIP_KERNELS_H
