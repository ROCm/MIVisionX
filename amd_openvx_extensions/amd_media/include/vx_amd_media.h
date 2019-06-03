/*
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef __VX_AMD_MEDIA_H__
#define __VX_AMD_MEDIA_H__

#include <VX/vx.h>

#ifdef  __cplusplus
extern "C" {
#endif


/*! \brief [Graph] Creates a Scale Layer Node.
 * \param [in] graph The handle to the graph.
 * \param [in] input_str The input string specifying the filename URL and decoding mode.
 * \param [out] output output image from the decoder.
 * \param [out] aux_data from the decoder: used when encoding
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL amdMediaDecoderNode(vx_graph graph, const char *input_str, vx_image output, vx_array aux_data, vx_int32 loop_decode=0);

/*! \brief [Graph] Creates a Scale Layer Node.
 * \param [in] graph The handle to the graph.
 * \param [in] output_str The output string specifying the filename URL.
 * \param [in] input input image for the encoder.
 * \param [in] aux_data_in input aux data
 * \param [out] aux_data_out output aux data
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL amdMediaEncoderNode(vx_graph graph, const char *output_str, vx_image input, vx_array aux_data_in, vx_array aux_data_out);

#ifdef  __cplusplus
}
#endif

#endif
