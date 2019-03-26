/*
Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef _VX_AMD_EXT_WINML_H_
#define _VX_AMD_EXT_WINML_H_

#include "VX/vx.h"
#include <VX/vx_compatibility.h>
#include <vx_ext_amd.h>

#ifndef dimof
#define dimof(x) (sizeof(x)/sizeof(x[0]))
#endif

#if _WIN32 
#define SHARED_PUBLIC __declspec(dllexport)
#else
#define SHARED_PUBLIC __attribute__ ((visibility ("default")))
#endif

vx_node vxCreateNodeByStructure(vx_graph graph, vx_enum kernelenum, vx_reference params[], vx_uint32 num);

#ifdef __cplusplus
extern "C" {
#endif

        /*!***********************************************************************************************************
                                                WinML VX_API_ENTRY C Function NODE
        *************************************************************************************************************/

				/*! \brief [Graph] Creates a WinML import ONNX Model and run function node.
				 * Kernel Name - com.winml.onnx_to_mivisionx
				 * \param [in] graph The reference to the graph.
				 * \param [in] input_1 The ONNX Model Location in vx_scalar.
				 * \param [in] input_2 The ONNX Model Input Tensor Name in vx_scalar.
				 * \param [in] input_3 The ONNX Model Output Tensor Name in vx_scalar.
				 * \param [in] input_4 The Input Tensor in <tt>\ref VX_FLOAT32</tt> format.
				 * \param [in] input_5 The setup Array for each model in <tt>\ref VX_TYPE_SIZE</tt> format.
				 * \param [out] output The output Tensor in <tt>\ref VX_FLOAT32</tt> format.
				 * \param [in] input_6 WinML Deploy Device Kind in vx_scalar [optional] (default: 0).				 
				 * \return <tt>\ref vx_node</tt>.
				 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
				extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtWinMLNode_OnnxToMivisionX
				(
					vx_graph graph,
					vx_scalar modelLocation,
					vx_scalar inputTensorName,
					vx_scalar outputTensorName,
					vx_tensor inputTensor,
					vx_array setupArray,
					vx_tensor outputTensor,
					vx_scalar deviceKind					
				);

				/*! \brief [Graph] Creates a WinML convert image to tensor node.
				 * Kernel Name - com.winml.convert_image_to_tensor
				 * \param [in] graph The reference to the graph.
				 * \param [in] input_1 input image vx_image.
				 * \param [out]  The output Tensor in <tt>\ref VX_FLOAT32</tt> format.
				 * \param [in] input_2 a in vx_scalar.
				 * \param [in] input_3 b in vx_scalar.
				 * \param [in] input_4 reverse channel order in vx_scalar.
				 * \return <tt>\ref vx_node</tt>.
				 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
				extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtWinMLNode_convertImageToTensor
				(
					vx_graph graph,
					vx_image input,
					vx_tensor output,
					vx_scalar a,
					vx_scalar b,
					vx_scalar reverse_channel_order
				);

				/*! \brief [Graph] Creates a output tensor to Top K labels node.
				 * Kernel Name - com.winml.get_top_k_labels
				 * \param [in] graph The reference to the graph.
				 * \param [in] input_1 probability tensor vx_tensor.
				 * \param [in] input_2 label text file location vx_scalar.
				 * \param [out] output_1 Top 1 in vx_scalar.
				 * \param [out] output_2 Top 2 in vx_scalar. [optional]
				 * \param [out] output_3 Top 3 in vx_scalar. [optional]
				 * \param [out] output_4 Top 4 in vx_scalar. [optional]
				 * \param [out] output_5 Top 5 in vx_scalar. [optional]
				 * \return <tt>\ref vx_node</tt>.
				 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
				extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtWinMLNode_getTopKLabels
				(
					vx_graph graph,
					vx_tensor prob_tensor,
					vx_scalar labelFile,
					vx_scalar output_1,
					vx_scalar output_2,
					vx_scalar output_3,
					vx_scalar output_4,
					vx_scalar output_5
				);

#ifdef __cplusplus
}
#endif


#endif
