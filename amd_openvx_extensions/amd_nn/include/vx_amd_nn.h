/* 
Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
 
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


#ifndef _VX_AMD_NN_H_
#define _VX_AMD_NN_H_

#include <VX/vx.h>

/*! \brief [Graph] Creates a Batch Normalization Layer Node.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [in] inputs The mean tensor data.
 * \param [in] inputs The variance tensor data.
 * \param [in] inputs The scale tensor data.
 * \param [in] inputs The bias tensor data.
 * \param [in] inputs The eps vx_float32 data.
 * \param [out] outputs The output tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxBatchNormalizationLayer(vx_graph graph, vx_tensor inputs, vx_tensor mean, vx_tensor variance, vx_tensor scale, vx_tensor bias, vx_float32 eps, vx_tensor output);

/*! \brief [Graph] Creates a Scale Layer Node.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [in] inputs The scale tensor data.
 * \param [in] inputs The bias tensor data.
 * \param [out] outputs The output tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxScaleLayer(vx_graph graph, vx_tensor inputs, vx_tensor scale, vx_tensor bias, vx_tensor output);

/*! \brief [Graph] Creates a Argmax Layer Node.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [out] outputs The output tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxArgmaxLayer(vx_graph graph, vx_tensor input, vx_reference output);

/*! \brief [Graph] Creates a Image to Tensor Node.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [out] outputs The output tensor data.
 * \param [in] inputs The a vx_float32 data.
 * \param [in] inputs The b vx_float32 data.
 * \param [in] inputs The reverse channel order vx_bool data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvertImageToTensorNode(vx_graph graph, vx_image input, vx_tensor output, vx_float32 a, vx_float32 b, vx_bool reverse_channel_order);

/*! \brief [Graph] Creates a Tensor to Image Node.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [out] outputs The output tensor data.
 * \param [in] inputs The a vx_float32 data.
 * \param [in] inputs The b vx_float32 data.
 * \param [in] inputs The reverse channel order vx_bool data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvertTensorToImageNode(vx_graph graph, vx_tensor input, vx_image output, vx_float32 a, vx_float32 b, vx_bool reverse_channel_order);

/*! \brief [Graph] Creates a Concat Layer Node.
 * \param [in] graph The handle to the graph.
 * \param [out] outputs The output tensor data.
 * \param [in] inputs The input 1 tensor data.
 * \param [in] inputs The input 2 tensor data.
 * \param [in] inputs The input 3 tensor data.
 * \param [in] inputs The input 4 tensor data.
 * \param [in] inputs The input 5 tensor data.
 * \param [in] inputs The input 6 tensor data.
 * \param [in] inputs The input 7 tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxConcatLayer(vx_graph graph, vx_tensor output, vx_tensor input1, vx_tensor input2, vx_tensor input3, vx_tensor input4, vx_tensor input5, vx_tensor input6, vx_tensor input7, vx_tensor input8, vx_int32 axis);

/*! \brief [Graph] Creates a Slice Layer Node.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [out] inputs The output 1 tensor data.
 * \param [out] inputs The output 2 tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxSliceLayer(vx_graph graph, vx_tensor input, vx_tensor output1, vx_tensor output2, vx_tensor output3, vx_tensor output4, vx_tensor output5, vx_tensor output6, vx_tensor output7, vx_tensor output8);

/*! \brief [Graph] Creates a Convolutional Network Upsampling Layer Node.
 * \details Upsampling is done on the width and height dimensions of the <tt>\ref vx_tensor</tt>. Therefore, we use here the term x for the width dimension and y for the height dimension.\n
 * The Upsampling accept input images as tensors of several types. They always output resized images as float32 tensors.
 * This function supports 4D and 3D tensors as input and output. 4D tensors are for batches of images, 3D tensors for individual images.
 * Upsampling use resize method NEAREST_NEIGHBOR.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [out] outputs The output tensor data. Output will have the same number of dimensions as input. Output tensor data type must be same as the inputs. The width and height dimensions of output must be integer multiple of input. The batch and channel dimensions of output and input must be same.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxUpsampleNearestLayer(vx_graph graph, vx_tensor input, vx_tensor output);

/*! \brief [Graph] Creates a Convolutional Network Reshape Layer Node.
 * \details Reshaping is done with alias if available (output tensor will point to input tensor memory). Otherwise it will do a copy.\n
 * This function supports 4D tensors as input and output.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [out] outputs The output tensor data. Output will have the same number of dimensions as input. Output tensor data type must be same as the inputs. The width and height dimensions of output must be integer multiple of input. The batch and channel dimensions of output and input must be same.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxReshapeLayer(vx_graph graph, vx_tensor input, vx_tensor output);

/*! \brief [Graph] Creates a Permute Layer Node.
 * \details Permute is done if the output tensor dimensions change. Otherwise it will do a copy.\n
 * This function supports 4D tensors as input and output.
 * \param [in] graph The handle to the graph.
 * \param [in] input The input tensor data.
 * \param [in] order The required output tensor dimensions.
 * \Order takes values: 
 * \ '0' : 0,1,2,3 (eg:nchw->nchw) 
 * \ '1' : 0,2,3,1 (eg:nchw->nhwc)
 * \param [out] output The output tensor data. Output will have the same number of dimensions as input. Output tensor data type must be same as the inputs. The width, height, batch and channel dimensions of output can be rearranged, but value must be the same as input.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxPermuteLayer(vx_graph graph, vx_tensor input, vx_array order, vx_tensor output);

/*! \brief [Graph] Creates a Prior Box Layer Node.
 * \details Prior box gives the coordinates of the bounding boxes for an image. 
 * This function supports 4D tensors as input and 3D tensor as output.
 * \param [in] graph The handle to the graph.
 * \param [in] input_1 The input tensor data (output of the previous layer)
 * \param [in] input_2 The input tensor data (image data)
 * \param [in] minSize The size for the first prior
 * \param [in] aspect_ratio Array of floats for different bounding boxes, with varying aspect ratio
 * \param [in] flip Input indicating whether aspect ratio can be flipped. Can take values 1(true)/0(false)
 * \param [in] clip Input indicating whether bounding box coordinates should be within [0,1]. Can take values 1(true)/0(false)
 * \param [in] offset Float value to give an offset for each bounding box
 * \param [out] outputs The output tensor data. Output tensor data type must be float. The batch dimensions of output and input must be same. The channel dimensions of the output will be 2. The 3rd dimension will depend on number of boxes.
 * \param [in] maxSize The size for the first prior (optional)
 * \param [in] variance The variance of each prior (optional)
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxPriorBoxLayer(vx_graph graph, vx_tensor input_1, vx_tensor input_2, vx_float32 minSize, vx_array aspect_ratio, vx_int32 flip, vx_int32 clip, 
                                                 vx_float32 offset, vx_tensor output, vx_array variance, vx_float32 maxSize);

/* \brief [Graph] Creates a Crop Layer Node.
 * \details Cropping is done on the dimensions of the input vx_tensor to fit the dimensions of the reference tensor. 
 * This function supports 4D tensors as input and ouput. The type of the tensor can be either float32 or float16.
 * \param [in] graph The handle to the graph.
 * \param [in] input The input tensor data.
 * \param [in] ref The reference tensor data.
 * \param [out] output The cropped tensor data.
 * \param [axis] The dimensions including and trailing 'axis' are cropped. [n x c x h x w]
 * \param [offset1] The offset to set the shift for dimension n. 
 * \param [offset2] The offset to set the shift for dimension c. 
 * \param [offset3] The offset to set the shift for dimension h. 
 * \param [offset4] The offset to set the shift for dimension w.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxCropLayer(vx_graph graph, vx_tensor input, vx_tensor ref, vx_tensor output, vx_scalar axis, vx_scalar offset1, vx_scalar offset2, vx_scalar offset3, vx_scalar offset4);

/* \brief [Graph] Creates a Crop_And_Resize Layer Node.
 * \details Cropping and Resizing is done on the width and height dimensions of the <tt>\ref vx_tensor</tt>. Like Upsampling Layer Node, we use the term x for the width dimension and y for the height dimension.\n
 * This function supports 4D tensors as input and ouput. The type of the tensor can be either float32 or float16.
 * There are two modes for the resize: NEAREST_NEIGHBOR(mode = 0, default) and BILINEAR_INTERPOLATION(mode = 1).
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [out] outputs The output tensor data. Output will have the same number of dimensions as input. Output tensor data type must be same as the inputs.
 * \param [x_coord] x_coord The x coordinate of the upper left point that will be cropped.
 * \param [y_coord] y_coord The y coordinate of the upper left point that will be cropped.
 * \param [width] width The width of the area that will be cropped.
 * \param [height] height The height of the are that will be cropped.
 * \param [scaleFactor] scaleFactor The scale factor that will be used to resize the cropped tensor.
 * \param [mode] mode The mode to decide which method will be used for the resize.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxCropAndResizeLayer(vx_graph graph, vx_tensor input, vx_tensor output, vx_scalar x_coord, vx_scalar y_coord, vx_scalar width, vx_scalar height, vx_scalar scaleFactor, vx_scalar mode);

/* \brief [Graph] Creates a Tensor_Min Layer Node.
 * \details Performs element-wise min on element values in the input <tt>\ref vx_tensor</tt>.
 * This function supports 4D tensors as input and ouput. The type of the tensor can be either float32 or float16.
 * \param [in] graph The handle to the graph.
 * \param [in] input The first input tensor data.
 * \param [in] input2 The second input tensor data. The dimensions and sizes of input2 match those of input1, unless the vx_tensor of one or more dimensions in input2 is 1. 
 *                    In this case, those dimensions are treated as if this tensor was expanded to match the size of the corresponding dimension of input1, and data was duplicated on all terms in that dimension. 
 *                    After this expansion, the dimensions will be equal. The data type must match the data type of input1.
 * \param [in] policy A vx_convert_policy_e enumeration.
 * \param [out] output The output tensor data with the same dimensions as the input tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorMinNode(vx_graph graph, vx_tensor input, vx_tensor input2, vx_enum policy, vx_tensor output);

/* \brief [Graph] Creates a Tensor_Max Layer Node.
 * \details Performs element-wise max on element values in the input <tt>\ref vx_tensor</tt>.
 * This function supports 4D tensors as input and ouput. The type of the tensor can be either float32 or float16.
 * \param [in] graph The handle to the graph.
 * \param [in] input The first input tensor data.
 * \param [in] input2 The second input tensor data. The dimensions and sizes of input2 match those of input1, unless the vx_tensor of one or more dimensions in input2 is 1. 
 *                    In this case, those dimensions are treated as if this tensor was expanded to match the size of the corresponding dimension of input1, and data was duplicated on all terms in that dimension. 
 *                    After this expansion, the dimensions will be equal. The data type must match the data type of input1.
 * \param [in] policy A vx_convert_policy_e enumeration.
 * \param [out] output The output tensor data with the same dimensions as the input tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorMaxNode(vx_graph graph, vx_tensor input, vx_tensor input2, vx_enum policy, vx_tensor output);

/* \brief [Graph] Creates a Detection Output Layer Node.
 * \details Gives the details of the detected ouutputs in an image with their label, confidence and the bounding box coordinates.
 * This function supports three 4D tensors as input and one 4D tensor as ouput. The type of the tensor can be either float32 or float16.
 * \param [in] graph The handle to the graph.
 * \param [in] input1 The first input tensor data (location values).
 * \param [in] input2 The second input tensor data (confidence values). 
 * \param [in] input3 The third input tensor data (prior box values).
 * \param [in] num_classes Integer value: Number of output classes. (example: tiny yolo = 20 classes)
 * \param [in] share_location Integer value: Label values change based on this
 * \param [in] background_label_id Integer value: Ignores the background classes
 * \param [in] nms_threshold Float value: NMS-Threshold for output boxes
 * \param [in] code_type Integer value: Decides if the bounding boxes are Center-type or Corner-type
 * \param [in] keep_top_k Integer value: Tells the number of output boxes to keep
 * \param [in] variance_encoded_in_target Integer value: Helps calculate the bounding box coordinates
 * \param [out] output The output tensor data with the same dimensions as the input tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxDetectionOutputLayer(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_tensor input3, vx_int32 num_classes, vx_int32 share_location, vx_int32 background_label_id, vx_float32 nms_threshold,
                                                        vx_int32 code_type, vx_int32 keep_top_k, vx_int32 variance_encoded_in_target, vx_tensor output);

/*! \brief [Graph] Creates a Cast Layer Node.
 * \details Converts all the elements of the input tensor to the data type specified by input_2 of the node.\n
 * This function supports 2D or 4D tensors as input and output.
 * \param [in] graph The handle to the graph.
 * \param [in] input The input tensor data. Can be VX_TYPE_FLOAT32, VX_TYPE_INT32, VX_TYPE_INT64.
 * \param [in] output_data_type The required output tensor data type. Integer value between 0-13.
 * \param [out] output The output tensor data. Output will have the same number of dimensions as input. Output tensor data type will be that specified by 'to'. 
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxCastLayer(vx_graph graph, vx_tensor input, vx_int32 output_data_type, vx_tensor output);

/* \brief [Graph] Creates a Tensor_Exp Layer Node.
 * \details Calculates the element-wise exponential of the element values in the input <tt>\ref vx_tensor</tt>.
 * This function supports 4D tensors as input and ouput. The type of the tensor can be either float32 or float16.
 * \param [in] graph The handle to the graph.
 * \param [in] input The input tensor data.
 * \param [out] output The output tensor data with the same dimensions as the input tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorExpNode(vx_graph graph, vx_tensor input, vx_tensor output);

/* \brief [Graph] Creates a Tensor_Log Layer Node.
 * \details Calculates the element-wise natural log of the element values in the input <tt>\ref vx_tensor</tt>.
 * This function supports 4D tensors as input and ouput. The type of the tensor can be either float32 or float16.
 * \param [in] graph The handle to the graph.
 * \param [in] input The input tensor data.
 * \param [out] output The output tensor data with the same dimensions as the input tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorLogNode(vx_graph graph, vx_tensor input, vx_tensor output);

#endif