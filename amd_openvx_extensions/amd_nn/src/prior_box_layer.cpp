#include "kernels.h"

static vx_status VX_CALLBACK validate(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[])
{   
    // check tensor dims.
    vx_enum type;
    vx_size num_dims;
    vx_size input_dims_1[4], input_dims_2[4],  output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type!= VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type!= VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_2, sizeof(input_dims_2)));

    vx_enum scalar_type;   
    vx_float32 minSize;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &minSize, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(minSize < 0.0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: priorbox: #3 scalar type=%f (min_size must be positive)\n", minSize);


    vx_size aspect_ratio_cap = 0;
    vx_size itemsize = 0;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[3], VX_ARRAY_ITEMTYPE, &type, sizeof(type)));
    if(type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[3], VX_ARRAY_CAPACITY, &aspect_ratio_cap, sizeof(aspect_ratio_cap)));
    if(aspect_ratio_cap != 2) return VX_ERROR_INVALID_DIMENSION;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[3], VX_ARRAY_ITEMSIZE, &itemsize, sizeof(itemsize)));
    if(itemsize != sizeof(float)) return VX_ERROR_INVALID_TYPE;
    
    vx_int32 flip;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &flip, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(flip < 0 || flip > 1) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: priorbox: #5 scalar value=%d (flip must be 1(true)/0(false))\n", flip);
    
    
    vx_int32 clip;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &clip, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(clip < 0 || clip > 1) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: priorbox: #6 scalar value=%d (clip must be 1(true)/0(false))\n", clip);
    
    vx_float32 offset;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[6], &offset, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(offset < 0.0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: priorbox: #7 scalar type=%f (offset must be positive)\n", offset);


    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 3) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));


    if(parameters[8]) {
        vx_float32 maxSize;
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
        if(scalar_type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[8], &maxSize, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        if(maxSize < 0.0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: priorbox: #8 scalar type=%f (max_size must be positive)\n", maxSize);
    }

    vx_size var_capacity = 0;
    if(parameters[9]) {
        ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[9], VX_ARRAY_ITEMTYPE, &type, sizeof(type)));
        if(type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
        ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[9], VX_ARRAY_CAPACITY, &var_capacity, sizeof(var_capacity)));
        if(var_capacity != 4) return VX_ERROR_INVALID_TYPE;
        ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[9], VX_ARRAY_ITEMSIZE, &itemsize, sizeof(itemsize)));
        if(itemsize != sizeof(float)) return VX_ERROR_INVALID_TYPE;
    }

    // output tensor configuration
    num_dims = 3;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[7], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[7], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[7], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    return VX_SUCCESS;

}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
    vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
    )
{

    supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK opencl_codegen(
    vx_node node,                                  // [input] node
    const vx_reference parameters[],               // [input] parameters
    vx_uint32 num,                                 // [input] number of parameters
    bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
    char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
    std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
    std::string& opencl_build_options,             // [output] options for clBuildProgram()
    vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
    vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
    vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
    vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
    vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
)
{
    

    //get tensor dimensions
    vx_size input_dims_1[4], input_dims_2[4], output_dims[3];
    vx_size num_of_dims;
    vx_enum type;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_2, sizeof(input_dims_2)));
    
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

    vx_int32 clip, flip;
    vx_float32 minSize, maxSize, offset;

    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &minSize, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &flip, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &clip, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[6], &offset, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    
    if(parameters[8])
    {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[8], &maxSize, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    else
    {
        maxSize = 0.0;
    }
    
   
    vx_array aspect_ratio_buf;
    vx_size aspect_ratio_cap, aspect_ratio_numitems;
    aspect_ratio_buf = (vx_array)parameters[3];
    ERROR_CHECK_STATUS(vxQueryArray(aspect_ratio_buf, VX_ARRAY_CAPACITY, &aspect_ratio_cap, sizeof(aspect_ratio_cap)));
    ERROR_CHECK_STATUS(vxQueryArray(aspect_ratio_buf, VX_ARRAY_NUMITEMS, &aspect_ratio_numitems, sizeof(aspect_ratio_numitems)));
    ERROR_CHECK_STATUS(vxReleaseArray(&aspect_ratio_buf));
    
    
       
    vx_array variance_buf;
    vx_size var_capacity, var_numitems;
    if(parameters[9]) {
        variance_buf = (vx_array)parameters[9];
        ERROR_CHECK_STATUS(vxQueryArray(variance_buf, VX_ARRAY_CAPACITY, &var_capacity, sizeof(var_capacity)));
        ERROR_CHECK_STATUS(vxQueryArray(variance_buf, VX_ARRAY_NUMITEMS, &var_numitems, sizeof(var_numitems)));
        ERROR_CHECK_STATUS(vxReleaseArray(&variance_buf));
        
    }
   
    
   
    strcpy(opencl_kernel_function_name, "prior_box_layer");
    

    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    const int layer_height = input_dims_1[2];
    const int layer_width = input_dims_1[3];
    const int img_height = input_dims_2[2];
    const int img_width = input_dims_2[3];
    
    const int output_num = output_dims[0] * output_dims[1] * output_dims[2];
    const int output_dims_ch2 = output_dims[2];

    opencl_work_dim = 2;
    opencl_global_work[0] = layer_width;
    opencl_global_work[1] = layer_height;


    if (num_of_dims == 4)
    {
        char item[8192];
        if(parameters[9])
        {
            sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in_1, uint in_1_offset, uint4 in_1_stride, __global uchar * in_2, uint in_2_offset, uint4 in_2_stride, const float s_minSize,"\
                "                 __global uchar * aspect_ratio_buf, uint aspect_ratio_offset, uint aspect_ratio_num, const uint s_flip, const uint s_clip, const float s_offset,"\
                "                 __global uchar * out, uint out_offset, uint4 out_stride, const float s_maxSize, __global uchar * variance_buf, uint variance_offset, uint variance_num) \n"
                "{   \n"
                "   __global uchar* out_ptr = out; \n"
                "   const int imgWidth = %d;"
                "   const int imgHeight = %d;"
                "   const int layerWidth = %d;"
                "   const int layerHeight = %d;"
                "   const float minSize = %f; \n"
                "   const float maxSize = %f; \n"
                "   const int clip = %d; \n"
                "   const int flip = %d; \n"
                "   const float offset = %f; \n"
                "   const int output_num = %d; \n"
                "   const int output_dims_ch2 = %d; \n"
                "   const float step_x = (float)imgWidth /(float)layerWidth; \n"
                "   const float step_y = (float)imgHeight /(float)layerHeight; \n"
                "   uint x = get_global_id(0); \n "
                "   uint y = get_global_id(1); \n "
                "   float center_x = (x+offset) * step_x; \n"
                "   float center_y = (y+offset) * step_y; \n"
                "   float box_width, box_height; \n"
                "   box_width = minSize; \n"
                "   box_height = minSize; \n"
                "   *(__global float *)&out[0] = (center_x - box_width / 2.) / imgWidth; \n"
                "   out += out_stride.s0; \n"
                "   *(__global float *)&out[0] = (center_y - box_height / 2.) / imgHeight; \n"
                "   out += out_stride.s0; \n"
                "   *(__global float *)&out[0] = (center_x + box_width / 2.) / imgWidth; \n"
                "   out += out_stride.s0; \n"
                "   *(__global float *)&out[0] = (center_y + box_height / 2.) / imgHeight; \n"
                "   if(maxSize > 0) {"
                "       box_width = sqrt((float)(minSize * maxSize)); \n"
                "       box_height = sqrt((float)(minSize * maxSize)); \n"
                "       out += out_stride.s0; \n" 
                "       *(__global float *)&out[0] = (center_x - box_width / 2.) / imgWidth; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_y - box_height / 2.) / imgHeight; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_x + box_width / 2.) / imgWidth; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_y + box_height / 2.) / imgHeight; \n"
                "   } \n"
                "   int r = 0; \n"
                "   while(r < aspect_ratio_num) \n"
                "   { \n"
                "       float ar = ((__global float *)(aspect_ratio_buf+aspect_ratio_offset))[r]; \n" 
                "       if(fabs(ar - (float)1.) < 1e-6) \n"
                "       { \n"
                "           continue;"
                "       } \n"
                "       box_width = minSize * sqrt(ar); \n"
                "       box_height = minSize / sqrt(ar); \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_x - box_width / 2.) / imgWidth; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_y - box_height / 2.) / imgHeight; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_x + box_width / 2.) / imgWidth; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_y + box_height / 2.) / imgHeight; \n"
                "       if(flip == 1) \n"
                "       { \n"
                "           float ar_flip=  1/ar; \n" 
                "           if(fabs(ar_flip - (float)1.) < 1e-6) \n"
                "           { \n"
                "               continue;"
                "           } \n"
                "           box_width = minSize * sqrt(ar_flip); \n"
                "           box_height = minSize / sqrt(ar_flip); \n"
                "           out += out_stride.s0; \n"
                "           *(__global float *)&out[0] = (center_x - box_width / 2.) / imgWidth; \n"
                "           out += out_stride.s0; \n"
                "           *(__global float *)&out[0] = (center_y - box_height / 2.) / imgHeight; \n"
                "           out += out_stride.s0; \n"
                "           *(__global float *)&out[0] = (center_x + box_width / 2.) / imgWidth; \n"
                "           out += out_stride.s0; \n"
                "           *(__global float *)&out[0] = (center_y + box_height / 2.) / imgHeight; \n"
                "       }"
                "       r++; \n"
                "   } \n"
                "   if(clip == 1) \n"
                "   { \n" 
                "       int idx = 0; \n"
                "       out = out_ptr; \n"
                "       while(idx < (output_dims_ch2 - 1)) { \n"
                "           ((__global float *)(out+out_offset))[idx] = min(max((float)out[idx], (float)0.), (float)1.); \n"
                "           idx++; \n"
                "       } \n"
                "   }   \n"
                "   int count = output_dims_ch2; \n"
                "   out = out_ptr; \n"
                "   while(count < output_num) { \n"
                "   ((__global float *)(out+out_offset))[count++] = ((__global float *)(variance_buf+variance_offset))[0];\n" 
                "   ((__global float *)(out+out_offset))[count++] = ((__global float *)(variance_buf+variance_offset))[1];\n"
                "   ((__global float *)(out+out_offset))[count++] = ((__global float *)(variance_buf+variance_offset))[2];\n" 
                "   ((__global float *)(out+out_offset))[count++] = ((__global float *)(variance_buf+variance_offset))[3];\n"
                "   }\n"
                "}\n", opencl_kernel_function_name, img_width, img_height, layer_width, layer_height, minSize, maxSize, clip, flip, offset, output_num, output_dims_ch2);
            opencl_kernel_code = item;
        }
        else{
            sprintf(item,
                   "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in_1, uint in_1_offset, uint4 in_1_stride, __global uchar * in_2, uint in_2_offset, uint4 in_2_stride, const float s_minSize,"\
                "                 __global uchar * aspect_ratio_buf, uint aspect_ratio_offset, uint aspect_ratio_num, const uint s_flip, const uint s_clip, const float s_offset,"\
                "                 __global uchar * out, uint out_offset, uint4 out_stride, const float s_maxSize) \n"
                "{   \n"
                "   __global uchar* out_ptr = out; \n"
                "   const int imgWidth = %d;"
                "   const int imgHeight = %d;"
                "   const int layerWidth = %d;"
                "   const int layerHeight = %d;"
                "   const float minSize = %f; \n"
                "   const float maxSize = %f; \n"
                "   const int clip = %d; \n"
                "   const int flip = %d; \n"
                "   const float offset = %f; \n"
                "   const int output_num = %d; \n"
                "   const int output_dims_ch2 = %d; \n"
                "   const float step_x = (float)imgWidth /(float)layerWidth; \n"
                "   const float step_y = (float)imgHeight /(float)layerHeight; \n"
                "   uint x = get_global_id(0); \n "
                "   uint y = get_global_id(1); \n "
                "   float center_x = (x+offset) * step_x; \n"
                "   float center_y = (y+offset) * step_y; \n"
                "   float box_width, box_height; \n"
                "   box_width = minSize; \n"
                "   box_height = minSize; \n"
                "   *(__global float *)&out[0] = (center_x - box_width / 2.) / imgWidth; \n"
                "   out += out_stride.s0; \n"
                "   *(__global float *)&out[0] = (center_y - box_height / 2.) / imgHeight; \n"
                "   out += out_stride.s0; \n"
                "   *(__global float *)&out[0] = (center_x + box_width / 2.) / imgWidth; \n"
                "   out += out_stride.s0; \n"
                "   *(__global float *)&out[0] = (center_y + box_height / 2.) / imgHeight; \n"
                "   if(maxSize > 0) {"
                "       box_width = sqrt((float)(minSize * maxSize)); \n"
                "       box_height = sqrt((float)(minSize * maxSize)); \n"
                "       out += out_stride.s0; \n" 
                "       *(__global float *)&out[0] = (center_x - box_width / 2.) / imgWidth; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_y - box_height / 2.) / imgHeight; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_x + box_width / 2.) / imgWidth; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_y + box_height / 2.) / imgHeight; \n"
                "   } \n"
                "   int r = 0; \n"
                "   while(r < aspect_ratio_num) \n"
                "   { \n"
                "       float ar = ((__global float *)(aspect_ratio_buf+aspect_ratio_offset))[r]; \n" 
                "       if(fabs(ar - (float)1.) < 1e-6) \n"
                "       { \n"
                "           continue;"
                "       } \n"
                "       box_width = minSize * sqrt(ar); \n"
                "       box_height = minSize / sqrt(ar); \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_x - box_width / 2.) / imgWidth; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_y - box_height / 2.) / imgHeight; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_x + box_width / 2.) / imgWidth; \n"
                "       out += out_stride.s0; \n"
                "       *(__global float *)&out[0] = (center_y + box_height / 2.) / imgHeight; \n"
                "       if(flip == 1) \n"
                "       { \n"
                "           float ar_flip=  1/ar; \n" 
                "           if(fabs(ar_flip - (float)1.) < 1e-6) \n"
                "           { \n"
                "               continue;"
                "           } \n"
                "           box_width = minSize * sqrt(ar_flip); \n"
                "           box_height = minSize / sqrt(ar_flip); \n"
                "           out += out_stride.s0; \n"
                "           *(__global float *)&out[0] = (center_x - box_width / 2.) / imgWidth; \n"
                "           out += out_stride.s0; \n"
                "           *(__global float *)&out[0] = (center_y - box_height / 2.) / imgHeight; \n"
                "           out += out_stride.s0; \n"
                "           *(__global float *)&out[0] = (center_x + box_width / 2.) / imgWidth; \n"
                "           out += out_stride.s0; \n"
                "           *(__global float *)&out[0] = (center_y + box_height / 2.) / imgHeight; \n"
                "       }"
                "       r++; \n"
                "   } \n"
                "   if(clip == 1) \n"
                "   { \n" 
                "       int idx = 0; \n"
                "       out = out_ptr; \n"
                "       while(idx < (output_dims_ch2 - 1)) { \n"
                "           ((__global float *)(out+out_offset))[idx] = min(max((float)out[idx], (float)0.), (float)1.); \n"
                "           idx++; \n"
                "       } \n"
                "   } \n"
                "   int count = output_dims_ch2; \n"
                "   out = out_ptr; \n"
                "   while(count < output_num) { \n"
                "   ((__global float *)(out+out_offset))[count++] = 0.1;\n" 
                "   ((__global float *)(out+out_offset))[count++] = 0.1;\n"
                "   ((__global float *)(out+out_offset))[count++] = 0.1;\n" 
                "   ((__global float *)(out+out_offset))[count++] = 0.1;\n"
                "   }\n"
                "}\n", opencl_kernel_function_name, img_width, img_height, layer_width, layer_height, minSize, maxSize, clip, flip, offset, output_num, output_dims_ch2);
            opencl_kernel_code = item;
        }
    }

    return VX_SUCCESS;
}


//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel publisher.
vx_status publishPriorBoxLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.prior_box_layer", VX_KERNEL_PRIOR_BOX_LAYER_AMD, host_kernel, 10, validate, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 7, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_OPTIONAL));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS;
}


/*first prior: aspect_ratio = 1; size = min_size
*second prior: aspect_ratio = 1; size = sqrt(min_size * max_size) 
*/
VX_API_ENTRY vx_node VX_API_CALL vxPriorBoxLayer(vx_graph graph, vx_tensor input_1, vx_tensor input_2, vx_float32 minSize, vx_array aspect_ratio, vx_int32 flip, vx_int32 clip, 
                                                 vx_float32 offset, vx_tensor output, vx_float32 maxSize, vx_array variance)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_minSize = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &minSize, sizeof(minSize));
        vx_scalar s_flip = vxCreateScalarWithSize(context, VX_TYPE_INT32, &flip, sizeof(flip));
        vx_scalar s_clip = vxCreateScalarWithSize(context, VX_TYPE_INT32, &clip, sizeof(clip));
        vx_scalar s_offset = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &offset, sizeof(offset));
        vx_scalar s_maxSize = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &maxSize, sizeof(maxSize));

        vx_reference params[] = {
            (vx_reference)input_1,
            (vx_reference)input_2,
            (vx_reference)s_minSize,
            (vx_reference)aspect_ratio,
            (vx_reference)s_flip,
            (vx_reference)s_clip,
            (vx_reference)s_offset,
            (vx_reference)output,
            (vx_reference)s_maxSize,
            (vx_reference)variance,
        };
        node = createNode(graph, VX_KERNEL_PRIOR_BOX_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
