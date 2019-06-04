#include "kernels.h"
#include <float.h>
#include <string.h>
#include <map>
#include <vector>
#include <utility>
#include <algorithm>
#include <assert.h>
#include <vx_ext_amd.h>
using namespace std;

struct NormalizedBBox
{
    float xmin, ymin,xmax,ymax;
    bool has_size;
};

static vx_status VX_CALLBACK validate(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dims.
    vx_enum type;
    vx_size num_dims;
    vx_size input_dims[4],  output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type!= VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type!= VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type!= VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    vx_enum scalar_type;
    vx_int32 num_classes;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &num_classes, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(num_classes < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #4 scalar type=%d (num classes must be greater than 0)\n", num_classes);

    vx_int32 share_location;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &share_location, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(share_location < 0 || share_location > 1) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #5 scalar type=%d (must be 1(true)/0(false))\n", share_location);

    vx_int32 background_label_id;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &background_label_id, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(background_label_id < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #6 scalar type=%d (must be greater than 0)\n", background_label_id);

    vx_float32 nms_threshold;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[6], &nms_threshold, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(nms_threshold < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #7 scalar type=%f (must be greater than 0)\n", nms_threshold);

    if(parameters[7])
    {
        vx_int32 top_k;
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
        if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[7], &top_k, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        if(top_k < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #8 scalar type=%d (must be greater than 0)\n", top_k);    
    }
    

    char* code_type; 
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_STRING_AMD) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[8], &code_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    vx_int32 keep_top_k;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[9], &keep_top_k, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(keep_top_k < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #10 scalar type=%d (must be greater than 0)\n", keep_top_k);

    if(parameters[10])
    {
        vx_float32 confidence_threshold;
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[10], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
        if(scalar_type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[10], &confidence_threshold, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        if(confidence_threshold < 0) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #11 scalar type=%f (must be greater than 0)\n", confidence_threshold);        
    }

    vx_int32 variance_encoded_in_target;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[11], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[11], &variance_encoded_in_target, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(variance_encoded_in_target < 0 || variance_encoded_in_target > 1) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #12 scalar type=%d (must be 1(true)/0(false))\n", variance_encoded_in_target);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[12], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[12], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[12], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // output tensor configuration
    type = VX_TYPE_FLOAT32;
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[12], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[12], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[12], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    

    printf("DEBUG: Validate Success!!!\n");
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

static void GetConfidenceScores(float* confData, const int num, const int numPriors, const int num_classes, vector<map<int, vector<float> > >* allConfidenceScores)
{
    allConfidenceScores->resize(num);
    for (int i = 0; i < num; i++)
    {
        map<int, vector<float> >&label_scores = (*allConfidenceScores)[i];
        for(int p = 0; p < numPriors; p++)
        {
           
            int start_index = p * num_classes;
            for(int c = 0; c < num_classes; c++)
            {
                printf("DEBUG VALUE: %d , %f   ", start_index+c,  confData[start_index+c]);
                label_scores[c].push_back(confData[start_index+c]);
            }
            
        }
        confData += num_classes * numPriors;
    }
}

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}


void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k, std::vector<std::pair<float, int> >& score_index_vec)
{
    for(int i = 0; i < scores.size(); i++)
    {
        if(scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }

    std::stable_sort(score_index_vec.begin() , score_index_vec.end(), SortScorePairDescend<int>);

    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

static float BBoxSize(const NormalizedBBox& bbox, bool normalized)
{
    if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin)
    {
        return 0; 
    }
    else
    {
        if (bbox.has_size == true)
        {
            return bbox.has_size;
        }
        else
        {
            float width = bbox.xmax - bbox.xmin;
            float height = bbox.ymax - bbox.ymin;
            if (normalized)
            {
                return width * height;
            }
            else
            {
                // If bbox is not within range [0, 1].
                return (width + 1) * (height + 1);
            }
        }
    }
}


static float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2, const bool normalized)
{
    NormalizedBBox intersect_bbox;
    intersect_bbox.xmin = std::max(bbox1.xmin, bbox2.xmin);
    intersect_bbox.ymin = std::max(bbox1.ymin, bbox2.ymin);
    intersect_bbox.xmax = std::min(bbox1.xmax, bbox2.xmax);
    intersect_bbox.ymax = std::min(bbox1.ymax, bbox2.ymax);

    float intersect_size = BBoxSize(intersect_bbox, normalized);
    if (intersect_size > 0)
    {
        float bbox1_size = BBoxSize(bbox1, normalized);
        float bbox2_size = BBoxSize(bbox2, normalized);
        return intersect_size / (bbox1_size + bbox2_size - intersect_size);
    }
    else
    {
        return 0.;
    }
}

void ApplyNMSFast(const vector<NormalizedBBox>& bboxes, const vector<float>& scores, const float score_threshold, const float nms_threshold, const int top_k, vector<int>* indices)
{

    assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    float adaptive_threshold = nms_threshold;
    indices->clear();
    //printf("len of vec = %lu\n", score_index_vec.size());

    for(int i = 0; i < score_index_vec.size(); i++)
    {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for(int k = 0; k < (int)indices->size() && keep; k++)
        {
            const int kept_idx = (*indices)[k];
            float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx], false);
            keep = overlap <= adaptive_threshold;
        }

        if (keep)
            indices->push_back(idx);
        printf("indices%d\n", (*indices)[0]);
    }
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
    printf("DEBUG: In codegen\n");
    //get tensor dimensions
    vx_size input_dims_0[4], input_dims_1[4], input_dims_2[4], output_dims[4];
    vx_size num_of_dims;
    vx_enum type;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_0, sizeof(input_dims_0)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, input_dims_2, sizeof(input_dims_2)));
    
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[12], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[12], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

    vx_int32 num_classes, share_location, background_label_id, top_k, keep_top_k, variance_encoded_in_target;
    vx_float32 nms_threshold, confidence_threshold;
    //vx_scalar code_type;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &num_classes, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &share_location, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &background_label_id, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[6], &nms_threshold, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(parameters[7])
    {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[7], &top_k, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    else
    {
        top_k = -1;
    }

    char* code_type = new char[15]; 
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[8], code_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    /*if((strcmp(code_type, "CENTER_SIZE") != 0) && (strcmp(code_type, "CORNER") != 0))
        return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: detection_output: #9 code type=%s \n", code_type);
    */
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[9], &keep_top_k, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(parameters[10])
    {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[10], &confidence_threshold, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));    
    }
    else
    {
        confidence_threshold = -FLT_MAX;
    }
    
    if(parameters[11])
    {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[11], &variance_encoded_in_target, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    else
    {
        variance_encoded_in_target = 0;
    }

    vx_int32 num_loc_classes = share_location ? 1:num_classes;
    vx_int32 num_priors = input_dims_2[2] / 4;
    if ((num_priors * num_loc_classes * 4) != input_dims_0[1])
    {
        printf("codegen: detection_output: Number of priors must match number of location predictions\n");
        exit(0);
    }            
    if((num_priors * num_classes) != input_dims_1[1])   
    {
        printf("codegen: detection_output: Number of priors must match number of confidence predictions\n");  
        exit(0);
    }             
    

    int num_batches = input_dims_1[0];
    int numPriors = input_dims_2[2]/4;

    vx_map_id map_id;
    vx_size stride[4]= {4,input_dims_1[0]*4,input_dims_1[0]*input_dims_1[1]*4,input_dims_1[0]*input_dims_1[1]*input_dims_1[2]*4};
    float * ptr;
    vx_enum usage = VX_READ_ONLY;
    vx_status status;
    vx_size count_tensor = input_dims_1[0]*input_dims_1[1]*input_dims_1[2]*input_dims_1[3];
    
    float *confData = new float[count_tensor];

    /*status = vxMapTensorPatch((vx_tensor)parameters[1], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for "  << std::endl;
        return -1;
    }

    printf("value at ptr = %f\n", ptr[10]);
    memcpy(confData, ptr, (count_tensor*sizeof(float)));

    status = vxUnmapTensorPatch((vx_tensor)parameters[1], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for "  << std::endl;
        return -1;
    }
    */
    //for(int i = 0; i < input_dims_1[0]*input_dims_1[1]*input_dims_1[2]*input_dims_1[3]; i++)
    //    printf(" %f  ", confData[i]);

    status = vxCopyTensorPatch((vx_tensor)parameters[1], num_of_dims, nullptr, nullptr, stride, confData, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
    {
        std::cerr << "ERROR: vxCopyTensorPatch() failed for input parameters[1] = "  << status << std::endl; 
        return -1;
    }

    

    vector<map<int, vector<float> > > allConfidenceScores;
    GetConfidenceScores(confData, num_batches, numPriors, num_classes, &allConfidenceScores);

    strcpy(opencl_kernel_function_name, "detection_output");

    opencl_work_dim = 1;
    //opencl_global_work[0] = input_dims_1[0];
    opencl_global_work[0] = input_dims_0[0]*input_dims_0[1]*input_dims_0[2]*input_dims_0[3];
    //opencl_global_work[1] = input_dims_2[2]/4;
    //opencl_global_work[0] = output_dims[2] * output_dims[3];

    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    if (num_of_dims == 4) {
        char item[8192];
        if(strcmp(code_type ,"CENTER_SIZE") == 0)
        {
            printf("in here 1\n");
            sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in_0, uint in_0_offset, uint4 in_0_stride, __global uchar * in_1, uint in_1_offset, uint4 in_1_stride, __global uchar * in_2, uint in_2_offset, uint4 in_2_stride, "\
                "                 const int s_num_classes, const int s_share_location, const int s_background_label_id, const float s_nms_threshold, const int s_top_k, const int code_type, const int s_keep_top_k, " \
                "                 const float s_confidence_threshold , const int s_variance_encoded_in_target, __global uchar * out, uint out_offset, uint4 out_stride)\n"
                "{ \n" 
                "   const int num_classes = %d; \n"
                "   const int share_location = %d; \n"
                "   const int background_label_id = %d; \n"
                "   const float nms_threshold = %f; \n"
                "   const int top_k = %d; \n"
                "   const int keep_top_k = %d; \n"
                "   const float confidence_threshold = %f; \n"
                "   const int variance_encoded_in_target = %d; \n"
                "   int num = get_global_size(0); \n"
                "   int numPriors = %d; \n"
                "   int num_loc_classes = share_location ? 1 : num_classes; \n"
                "   float bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax; \n"
                "   int index = get_global_id(0); \n"
                "   int i = (index) %% 4; \n"
                //"   i *= sizeof(float); \n"
                "   int p = (((index) / 4 / num_loc_classes) %% numPriors) * 4; \n"
                //"   p *= sizeof(float); \n"
                "   int c = ((index) / 4) %% num_loc_classes; \n"
                //"   c *= sizeof(float); \n"
                "   int label = share_location ? -1 : c; \n"

                "   if(label == background_label_id) \n"
                "   { \n"
                "       return; \n"
                "   } \n"
                "   const int startIndex = numPriors * 4 + p; \n"
                "   __global uchar* prior_variance = in_2 + startIndex*sizeof(float); \n"

                "   bbox_xmin = variance_encoded_in_target ? (*(__global float *)&in_0[index - i + 0*sizeof(float)]) : ((__global float *)(prior_variance))[0] * (*(__global float *)&in_0[index - i + 0*sizeof(float)]); \n"
                "   bbox_ymin = variance_encoded_in_target ? (*(__global float *)&in_0[index - i + 1*sizeof(float)]) : ((__global float *)(prior_variance))[1] * (*(__global float *)&in_0[index - i + 1*sizeof(float)]); \n"
                "   bbox_xmax = variance_encoded_in_target ? (*(__global float *)&in_0[index - i + 2*sizeof(float)]) : ((__global float *)(prior_variance))[2] * (*(__global float *)&in_0[index - i + 2*sizeof(float)]); \n"
                "   bbox_ymax = variance_encoded_in_target ? (*(__global float *)&in_0[index - i + 3*sizeof(float)]) : ((__global float *)(prior_variance))[3] * (*(__global float *)&in_0[index - i + 3*sizeof(float)]); \n"
                "   float val; \n"

                "   float prior_width = (*(__global float *)&in_2[p+2*sizeof(float)]) - (*(__global float *)&in_2[p+0*sizeof(float)]); \n"
                "   float prior_height = (*(__global float *)&in_2[p+3*sizeof(float)]) - (*(__global float *)&in_2[p+1*sizeof(float)]); \n"
                "   float prior_center_x = ((*(__global float *)&in_2[p+0*sizeof(float)]) + (*(__global float *)&in_2[p+2*sizeof(float)]))*0.5; \n"
                "   float prior_center_y = ((*(__global float *)&in_2[p+1*sizeof(float)]) + (*(__global float *)&in_2[p+3*sizeof(float)]))*0.5; \n"

                "   float decode_bbox_center_x = bbox_xmin * prior_width + prior_center_x; \n"
                "   float decode_bbox_center_y = bbox_ymin * prior_height + prior_center_y; \n"
                "   float decode_bbox_width = exp(bbox_xmax) * prior_width; \n"
                "   float decode_bbox_height = exp(bbox_ymax) * prior_height; \n"
        
                "   switch(i) \n"
                "   { \n"
                "       case 0:  \n"
                "           val = decode_bbox_center_x - decode_bbox_width * 0.5; \n"
                "           break; \n"
                "       case 1: \n"
                "           val = decode_bbox_center_y - decode_bbox_height * 0.5; \n"
                "           break; \n"
                "       case 2: \n"
                "           val = decode_bbox_center_x + decode_bbox_width * 0.5; \n"
                "           break; \n"
                "       case 3: \n"
                "           val = decode_bbox_center_y + decode_bbox_height * 0.5; \n"
                "           break; \n"
                "   }   \n"  
        
                "   out += index*sizeof(float); \n"
                "   *(__global float *)&out[0] = val; \n"
                "}\n", opencl_kernel_function_name, num_classes, share_location, background_label_id, nms_threshold, top_k, keep_top_k, confidence_threshold, variance_encoded_in_target, numPriors);
            opencl_kernel_code = item;
        }
        else if (strcmp(code_type ,"CORNER") == 0)
        {
            sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                             "__kernel void %s(__global uchar * in_0, uint in_0_offset, uint4 in_0_stride, __global uchar * in_1, uint in_1_offset, uint4 in_1_stride, __global uchar * in_2, uint in_2_offset, uint4 in_2_stride, "\
                "                 const int s_num_classes, const int s_share_location, const int s_background_label_id, const float s_nms_threshold, const int s_top_k, const int code_type, const int s_keep_top_k, " \
                "                 const float s_confidence_threshold , const int s_variance_encoded_in_target, __global uchar * out, uint out_offset, uint4 out_stride)\n"
                "{ \n" 
                "   const int num_classes = %d; \n"
                "   const int share_location = %d; \n"
                "   const int background_label_id = %d; \n"
                "   const float nms_threshold = %f; \n"
                "   const int top_k = %d; \n"
                "   const int keep_top_k = %d; \n"
                "   const float confidence_threshold = %f; \n"
                "   const int variance_encoded_in_target = %d; \n"
                "   int num = get_global_size(0); \n"
                "   int numPriors = %d; \n"
                "   int num_loc_classes = share_location ? 1 : num_classes; \n"
                "   float bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax; \n"
                "   int index = get_global_id(0); \n"
                "   int i = (index) %% 4; \n"
                //"   i *= sizeof(float); \n"
                "   int p = (((index) / 4 / num_loc_classes) %% numPriors) * 4; \n"
                //"   p *= sizeof(float); \n"
                "   int c = ((index) / 4) %% num_loc_classes; \n"
                //"   c *= sizeof(float); \n"
                "   int label = share_location ? -1 : c; \n"

                "   if(label == background_label_id) \n"
                "   { \n"
                "       return; \n"
                "   } \n"
                "   const int startIndex = numPriors * 4 + p; \n"
                "   __global uchar* prior_variance = in_2 + startIndex*sizeof(float); \n"

                "   bbox_xmin = variance_encoded_in_target ? (*(__global float *)&in_0[index - i + 0*sizeof(float)]) : ((__global float *)(prior_variance))[0] * (*(__global float *)&in_0[index - i + 0*sizeof(float)]); \n"
                "   bbox_ymin = variance_encoded_in_target ? (*(__global float *)&in_0[index - i + 1*sizeof(float)]) : ((__global float *)(prior_variance))[1] * (*(__global float *)&in_0[index - i + 1*sizeof(float)]); \n"
                "   bbox_xmax = variance_encoded_in_target ? (*(__global float *)&in_0[index - i + 2*sizeof(float)]) : ((__global float *)(prior_variance))[2] * (*(__global float *)&in_0[index - i + 2*sizeof(float)]); \n"
                "   bbox_ymax = variance_encoded_in_target ? (*(__global float *)&in_0[index - i + 3*sizeof(float)]) : ((__global float *)(prior_variance))[3] * (*(__global float *)&in_0[index - i + 3*sizeof(float)]); \n"
                "   float val; \n"

                "   switch(i) \n"
                "   { \n"
                "       case 0:  \n"
                "           val = (*(__global float *)&in_2[p+0*sizeof(float)]) + bbox_xmin; \n"
                "           break; \n"
                "       case 1: \n"
                "           val = (*(__global float *)&in_2[p+1*sizeof(float)]) + bbox_ymin; \n"
                "           break; \n"
                "       case 2: \n"
                "           val = (*(__global float *)&in_2[p+2*sizeof(float)]) + bbox_xmax; \n"
                "           break; \n"
                "       case 3: \n"
                "           val = (*(__global float *)&in_2[p+3*sizeof(float)]) + bbox_ymax; \n"
                "           break; \n"
                "   }   \n"  
        
                "   out += index*sizeof(float); \n"
                "   *(__global float *)&out[0] = val; \n"
                "}\n", opencl_kernel_function_name, num_classes, share_location, background_label_id, nms_threshold, top_k, keep_top_k, confidence_threshold, variance_encoded_in_target, numPriors);
            opencl_kernel_code = item;
        }
        
    }
    
    
    vx_size count_output_opencl = output_dims[0]*output_dims[1]*output_dims[2]*output_dims[3];
    float *decode_data = new float[count_output_opencl];
    vx_size stride_output_opencl[4] = {4, output_dims[0]*4, output_dims[0]*output_dims[1]*4, output_dims[0]*output_dims[1]*output_dims[2]*4};

    //printf("%lu  %lu %lu %lu  %lu\n", stride_output_opencl[0], stride_output_opencl[1], stride_output_opencl[2], stride_output_opencl[3],count_output_opencl);
    status = vxCopyTensorPatch((vx_tensor)parameters[12], num_of_dims, nullptr, nullptr, stride_output_opencl, decode_data, usage, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
    {
        std::cerr << "ERROR: vxCopyTensorPatch() failed for output opencl = "  << status << std::endl; 
        return -1;
    }
    
    /*for(int i= 0 ; i < count_output_opencl; i++)
    {
        printf("  %f  ", decode_data[i]);
    }
    
    status = vxMapTensorPatch((vx_tensor)parameters[12], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for decode_data"  << std::endl;
        return -1;
    }

    memcpy(decode_data, ptr, (count_output*sizeof(float)));
    
    status = vxUnmapTensorPatch((vx_tensor)parameters[12], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for decode_data"  << std::endl;
        return -1;
    }
    
    */
    printf("\nDEBUG: HEre 2\n");
    typedef std::map<int, std::vector<NormalizedBBox> > LabelBBox;
    std::vector<LabelBBox> allDecodedBBoxes;
    allDecodedBBoxes.clear();
    allDecodedBBoxes.resize(num_batches);

    for(int i = 0; i < num_batches; i++)
    {
        LabelBBox& decode_bboxes = allDecodedBBoxes[i];
        for (int c = 0; c < num_loc_classes; ++c)
        {
            int label = share_location ? -1 : c;
            decode_bboxes[label].resize(numPriors);
            for(int p = 0; p < numPriors; p++)
            {
                int startIndex = p * num_loc_classes * 4;
                //printf("startIndex , c == %d %d \n",startIndex, c );
                NormalizedBBox bbox = decode_bboxes[label][p];
                bbox.xmin = decode_data[startIndex + c * 4];
                bbox.ymin = decode_data[startIndex + c * 4 + 1];
                bbox.xmax = decode_data[startIndex + c * 4 + 2];
                bbox.ymax = decode_data[startIndex + c * 4 + 3];
                //printf("%f %f %f %f\n", decode_data[startIndex + c * 4], decode_data[startIndex + c * 4 +1], decode_data[startIndex + c * 4 + 2], decode_data[startIndex + c * 4+3]);
            } 

        }

    }   
      
    printf("\nDEBUG: HEre 3\n");
    
    size_t numKept = 0;
    std::vector<std::map<int, std::vector<int> > > allIndices;
    for (int i = 0; i < num_batches; i++)
    {
        LabelBBox decode_bboxes = allDecodedBBoxes[i];
        std::map<int, std::vector<float> > confidenceScores = allConfidenceScores[i];
        std::map<int, std::vector<int> > indices;
        int num_det = 0;
        for(int c = 0; c < num_classes; c++)
        {
            if(c == background_label_id)
                continue;
            if(confidenceScores.find(c) == confidenceScores.end())
                ERRMSG(VX_ERROR_INVALID_VALUE, "codegen: could not find confidence predictions for label %d\n", c);

            const vector<float>& scores = confidenceScores.find(c)->second;
            int label = share_location ? -1 : c;
            if (decode_bboxes.find(label) == decode_bboxes.end()) {
                ERRMSG(VX_ERROR_INVALID_VALUE, "codegen: could not find location predictions for label %d\n", label);
            }
            const vector<NormalizedBBox> bboxes = decode_bboxes.find(label)->second;
            ApplyNMSFast(bboxes, scores, confidence_threshold, nms_threshold, top_k, &(indices[c]));

            num_det += indices[c].size();   
            //printf("num_det = %d\n", num_det );   
        }

        if (keep_top_k > -1 && num_det > keep_top_k)
        {
            std::vector<std::pair<float, std::pair<int, int> > > scoreIndexPairs;
            for (std::map<int, std::vector<int> >::iterator it = indices.begin(); it != indices.end(); ++it)
            {
                int label = it->first;
                const std::vector<int>& labelIndices = it->second;
                if (confidenceScores.find(label) == confidenceScores.end())
                {
                    ERRMSG(VX_ERROR_INVALID_VALUE, "codegen: could not find location predictions for label %d", label);
                    continue;
                }   
                const std::vector<float>& scores = confidenceScores.find(label)->second;
                for (size_t j = 0; j < labelIndices.size(); ++j)
                {
                    size_t idx = labelIndices[j];
                    assert(idx < scores.size());
                    scoreIndexPairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
                }
            }

            std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(), SortScorePairDescend<pair<int, int> >);
            scoreIndexPairs.resize(keep_top_k);
            // Store the new indices.
            map<int, vector<int> > newIndices;
            for (int j = 0; j < scoreIndexPairs.size(); ++j)
            {
                int label = scoreIndexPairs[j].second.first;
                int idx = scoreIndexPairs[j].second.second;
                newIndices[label].push_back(idx);
            }
            allIndices.push_back(newIndices);
            numKept += keep_top_k;
        } 
        else
        {
            allIndices.push_back(indices);
            numKept += num_det;
        }
    }
    printf("numkept = %lu\n", numKept);
    printf("DEBUG: HEre 4\n");
    
    vx_size outputShape[4] = {1, 1, (vx_size)numKept, 7};
    vx_size count_output_final = outputShape[0]*outputShape[1]*outputShape[2]*outputShape[3];
    vx_size stride_output_final[4] = {4, outputShape[0]*4, outputShape[0]*outputShape[1]*4, outputShape[0]*outputShape[1]*outputShape[2]*4};    
    float * outputData = new float[count_output_final];
    if(numKept == 0)
    {
        output_dims[2] = num_batches;
        for(int i = 0; i < num_batches; i++)
        {
            outputData[0] = i;
            outputData += 7;
        }
        return VX_SUCCESS;
    }
    printf("DEBUG: HEre 5\n");
    int count = 0;
    for (int i = 0; i < num_batches; i++)
    {
        const map<int, vector<float> >& confScores = allConfidenceScores[i];
        const LabelBBox& decodeBBoxes = allDecodedBBoxes[i];

        for (map<int, vector<int> >::iterator it = allIndices[i].begin(); it != allIndices[i].end(); ++it)
        {
            int label = it->first;
            if (confScores.find(label) == confScores.end()) 
            {
                ERRMSG(VX_ERROR_INVALID_VALUE,"Could not find confidence predictions for %d\n",label);
                continue;
            }

            const vector<float>& scores = confScores.find(label)->second;
            int locLabel = share_location ? -1 : label;
            if (decodeBBoxes.find(locLabel) == decodeBBoxes.end())
            {
                ERRMSG(VX_ERROR_INVALID_VALUE,"Could not find location predictions for %d\n",locLabel);
                continue;
            }

            const vector<NormalizedBBox>& bboxes = decodeBBoxes.find(locLabel)->second;
            vector<int>& indices = it->second;

            for (int j = 0; j < indices.size(); ++j) 
            {
                int idx = indices[j];
                outputData[count * 7] = i;
                outputData[count * 7 + 1] = label;
                outputData[count * 7 + 2] = scores[idx];
                const NormalizedBBox& bbox = bboxes[idx];
                outputData[count * 7 + 3] = bbox.xmin;
                outputData[count * 7 + 4] = bbox.ymin;
                outputData[count * 7 + 5] = bbox.xmax;
                outputData[count * 7 + 6] = bbox.ymax;
                count++;
            }
        }           
    }

    printf("%f %f %f %f %f %f %f\n", outputData[0],outputData[1],outputData[2],outputData[3],outputData[4], outputData[5],outputData[6]);

    //vxCopyTensorPatch((vx_tensor)parameters[12], num_of_dims, 0, (const vx_size *)outputShape, stride, outputData, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

   
    //printf("%lu  %lu %lu %lu  %lu\n", stride_output_final[0], stride_output_final[1], stride_output_final[2], stride_output_final[3],count_output_final);
    status = vxCopyTensorPatch((vx_tensor)parameters[12], num_of_dims, nullptr, nullptr, stride_output_final, decode_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
    {
        std::cerr << "ERROR: vxCopyTensorPatch() failed for final output = "  << status << std::endl; 
        return -1;
    }
    /*status = vxMapTensorPatch((vx_tensor)parameters[12], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for "  << std::endl;
        return -1;
    }

    memcpy((vx_tensor)parameters[12], ptr, (outputShape[0]*outputShape[1]*outputShape[2]*outputShape[3]*sizeof(float)));
    
    status = vxUnmapTensorPatch((vx_tensor)parameters[12], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for "  << std::endl;
        return -1;
    }
    */
    printf("DEBUG: OpenCL codegen Success!!!\n");
    delete [] confData;
    return VX_SUCCESS;
}


//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel publisher.
vx_status publishDetectionOutputLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.detection_output", VX_KERNEL_DETECTION_OUTPUT_LAYER_AMD, host_kernel, 13, validate, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 11, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 12, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    printf("DEBUG: Publish Success!!!\n");
    return VX_SUCCESS;
}


VX_API_ENTRY vx_node VX_API_CALL vxDetectionOutputLayer(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_tensor input3, vx_int32 num_classes, vx_int32 share_location, vx_int32 background_label_id, vx_float32 nms_threshold,
                                                        vx_int32 top_k, vx_scalar code_type, vx_int32 keep_top_k, vx_float32 confidence_threshold, vx_int32 variance_encoded_in_target, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_num_classes = vxCreateScalarWithSize(context, VX_TYPE_INT32, &s_num_classes, sizeof(s_num_classes));
        vx_scalar s_share_location = vxCreateScalarWithSize(context, VX_TYPE_INT32, &s_share_location, sizeof(s_share_location));
        vx_scalar s_background_label_id = vxCreateScalarWithSize(context, VX_TYPE_INT32, &s_background_label_id, sizeof(s_background_label_id));
        vx_scalar s_nms_threshold = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &s_nms_threshold, sizeof(s_nms_threshold));
        vx_scalar s_top_k = vxCreateScalarWithSize(context, VX_TYPE_INT32, &s_top_k, sizeof(s_top_k));
        vx_scalar s_code_type = vxCreateScalarWithSize(context, VX_TYPE_STRING_AMD, &s_code_type, sizeof(s_code_type));
        vx_scalar s_keep_top_k = vxCreateScalarWithSize(context, VX_TYPE_INT32, &s_keep_top_k, sizeof(s_keep_top_k));
        vx_scalar s_confidence_threshold = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &s_confidence_threshold, sizeof(s_confidence_threshold));
        vx_scalar s_variance_encoded_in_target = vxCreateScalarWithSize(context, VX_TYPE_INT32, &s_variance_encoded_in_target, sizeof(s_variance_encoded_in_target));


        vx_reference params[] = {
            (vx_reference)input1,
            (vx_reference)input2,
            (vx_reference)input3,
            (vx_reference)s_num_classes,
            (vx_reference)s_share_location,
            (vx_reference)s_background_label_id,
            (vx_reference)s_nms_threshold,
            (vx_reference)s_top_k,
            (vx_reference)s_code_type,
            (vx_reference)s_keep_top_k,
            (vx_reference)s_confidence_threshold,
            (vx_reference)s_variance_encoded_in_target,
            (vx_reference)output,
        };
        node = createNode(graph, VX_KERNEL_DETECTION_OUTPUT_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    printf("DEBUG: Layer Entry Success!!!\n");
    return node;
}
