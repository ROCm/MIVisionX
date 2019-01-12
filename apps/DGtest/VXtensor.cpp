#include "VXtensor.h"
#include <stdio.h>
#include <iostream>
#if ENABLE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#endif

#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status     , "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; } }
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); exit(-1); } }

VXtensor::VXtensor(vx_context context, vx_size batchSize, std::string fileName, vx_enum usage) : mFileName(fileName){
    if (usage == VX_READ_ONLY) {
        vx_size dim[4] = {28, 28, 1, batchSize};
        mDimension = dim;
    }
    else if (usage == VX_WRITE_ONLY) {
        vx_size dim[4] = {1, 1, 10, batchSize};
        mDimension = dim;
    }
    else {
        std::cerr << "Wrong Usage" << std::endl;
        exit(-1);
    }
    mTensor = vxCreateTensor(context, 4, mDimension, VX_TYPE_FLOAT32, 0);
    if(vxGetStatus((vx_reference)mTensor)) {
        std::cerr << "ERROR: vxCreateTensor() failed" << std::endl;
        exit(-1);
    }
}

VXtensor::~VXtensor(){
};

vx_tensor VXtensor::getTensor(){
    return mTensor;
}
std::string VXtensor::getFileName() {
    return mFileName;
}
vx_status VXtensor::readTensor()
{
    // access the tensor object
    vx_enum data_type = VX_TYPE_FLOAT32;
    vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];
    vxQueryTensor(mTensor, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
    vxQueryTensor(mTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
    vxQueryTensor(mTensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
    if(data_type != VX_TYPE_FLOAT32) {
        std::cerr << "ERROR: readTensor() supports only VX_TYPE_FLOAT32: invalid for " << mFileName << std::endl;
        return -1;
    }
    vx_size count = dims[0] * dims[1] * dims[2] * dims[3];
    vx_map_id map_id;
    float * ptr;
    vx_status status = vxMapTensorPatch(mTensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for " << mFileName << std::endl;
        return -1;
    }
#if ENABLE_OPENCV
    if(dims[2] == 3 && mFileName.size() > 4 && (mFileName.substr(mFileName.size()-4, 4) == ".png" || mFileName.substr(mFileName.size()-4, 4) == ".jpg"))
    {
        for(size_t n = 0; n < dims[3]; n++) {
            char imgFileName[1024];
            sprintf(imgFileName, mFileName.c_str(), (int)n);
            cv::Mat img = cv::imread(imgFileName, CV_LOAD_IMAGE_COLOR);
            if(!img.data || img.rows != dims[1] || img.cols != dims[0]) {
                printf("ERROR: invalid image or dimelosssions: %s\n", imgFileName);
                return -1;
            }
            for(vx_size y = 0; y < dims[1]; y++) {
                unsigned char * src = img.data + y*dims[0]*3;
                float * dstR = ptr + ((n * stride[3] + y * stride[1]) >> 2);
                float * dstG = dstR + (stride[2] >> 2);
                float * dstB = dstG + (stride[2] >> 2);
                for(vx_size x = 0; x < dims[0]; x++, src += 3) {
                    *dstR++ = src[2];
                    *dstG++ = src[1];
                    *dstB++ = src[0];
                }
            }
        }
    }
#endif
    // open the specified input tensor
    FILE* fp = fopen(mFileName.c_str(), "rb");
    if(!fp) {
        std::cerr << "ERROR: unable to open: " << mFileName << std::endl;
        return -1;
    }
    for(size_t n = 0; n < dims[3]; n++) {
        for(size_t c = 0; c < dims[2]; c++) {
            for(size_t y = 0; y < dims[1]; y++) {
                float * ptrY = ptr + ((n * stride[3] + c * stride[2] + y * stride[1]) >> 2);
                //read into the tensor object
                vx_size n = fread(ptrY, sizeof(float), dims[0], fp);
                if(n != dims[0]) {
                    std::cerr << "ERROR: expected char[" << count*sizeof(float) << "], but got less in " << mFileName << std::endl;
                    return -1;
                }
            }
        }
    }
    fclose(fp);
    status = vxUnmapTensorPatch(mTensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << mFileName << std::endl;
        return -1;
    }
    return 0;
}
    
vx_status VXtensor::writeTensor() {
    // access the tensor object
    vx_enum data_type = VX_TYPE_FLOAT32;
    vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];
    vxQueryTensor(mTensor, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
    vxQueryTensor(mTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
    vxQueryTensor(mTensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
    if(data_type != VX_TYPE_FLOAT32) {
        std::cerr << "ERROR: readTensor() supports only VX_TYPE_FLOAT32: invalid for " << mFileName << std::endl;
        return -1;
    }
    vx_size count = dims[0] * dims[1] * dims[2] * dims[3];
    vx_map_id map_id;
    float * ptr;
    vx_status status = vxMapTensorPatch(mTensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for " << mFileName << std::endl;
        return -1;
    }
    //open the specified output tensor
    FILE * fp = fopen(mFileName.c_str(), "wb");
    if(!fp) {
        std::cerr << "ERROR: unable to open: " << mFileName << std::endl;
        return -1;
    }
    //write out to the specified tensor
    fwrite(ptr, sizeof(float), count, fp);
    fclose(fp);

    status = vxUnmapTensorPatch(mTensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << mFileName << std::endl;
        return -1;
    }
    return 0;
}