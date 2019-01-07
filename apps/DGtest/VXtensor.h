#pragma once

#include <vx_ext_amd.h>
#include <vx_amd_nn.h>

/**
 *  Class to store tensor object and its attributes
 */
 
class VXtensor 
{
public:

    VXtensor(vx_context context, vx_size batchSize, std::string fileName, vx_enum usage);
    ~VXtensor();

    /**
     *  Get the tensor object
     */
    vx_tensor getTensor();

    /**
     *  Get the filename of the tensor
     */
    std::string getFileName();

    /**
     *  Reads in the tensor object from the specified .f32 file
     */
    vx_status readTensor();

    /**
     *  Writes out the tensor object to the specified .f32 file
     */
    vx_status writeTensor() ;

private:
    /**
     *  The dimension of the tensor.
     */
    vx_size * mDimension;

    /**
     *  The tensor object.
     */
    vx_tensor mTensor;

    /**
     *  The name of the tensor.
     */
    std::string mFileName;
};