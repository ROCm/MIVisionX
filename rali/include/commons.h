/*
 * dtypes.h
 *
 *  Created on: Jan 29, 2019
 *      Author: root
 */

#pragma once
#include "exception.h"
#include "log.h"


enum class RaliTensorFormat
{
    NHWC = 0,
    NCHW
};
enum class RaliAffinity
{
    GPU = 0,
    CPU
};

/*! \brief Color formats currently supported by Rali SDK as input/output
 *
 */
enum class RaliColorFormat 
{
    RGB24 = 0,
    BGR24,
    U8
};

/*! \brief Memory type, host or device
 * 
 *  Currently supports HOST and OCL, will support HIP in future
 */
enum class RaliMemType 
{
    HOST = 0,
    OCL
};

enum class ImageBufferAllocation
{
    external = 0,
    none = 1
};
