/*
 * dtypes.h
 *
 *  Created on: Jan 29, 2019
 *      Author: root
 */

#pragma once
#include <vector>
#include "exception.h"
#include "log.h"


enum class RaliTensorFormat
{
    NHWC = 0,
    NCHW
};
enum class RaliTensorDataType
{
    FP32 = 0,
    FP16
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

enum class MetaDataType
{
    Label,
    BoundingBox
};

struct MetaDataConfig
{
private:
    MetaDataType _type;
    std::string _path;
public:
    MetaDataConfig(const MetaDataType& type, const std::string& path ):_type(type), _path(path){}
    MetaDataConfig() = delete;
    MetaDataType type() const { return _type; }
    std::string path() const { return  _path; }
};

struct Timing
{
    // The following timings are accumulated timing not just the most recent activity
    long long unsigned image_read_time= 0;
    long long unsigned image_decode_time= 0;
    long long unsigned to_device_xfer_time= 0;
    long long unsigned from_device_xfer_time= 0;
    long long unsigned copy_to_output = 0;
    long long unsigned image_process_time= 0;
    long long unsigned bb_process_time= 0;
    long long unsigned mask_process_time= 0;
    long long unsigned label_load_time= 0;
    long long unsigned bb_load_time= 0;
    long long unsigned mask_load_time = 0;
};