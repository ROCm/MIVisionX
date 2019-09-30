#ifndef MIVISIONX_RALI_API_TYPES_H
#define MIVISIONX_RALI_API_TYPES_H

#include <cstdlib>

#ifndef RALI_API_CALL
#if defined(_WIN32)
#define RALI_API_CALL __stdcall
#else
#define RALI_API_CALL
#endif
#endif

typedef struct FloatParam * RaliFloatParam;
typedef struct IntParam * RaliIntParam;
typedef struct Context* RaliContext;
typedef struct Image* RaliImage;

struct TimingInfo
{
    long long unsigned load_time;
    long long unsigned decode_time;
    long long unsigned process_time;
    long long unsigned transfer_time;
};
enum RaliStatus
{
    RALI_OK = 0,
    RALI_CONTEXT_INVALID,
    RALI_RUNTIME_ERROR,
    RALI_UPDATE_PARAMETER_FAILED,
    RALI_INVALID_PARAMETER_TYPE
};


enum RaliImageColor
{
    RALI_COLOR_RGB24 = 0,
    RALI_COLOR_BGR24 = 1,
    RALI_COLOR_U8  = 2
};

enum RaliProcessMode
{
    RALI_PROCESS_GPU = 0,
    RALI_PROCESS_CPU = 1
};

enum RaliFlipAxis
{
    RALI_FLIP_HORIZONTAL = 0,
    RALI_FLIP_VERTICAL = 1
};

enum RaliImageSizeEvaluationPolicy
{
    RALI_USE_MAX_SIZE = 0,
    RALI_USE_USER_GIVEN_SIZE = 1,
    RALI_USE_MOST_FREQUENT_SIZE = 2,
};

enum RaliTensorLayout
{
    RALI_NHWC = 0,
    RALI_NCHW = 1
};
#endif //MIVISIONX_RALI_API_TYPES_H
