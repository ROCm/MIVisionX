#pragma  once
#include <memory>
#include "reader.h"
#include "decoder.h"
#include "commons.h"
#include "image.h"


enum class LoaderModuleStatus 
{
    OK = 0,
    OCL_BUFFER_SWAP_FAILED,
    OCL_BUFFER_WRITE_FAILED,
    HOST_BUFFER_SWAP_FAILED,
    NO_FILES_TO_READ,
    DECODE_FAILED,
    NO_MORE_DATA_TO_READ,
    UNSUPPORTED_STORAGE_TYPE,
    UNSUPPORTED_DECODER_TYPE,
    INTERNAL_BUFFER_INITIALIZATION_FAILED,
    NOT_INITIALIZED
};

enum class StorageType
{
    FILE_SYSTEM = 0,
    RECORDIO,
    TFRecord,
    LMDB
};

enum class DecoderType 
{
    TURBO_JPEG = 0,//!< Can only decode 
    OPEN_CV = 1,//!< OpenCV can decode compressed images of different type
    OVX_FFMPEG,//!< Uses FFMPEG to decode video streams, can decode up to 4 video streams simultaneously
};


class LoaderModuleConfig 
{
public:	
    LoaderModuleConfig(int batch_size, RaliMemType mem_type):  _batch_size(batch_size),_mem_type(mem_type) {}
    int _batch_size;
    RaliMemType _mem_type;
    virtual StorageType storage_type() = 0;
    virtual DecoderType decoder_type() = 0;
};



/*! \class LoaderModule The interface defining the API and requirements of loader modules*/
class LoaderModule 
{
public:
    virtual LoaderModuleStatus create(LoaderModuleConfig* desc) = 0;
    virtual LoaderModuleStatus set_output_image(Image* output_image) = 0;
    virtual LoaderModuleStatus load_next() = 0;//swapBuffers();
    virtual void reset() = 0; // Resets the loader to load from the beginning of the media
    virtual size_t count() = 0; // Returns the number of available images to be loaded
    virtual ~LoaderModule()= default;
    virtual std::vector<long long unsigned> timing() = 0;// Returns timing info
};

using pLoaderModule = std::shared_ptr<LoaderModule>;