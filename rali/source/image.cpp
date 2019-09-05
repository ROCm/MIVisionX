#include <cstdio>
#include <CL/cl.h>
#include <stdexcept>
#include <vx_ext_amd.h>
#include <cstring>
#include "commons.h"
#include "image.h"

vx_enum vx_mem_type(RaliMemType mem) 
{
    switch(mem)
    {
        case RaliMemType::OCL:
        {
            return VX_MEMORY_TYPE_OPENCL;
        }
        break;
        case RaliMemType::HOST:
        {
            return VX_MEMORY_TYPE_HOST;
        }
        break;
        default:
            throw std::runtime_error("Memory type not valid");
    }
    
} 
    
ImageInfo::ImageInfo():
        _width(0),
        _height(0),
        color_planes(1),
        batch_size(1),
        data_size(0),
        mem_type(RaliMemType::HOST),
        _type(Type::UNKNOWN),
        color_fmt(RaliColorFormat::U8){}

ImageInfo::ImageInfo(
    unsigned width_,
    unsigned height_,
    unsigned batches,
    unsigned planes,
    RaliMemType mem_type_, 
    RaliColorFormat col_fmt_):
        _width(width_),
        _height(height_),
        color_planes(planes),
        batch_size(batches),
        data_size(width_* height_ * batch_size * planes),
        mem_type(mem_type_),
        _type(Type::UNKNOWN),
        color_fmt(col_fmt_) {}

bool ImageInfo::operator==(const ImageInfo& other)
{
    return (width() == other._width &&
            height_batch() == other._height &&
            mem_type == other.mem_type && 
            color_fmt == other.color_fmt && 
            color_planes == other.color_planes);
}    

//Image::Image() {}


Image::~Image()
{  
    vxReleaseImage(&img);

    if(buf == nullptr) 
        return;

    if(_info.mem_type == RaliMemType::OCL) 
    {    
        clReleaseMemObject((cl_mem)buf);
    } 
    else 
    {
        delete[] (float*)(buf);
    }

    buf = nullptr;
}

//! Converts the Rali color format type to OpenVX
/*!
 * For OpenVX there is no BGR color format supported.
 * The input images loaded in GBR format should be
 * reordered before passed to OpenVX, and match RGB.
 */
vx_df_image interpret_color_fmt(RaliColorFormat color_format) 
{
    switch(color_format){   

        case RaliColorFormat::RGB24:
        case RaliColorFormat::BGR24:
            return VX_DF_IMAGE_RGB;    

        case RaliColorFormat::U8:
            return VX_DF_IMAGE_U8;

        default:
            THROW("Unsupported Image type "+ TOSTR(color_format))
    }
}

Image::Image(const ImageInfo& img_info):_info(img_info)
{
    _info._type = ImageInfo::Type::UNKNOWN;
}

int Image::create_virtual(vx_context context, vx_graph graph)
{
    if(img != 0)
        return 0;
    // create a virtual image as the output image for this node
    img = vxCreateVirtualImage(graph, _info.width(), _info.height_batch(), VX_DF_IMAGE_VIRT);
    vx_status status;
    if((status = vxGetStatus((vx_reference)img)) != VX_SUCCESS)
        THROW("Error: vxCreateVirtualImage(input:[" + TOSTR(_info.width()) + "x" + TOSTR(_info.height_batch()) + "]): failed " + TOSTR(status))

    _info._type = ImageInfo::Type::VIRTUAL;
    return 0;                                             
}

int Image::create_from_handle(vx_context context, ImageBufferAllocation policy)
{
    if(img != 0)
        return 0;

    // TODO: the pointer passed here changes if the number of planes are more than one
    vx_imagepatch_addressing_t addr_in = { 0 };
    void *ptr[1] = { nullptr };

    addr_in.step_x = 1;
    addr_in.step_y = 1;
    addr_in.scale_x = VX_SCALE_UNITY;
    addr_in.scale_y = VX_SCALE_UNITY;
    addr_in.dim_x = _info.width();
    addr_in.dim_y = _info.height_batch();

    vx_uint32 alignpixels = 32;

    addr_in.stride_x = _info.color_planes;

    if (alignpixels == 0)
        addr_in.stride_y = addr_in.dim_x *addr_in.stride_x;
    else
        addr_in.stride_y = ((addr_in.dim_x + alignpixels - 1) & ~(alignpixels - 1))*addr_in.stride_x;

    if(_info.height_batch() == 0 || _info.width() == 0 || _info.color_planes == 0)
        THROW("Invalid image dimension " + TOSTR(_info.height_batch()) + " x " + TOSTR(_info.width()) + " x " + TOSTR(_info.color_planes));

    vx_status status;
    vx_size size = (addr_in.dim_y+0) * (addr_in.stride_y+0);
    if(policy == ImageBufferAllocation::external)
    {
        if(_info.mem_type == RaliMemType::OCL) 
        {
            cl_context opencl_context = nullptr;
            // allocate opencl buffer with required dim
            status = vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &opencl_context, sizeof(opencl_context));
            if (status != VX_SUCCESS)
                THROW("vxQueryContext of failed "+TOSTR( status));


            cl_int ret = CL_SUCCESS;
            cl_mem clImg = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, size, NULL, &ret);
            if (!clImg || ret)
            {
                if(ret == CL_INVALID_BUFFER_SIZE)
                    ERR("Requested"+TOSTR(size)+"bytes which is more than max allocation on the device");
                THROW("clCreateBuffer of size "+TOSTR(size)+"failed "+ TOSTR(ret));
            }
            ptr[0] = clImg;

        } 
        else
        {
            unsigned char* hostImage = new unsigned char[size];
            ptr[0] = hostImage;
        }
        buf = ptr[0];
    }
    vx_df_image vx_color_format = interpret_color_fmt(_info.color_fmt);
    img = vxCreateImageFromHandle(context, vx_color_format , &addr_in, ptr, vx_mem_type(_info.mem_type));
    if((status = vxGetStatus((vx_reference)img)) != VX_SUCCESS)
        THROW("Error: vxCreateImageFromHandle(input:[" + TOSTR(_info.width()) + "x" + TOSTR(_info.height_batch()) + "]): failed " + TOSTR(status))

    _info._type = ImageInfo::Type::HANDLE;
    _info.data_size = size;
    return 0;
}
int Image::create(vx_context context) {
    vx_status status;
    vx_df_image vx_color_format = interpret_color_fmt(_info.color_fmt);
    img = vxCreateImage( context, _info.width(), _info.height_batch(), vx_color_format);
    if((status = vxGetStatus((vx_reference)img)) != VX_SUCCESS)
        THROW("Error: vxCreateImage(input:[" + TOSTR(_info.width()) + "x" + TOSTR(_info.height_batch()) + "]): failed " + TOSTR(status))
    _info._type = ImageInfo::Type::REGULAR;
    return 0;
}

unsigned Image::copy_data(unsigned char* user_buffer, bool sync)
{
    if(_info._type != ImageInfo::Type::HANDLE)
        return 0;
    if(!_queue)
    {
        ERR("Command queue not initialized for the image\n")
        return 0;
    }

    unsigned size = _info.width() *
                    _info.height_batch() *
                    _info.color_plane_count();

    if(_info.mem_type == RaliMemType::OCL)
    {

        cl_int status;
        if((status = clEnqueueReadBuffer(_queue,
                                         (cl_mem) buf,
                                         sync?(CL_TRUE):CL_FALSE,
                                         0,
                                         size,
                                         user_buffer,
                                         0 , nullptr, nullptr)) != CL_SUCCESS)
            THROW("clEnqueueReadBuffer failed: " + TOSTR(status))

    } else
    {
        memcpy(user_buffer, buf, size);
    }
    return size;
}
unsigned Image::copy_data(cl_mem user_buffer, bool sync)
{
    return 0;
}
