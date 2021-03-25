/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include <CL/cl.h>
#include <vx_ext_amd.h>
#include <VX/vx_types.h>
#include <cstring>
#include <sched.h>
#include <half.hpp>
#include "master_graph.h"
#include "parameter_factory.h"
#include "ocl_setup.h"
#include "meta_data_reader_factory.h"
#include "meta_data_graph_factory.h"
#include "randombboxcrop_meta_data_reader_factory.h"

using half_float::half;

#if ENABLE_SIMD
#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif
#endif

auto get_ago_affinity_info = []
    (RaliAffinity rali_affinity,
     int cpu_id,
     int gpu_id)
{
    AgoTargetAffinityInfo affinity;
    switch(rali_affinity) {
        case RaliAffinity::GPU:
            affinity.device_type =  AGO_TARGET_AFFINITY_GPU;
            affinity.device_info = (gpu_id >=0 && gpu_id <=9)? gpu_id : 0;
            break;
        case RaliAffinity::CPU:
            affinity.device_type = AGO_TARGET_AFFINITY_CPU;
            affinity.device_info = (cpu_id >=0 && cpu_id <=9)? cpu_id : 0;
            break;
        default:
            throw std::invalid_argument("Unsupported affinity");
    }
    return affinity;
};

MasterGraph::~MasterGraph()
{
    release();
}

MasterGraph::MasterGraph(size_t batch_size, RaliAffinity affinity, int gpu_id, size_t cpu_threads):
        _ring_buffer(OUTPUT_RING_BUFFER_DEPTH),
        _output_tensor(nullptr),
        _graph(nullptr),
        _affinity(affinity),
        _gpu_id(gpu_id),
        _convert_time("Conversion Time", DBG_TIMING),
        _user_batch_size(batch_size),
        _cpu_threads(cpu_threads),
        _mem_type ((_affinity == RaliAffinity::GPU) ? RaliMemType::OCL : RaliMemType::HOST),
        _process_time("Process Time", DBG_TIMING),
        _first_run(true),
        _processing(false),
        _internal_batch_size(compute_optimum_internal_batch_size(batch_size, affinity)),
        _user_to_internal_batch_ratio (_user_batch_size/_internal_batch_size)
{
    try {
        vx_status status;
        _context = vxCreateContext();
        auto vx_affinity = get_ago_affinity_info(_affinity, 0, gpu_id);
        if ((status = vxGetStatus((vx_reference) _context)) != VX_SUCCESS)
            THROW("vxCreateContext failed" + TOSTR(status))

        if(affinity == RaliAffinity::GPU)
        {
            cl_context _cl_context = nullptr;
            cl_device_id _cl_device_id = nullptr;
            get_device_and_context(gpu_id, &_cl_context, &_cl_device_id, CL_DEVICE_TYPE_GPU);
            if((status = vxSetContextAttribute(_context,
                    VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT,
                    &_cl_context, sizeof(cl_context)) != VX_SUCCESS))
                THROW("vxSetContextAttribute for CL_CONTEXT failed " + TOSTR(status))
        }


        // Setting attribute to run on CPU or GPU should be called before load kernel module
        if ((status = vxSetContextAttribute(_context,
                                            VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY,
                                            &vx_affinity,
                                            sizeof(vx_affinity))) != VX_SUCCESS)
            THROW("vxSetContextAttribute for AMD_AFFINITY failed " + TOSTR(status))

        // loading OpenVX RPP modules
        if ((status = vxLoadKernels(_context, "vx_rpp")) != VX_SUCCESS)
            THROW("Cannot load vx_rpp extension (vx_rpp), vxLoadKernels failed " + TOSTR(status))
        else
            LOG("vx_rpp module loaded successfully")
#ifdef RALI_VIDEO
        // loading video decoder modules
        if ((status = vxLoadKernels(_context, "vx_amd_media")) != VX_SUCCESS)
            WRN("Cannot load vx_amd_media extension, video decode functionality will not be available")
        else
            LOG("vx_amd_media module loaded")
#endif
        if(_affinity == RaliAffinity::GPU)
            _device.init_ocl(_context);
    }
    catch(const std::exception& e)
    {
        release();
        throw;
    }
}

MasterGraph::Status
MasterGraph::run()
{
    if(!_processing)// The user should not call the run function before the build() is called or while reset() is happening
        return MasterGraph::Status::NOT_RUNNING;

    if(no_more_processed_data()) {
        return MasterGraph::Status::NO_MORE_DATA;
    }

    _ring_buffer.block_if_empty();// wait here if the user thread (caller of this function) is faster in consuming the processed images compare to th output routine in producing them

    if(_first_run)
    {
        // calling run pops the processed images that have been used by user, when user calls run() for the first time
        // they've not used anything yet, so we don't pop a batch from the _ring_buffer
        _first_run = false;
    } else {
        _ring_buffer.pop(); // Pop previously used output images and metadata from the ring buffer
    }

    // If the last batch of processed imaged has been just popped from the ring_buffer it means user has previously consumed all the processed images.
    // User should check using the IsEmpty() API and not call run() or copy() API when there is no more data. run() will return MasterGraph::Status::NO_MORE_DATA flag to notify it.
    if(no_more_processed_data()) {
        return MasterGraph::Status::NO_MORE_DATA;
    }

    decrease_image_count();

    return MasterGraph::Status::OK;
}

void
MasterGraph::decrease_image_count()
{
    if(!_loop)
        _remaining_images_count -= _user_batch_size;
}
void
MasterGraph::create_single_graph()
{
    // Actual graph creating and calls into adding nodes to graph is deferred and is happening here to enable potential future optimizations
    _graph = std::make_shared<Graph>(_context, _affinity, 0, _gpu_id);
    for(auto& node: _nodes)
    {
        // Any image not yet created can be created as virtual image
        for(auto& image: node->output())
            if(image->info().type() == ImageInfo::Type::UNKNOWN)
            {
                image->create_virtual(_context, _graph->get());
                _internal_images.push_back(image);
            }
        node->create(_graph);
    }
    _graph->verify();
}

MasterGraph::Status
MasterGraph::build()
{
    if(_output_images.empty())
        THROW("No output images are there, cannot create the pipeline")

    // Verify all output images have the same dimension, otherwise creating a unified tensor from them is not supported
    _output_image_info = _output_images.front()->info();
    for(auto&& output_image : _output_images)
        if(!(output_image->info() == _output_image_info))
            THROW("Dimension of the output images do not match")

    allocate_output_tensor();

    _ring_buffer.init(_mem_type, _device.resources(), output_byte_size(), _output_images.size());
    create_single_graph();
    start_processing();
    return Status::OK;
}
Image *
MasterGraph::create_loader_output_image(const ImageInfo &info)
{
    /*
    *   NOTE: Output image for a source node needs to be created as a regular (non-virtual) image
    */
    auto output = new Image(info);

    if(output->create_from_handle(_context) != 0)
        THROW("Creating output image for loader failed");

    _internal_images.push_back(output);

    return output;
}

Image *
MasterGraph::create_image(const ImageInfo &info, bool is_output)
{
    auto* output = new Image(info);
    // if the image is not an output image, the image creation is deferred and later it'll be created as a virtual image
    if(is_output)
    {
        if (output->create_from_handle(_context) != 0)
            THROW("Cannot create the image from handle")

        _output_images.push_back(output);
    }

    return output;
}

void MasterGraph::release()
{
    LOG("MasterGraph release ...")
    stop_processing();
    _nodes.clear();
    _root_nodes.clear();
    _image_map.clear();

    // release all openvx resources.
    vx_status status;
    for(auto& image: _internal_images)
        delete image;// It will call the vxReleaseImage internally in the destructor
    for(auto& image: _output_images)
        delete image;// It will call the vxReleaseImage internally in the destructor
    if(_graph != nullptr)
        _graph->release();
    if(_context && (status = vxReleaseContext(&_context)) != VX_SUCCESS)
        LOG ("Failed to call vxReleaseContext " + TOSTR(status))

    _augmented_meta_data = nullptr;
    _meta_data_graph = nullptr;
    _meta_data_reader = nullptr;
    deallocate_output_tensor();
}

MasterGraph::Status
MasterGraph::update_node_parameters()
{
    // Randomize random parameters
    ParameterFactory::instance()->renew_parameters();

    // Apply renewed parameters to VX parameters used in augmentation
    for(auto& node: _nodes)
        node->update_parameters();

    return Status::OK;
}

size_t
MasterGraph::augmentation_branch_count()
{
    return _output_images.size();
}

RaliColorFormat
MasterGraph::output_color_format()
{
    return _output_image_info.color_format();
}

size_t
MasterGraph::output_width()
{
    return _output_image_info.width();
}

size_t
MasterGraph::output_height()
{
    return _output_image_info.height_batch()*_user_to_internal_batch_ratio;
}

MasterGraph::Status
MasterGraph::allocate_output_tensor()
{
    // creating a float buffer that can accommodates all output images
    size_t output_float_buffer_size = output_byte_size() * _output_images.size();

    if(processing_on_device())
    {
        cl_int ret = CL_SUCCESS;
        _output_tensor = nullptr;
        size_t size = output_float_buffer_size*sizeof(cl_float);
        cl_mem clImgFloat  = clCreateBuffer(_device.resources().context,
                                            CL_MEM_READ_WRITE,
                                            size,
                                            nullptr, &ret);

        if (!clImgFloat || ret != CL_SUCCESS)
            THROW("clCreateBuffer of size " + TOSTR(size) + " failed " + TOSTR(ret))

        _output_tensor = clImgFloat;
    }
    return Status::OK;
}

MasterGraph::Status
MasterGraph::deallocate_output_tensor()
{
    if(processing_on_device() && _output_tensor != nullptr)
        clReleaseMemObject(_output_tensor );

    return Status::OK;
}

MasterGraph::Status
MasterGraph::reset()
{
    // stop the internal processing thread so that the
    _processing = false;
    _ring_buffer.unblock_writer();
    if(_output_thread.joinable())
        _output_thread.join();
    _ring_buffer.reset();
    // clearing meta ring buffer

    // resetting loader module to start from the beginning of the media and clear it's internal state/buffers
    _loader_module->reset();
    // restart processing of the images
    _first_run = true;
    _output_routine_finished_processing = false;
    start_processing();
    return Status::OK;
}

size_t
MasterGraph::remaining_images_count()
{
    return (_remaining_images_count >= 0) ? _remaining_images_count:0;
}

RaliMemType
MasterGraph::mem_type()
{
    return _mem_type;
}

Timing
MasterGraph::timing()
{
    Timing t = _loader_module->timing();
    t.image_process_time += _process_time.get_timing();
    t.copy_to_output += _convert_time.get_timing();
    return t;
}


MasterGraph::Status
MasterGraph::copy_output(
        cl_mem out_ptr,
        size_t out_size)
{
    if(no_more_processed_data())
        return MasterGraph::Status::NO_MORE_DATA;

    return Status::NOT_IMPLEMENTED;
    _convert_time.start();
    _convert_time.end();
    return Status::OK;
}

#define CHECK_CL_CALL_RET(x) { cl_int ret; ret = x; if( ret != CL_SUCCESS) THROW("ocl call failed "+STR(#x)+" error "+TOSTR(ret)) }

MasterGraph::Status
MasterGraph::copy_out_tensor(void *out_ptr, RaliTensorFormat format, float multiplier0, float multiplier1,
                             float multiplier2, float offset0, float offset1, float offset2, bool reverse_channels, RaliTensorDataType output_data_type)
{
    if(no_more_processed_data())
        return MasterGraph::Status::NO_MORE_DATA;

    if (output_color_format() == RaliColorFormat::RGB_PLANAR)
        return MasterGraph::copy_out_tensor_planar(out_ptr,format,multiplier0, multiplier1, multiplier2, offset0, offset1, offset2, reverse_channels, output_data_type);

    _convert_time.start();
    // Copies to the output context given by the user
    unsigned int n = _user_batch_size;
    const size_t c = output_depth();
    const size_t h = _output_image_info.height_single();
    const size_t w = output_width();
    const size_t single_output_image_size = output_byte_size();

    if(_output_image_info.mem_type() == RaliMemType::OCL)
    {
        if(output_data_type == RaliTensorDataType::FP16)
            THROW("FP16 tensor output for GPU affinity is not implemented")
        // OCL device memory
        cl_int status;

        size_t global_work_size = output_sample_size();
        size_t local_work_size = 256;

        // TODO: Use the runKernel function instead

        auto kernel_name = (format == RaliTensorFormat::NHWC)? "copyInt8ToNHWC" : "copyInt8ToNCHW";
        cl_kernel kernel = _device["utility"][kernel_name];
        auto queue = _device.resources().cmd_queue;
        unsigned dest_buf_offset = 0;
        auto output_buffers =_ring_buffer.get_read_buffers();
        for( auto&& out_image: output_buffers)
        {
            int argIdx = 0;
            unsigned reverse_chnl = reverse_channels ? 1 : 0;
            auto img_buffer = out_image;
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_mem), (void*)& (img_buffer)))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_mem), (void*)&_output_tensor ))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_uint), (void*)& dest_buf_offset))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_uint), (void*)& w))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_uint), (void*)& h))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_uint), (void*)& c))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_float), (void*)& multiplier0))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_float), (void*)& multiplier1))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_float), (void*)& multiplier2))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_float), (void*)& offset0))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_float), (void*)& offset1))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_float), (void*)& offset2))
            CHECK_CL_CALL_RET(clSetKernelArg( kernel, argIdx++, sizeof(cl_uint), (void*)& reverse_chnl))


            if((status = clEnqueueNDRangeKernel(queue,
                                                kernel,
                                                1,
                                                nullptr,
                                                &global_work_size,
                                                &local_work_size,
                                                0 , nullptr, nullptr)) != CL_SUCCESS)
                THROW("clEnqueueNDRangeKernel failed on kernel "+STR(kernel_name)+" error " + TOSTR(status))

            dest_buf_offset += single_output_image_size;
        }

        int read_size = single_output_image_size*_output_images.size()*sizeof(cl_float);
        if((status = clEnqueueReadBuffer(queue,
                                         _output_tensor,
                                         CL_TRUE,
                                         0,
                                         read_size,
                                         out_ptr,
                                         0 , nullptr, nullptr)) != CL_SUCCESS)
            THROW("clEnqueueReadBuffer failed: " + TOSTR(status))
    }
    if(_output_image_info.mem_type() == RaliMemType::HOST)
    {
        float multiplier[3] = {multiplier0, multiplier1, multiplier2 };
        float offset[3] = {offset0, offset1, offset2 };
        size_t dest_buf_offset = 0;

        auto output_buffers =_ring_buffer.get_read_buffers();
        for( auto&& out_image: output_buffers)
        {
            auto in_buffer = (unsigned char*)out_image;
            if(format == RaliTensorFormat::NHWC)
            {
                if(output_data_type == RaliTensorDataType::FP32)
                {
                    for (unsigned int nCount = 0; nCount < n; nCount++) {
                        float *output_tensor_32 = static_cast<float *>(out_ptr);
                        auto channel_size = w * h;
                        for (unsigned channel_idx = 0; channel_idx < c; channel_idx++) {
                            for (unsigned i = 0; i < channel_size; i++)
                                output_tensor_32[dest_buf_offset + channel_idx + i * c] =
                                        offset[channel_idx] + multiplier[channel_idx] *
                                                              (reverse_channels ? (float) (in_buffer[i * c + c - channel_idx - 1])
                                                                                : (float) (in_buffer[i * c + channel_idx]));
                        }
                        in_buffer += (w * c * h);
                        dest_buf_offset += (w * c * h);
                    }
                }
                else if(output_data_type == RaliTensorDataType::FP16)
                {
                    for (unsigned int nCount = 0; nCount < n; nCount++) {
                        half *output_tensor_16 = static_cast<half *>(out_ptr);
                        auto channel_size = w * h;
                        for (unsigned channel_idx = 0; channel_idx < c; channel_idx++) {
                            for (unsigned i = 0; i < channel_size; i++)
                                output_tensor_16[dest_buf_offset + channel_idx + i * c] =
                                        offset[channel_idx] + multiplier[channel_idx] *
                                                              (reverse_channels ? (half) (in_buffer[i * c + c - channel_idx - 1])
                                                                                : (half) (in_buffer[i * c + channel_idx]));
                        }
                        dest_buf_offset += (w * c * h);
                        in_buffer += (w * c * h);
                    }
                }
            }
            if(format == RaliTensorFormat::NCHW)
            {
                if(output_data_type == RaliTensorDataType::FP32)
                {
                    float *output_tensor_32 = static_cast<float *>(out_ptr);
                    auto channel_size  = w * h;
                    if(c != 3)
                    {
                        for (unsigned int nCount = 0; nCount < n; nCount++)
                        {
                            for(unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                                for(unsigned i = 0; i < channel_size; i++)
                                    output_tensor_32[dest_buf_offset+channel_idx*channel_size + i] =
                                            offset[channel_idx] + multiplier[channel_idx]*(reverse_channels ? (float)(in_buffer[dest_buf_offset + (c*i+c-channel_idx-1)]) : (float)(in_buffer[dest_buf_offset + (c*i+channel_idx)]));

                            dest_buf_offset += (w * c * h);
                        }
                    }
                    else {
#if (ENABLE_SIMD && __AVX2__)
                        for (unsigned int nCount = 0; nCount < n; nCount++) {
                            float *B_buf = output_tensor_32 + dest_buf_offset;
                            float *G_buf = B_buf + channel_size;
                            float *R_buf = G_buf + channel_size;

                            __m256i mask_B, mask_G, mask_R;
                            if (reverse_channels) {
                                mask_B = _mm256_setr_epi32(0x80808000, 0x80808003, 0x80808006, 0x80808009, 0x80808000,
                                                           0x80808003, 0x80808006, 0x80808009);
                                mask_G = _mm256_setr_epi32(0x80808001, 0x80808004, 0x80808007, 0x8080800A, 0x80808001,
                                                           0x80808004, 0x80808007, 0x8080800A);
                                mask_R = _mm256_setr_epi32(0x80808002, 0x80808005, 0x80808008, 0x8080800B, 0x80808002,
                                                           0x80808005, 0x80808008, 0x8080800B);
                            } else {
                                mask_R = _mm256_setr_epi32(0x80808000, 0x80808003, 0x80808006, 0x80808009, 0x80808000,
                                                           0x80808003, 0x80808006, 0x80808009);
                                mask_G = _mm256_setr_epi32(0x80808001, 0x80808004, 0x80808007, 0x8080800A, 0x80808001,
                                                           0x80808004, 0x80808007, 0x8080800A);
                                mask_B = _mm256_setr_epi32(0x80808002, 0x80808005, 0x80808008, 0x8080800B, 0x80808002,
                                                           0x80808005, 0x80808008, 0x8080800B);
                            }
                            __m256 pmul0 = _mm256_set1_ps(multiplier0);
                            __m256 pmul1 = _mm256_set1_ps(multiplier1);
                            __m256 pmul2 = _mm256_set1_ps(multiplier2);
                            __m256 padd0 = _mm256_set1_ps(offset0);
                            __m256 padd1 = _mm256_set1_ps(offset1);
                            __m256 padd2 = _mm256_set1_ps(offset2);
                            unsigned int alignedLength = (channel_size & ~7);    // multiple of 8
                            unsigned int i = 0;

                            __m256 fR, fG, fB;
                            for (; i < alignedLength; i += 8) {
                                __m256i pix0 = _mm256_loadu_si256((const __m256i *) in_buffer);
                                pix0 = _mm256_permutevar8x32_epi32(pix0, _mm256_setr_epi32(0, 1, 2, 3, 3, 4, 5, 6));
                                fB = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(pix0, mask_R));
                                fG = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(pix0, mask_G));
                                fR = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(pix0, mask_B));
                                fB = _mm256_mul_ps(fB, pmul0);
                                fG = _mm256_mul_ps(fG, pmul1);
                                fR = _mm256_mul_ps(fR, pmul2);
                                fB = _mm256_add_ps(fB, padd0);
                                fG = _mm256_add_ps(fG, padd1);
                                fR = _mm256_add_ps(fR, padd2);
                                _mm256_storeu_ps(B_buf, fB);
                                _mm256_storeu_ps(G_buf, fG);
                                _mm256_storeu_ps(R_buf, fR);
                                B_buf += 8;
                                G_buf += 8;
                                R_buf += 8;
                                in_buffer += 24;
                            }
                            for (; i < channel_size; i++, in_buffer += 3) {
                                *B_buf++ = (in_buffer[0] * multiplier0) + offset0;
                                *G_buf++ = (in_buffer[1] * multiplier1) + offset1;
                                *R_buf++ = (in_buffer[2] * multiplier2) + offset1;
                            }
                            dest_buf_offset += (w * c * h);
                        }
#else
                        for (unsigned int nCount = 0; nCount < n; nCount++)
                        {
                            for(unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                                for(unsigned i = 0; i < channel_size; i++)
                                    output_tensor_32[dest_buf_offset+channel_idx*channel_size + i] =
                                            offset[channel_idx] + multiplier[channel_idx]*(reverse_channels ? (float)(in_buffer[dest_buf_offset + (c*i+c-channel_idx-1)]) : (float)(in_buffer[dest_buf_offset + (c*i+channel_idx)]));

                            dest_buf_offset += (w * c * h);
                        }
#endif
                    }
                }
                else if(output_data_type == RaliTensorDataType::FP16)
                {
                    for (unsigned int nCount = 0; nCount < n; nCount++) {
                        half *output_tensor_16 = static_cast<half *>(out_ptr);
                        auto channel_size = w * h;
                        for (unsigned channel_idx = 0; channel_idx < c; channel_idx++) {
                            for (unsigned i = 0; i < channel_size; i++)
                                output_tensor_16[dest_buf_offset + channel_idx * channel_size + i] =
                                        offset[channel_idx] + multiplier[channel_idx] *
                                                              (reverse_channels ? (half) (in_buffer[dest_buf_offset + (c*i+c-channel_idx-1)])
                                                                                : (half) (in_buffer[dest_buf_offset + (c * i + channel_idx)]));
                        }
                        dest_buf_offset += (w * c * h);
                        in_buffer += (w * c * h);
                    }
                }

            }
            dest_buf_offset += single_output_image_size;
        }
    }
    _convert_time.end();
    return Status::OK;
}

MasterGraph::Status
MasterGraph::copy_output(unsigned char *out_ptr)
{
    if(no_more_processed_data())
        return MasterGraph::Status::NO_MORE_DATA;

    _convert_time.start();
    // Copies to the output context given by the user
    size_t size = output_byte_size();

    size_t dest_buf_offset = 0;

    if(processing_on_device())
    {
        //NOTE: the CL_TRUE flag is only used on the last buffer read call,
        // to avoid unnecessary sequence of synchronizations

        // get_read_buffers() calls block_if_empty() internally and blocks if buffers are empty until a new batch is processed
        auto output_buffers =_ring_buffer.get_read_buffers();
        auto out_image_idx = output_buffers.size();
        for( auto&& output_handle: output_buffers)
        {
            bool sync_flag = (--out_image_idx == 0) ? CL_TRUE : CL_FALSE;
            cl_int status;
            if((status = clEnqueueReadBuffer(_device.resources().cmd_queue,
                                             (cl_mem) output_handle,
                                             sync_flag?(CL_TRUE):CL_FALSE,
                                             0,
                                             size,
                                             out_ptr+dest_buf_offset,
                                             0 , nullptr, nullptr)) != CL_SUCCESS)
                THROW("clEnqueueReadBuffer failed: " + TOSTR(status))
            dest_buf_offset += size;
        }
    }
    else
    {
        // get_host_master_read_buffer is blocking if _ring_buffer is empty, and blocks this thread till internal processing thread process a new batch and store in the _ring_buffer
        memcpy(out_ptr, _ring_buffer.get_host_master_read_buffer(), size * _output_images.size());
    }
    _convert_time.end();
    return Status::OK;
}

ImageNameBatch& operator+=(ImageNameBatch& dest, const ImageNameBatch& src)
{
    dest.insert(dest.end(), src.cbegin(), src.cend());
    return dest;
}

void MasterGraph::output_routine()
{
    INFO("Output routine started with "+TOSTR(_remaining_images_count) + " to load");
    if(processing_on_device() && _user_to_internal_batch_ratio != 1)
        THROW("Internal failure, in the GPU processing case, user and input batch size must be equal")

    try {
        while (_processing)
        {
            const size_t each_cycle_size = output_byte_size()/_user_to_internal_batch_ratio;

            ImageNameBatch full_batch_image_names = {};
            pMetaDataBatch full_batch_meta_data = nullptr;
            pMetaDataBatch augmented_batch_meta_data = nullptr; 

            if (_loader_module->remaining_count() < _user_batch_size)
            {
                // If the internal process routine ,output_routine(), has finished processing all the images, and last
                // processed images stored in the _ring_buffer will be consumed by the user when it calls the run() func
                notify_user_thread();
                // the following call is required in case the ring buffer is waiting for more data to be loaded and there is no more data to process.
                _ring_buffer.release_if_empty();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            // _ring_buffer.get_write_buffers() is blocking and blocks here until user uses processed image by calling run() and frees space in the ring_buffer
            auto write_buffers = _ring_buffer.get_write_buffers();

            // When executing on CPU the internal batch count can be smaller than the user batch count
            // In that case the user_batch_size will be an integer multiple of the _internal_batch_size
            // Multiple cycles worth of internal_batch_size images should be processed to complete a full _user_batch_size
            for(unsigned cycle_idx = 0; cycle_idx< _user_to_internal_batch_ratio; cycle_idx++)
            {
                // Swap handles on the input image, so that new image is loaded to be processed
                auto load_ret = _loader_module->load_next();
                if (load_ret != LoaderModuleStatus::OK)
                    THROW("Loader module failed to load next batch of images, status " + TOSTR(load_ret))

                if (!_processing)
                    break;

                auto this_cycle_names =  _loader_module->get_id();
                auto decode_image_info = _loader_module->get_decode_image_info();                

                if(this_cycle_names.size() != _internal_batch_size)
                    WRN("Internal problem: names count "+ TOSTR(this_cycle_names.size()))

                // meta_data lookup is done before _meta_data_graph->process() is called to have the new meta_data ready for processing
                if (_meta_data_reader)
                    _meta_data_reader->lookup(this_cycle_names);

                full_batch_image_names += this_cycle_names;

                if (!_processing)
                    break;

                // Swap handles on the output images, so that new processed image will be written to the a new buffer
                for (size_t idx = 0; idx < _output_images.size(); idx++)
                {
                    if(_affinity == RaliAffinity::GPU)
                        _output_images[idx]->swap_handle(write_buffers[idx]);
                    else
                    {
                        auto this_cycle_buffer_ptr = (unsigned char *) write_buffers[idx] + each_cycle_size * cycle_idx;
                        _output_images[idx]->swap_handle(this_cycle_buffer_ptr);
                    }
                }

                if (!_processing)
                    break;

                for(auto node: _nodes)
                {
                    if(node->_is_ssd)
                    {
                        node->set_meta_data(_augmented_meta_data);
                    }
                }

                update_node_parameters();
                if(_augmented_meta_data)
                {
                    if (_meta_data_graph)
                    {
                        if(_is_random_bbox_crop)
                        {
                            _randombboxcrop_meta_data_reader->lookup(this_cycle_names);
                            _meta_data_graph->update_random_bbox_meta_data(_random_bbox_crop_cords_data ,_augmented_meta_data, decode_image_info);
                        }
                        else
                        {
                            _meta_data_graph->update_meta_data(_augmented_meta_data, decode_image_info);
                        }
                        _meta_data_graph->process(_augmented_meta_data);
                    }
                    if (full_batch_meta_data)
                        full_batch_meta_data->concatenate(_augmented_meta_data);
                    else
                        full_batch_meta_data = _augmented_meta_data->clone();
                }
                _process_time.start();
                _graph->process();
                _process_time.end();
            }

            _ring_buffer.set_meta_data(full_batch_image_names, full_batch_meta_data);
            _ring_buffer.push(); // Image data and metadata is now stored in output the ring_buffer, increases it's level by 1

        }
    }
    catch (const std::exception &e)
    {
        ERR("Exception thrown in the process routine: " + STR(e.what()) + STR("\n"));
        _processing = false;
        _ring_buffer.release_all_blocked_calls();
    }
}

void MasterGraph::start_processing()
{
    _processing = true;
    _remaining_images_count = _loader_module->remaining_count();
    _output_thread = std::thread(&MasterGraph::output_routine, this);
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
//  Changing thread scheduling policy and it's priority does not help on latest Ubuntu builds
//  and needs tweaking the Linux security settings , can be turned on for experimentation
#if 0
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    auto thread = _output_thread.native_handle();
    auto ret = pthread_setschedparam(thread, SCHED_FIFO, &params);
    if (ret != 0)
        WRN("Unsuccessful in setting thread realtime priority for process thread err = "+STR(std::strerror(ret)))
#endif
#endif
}

void MasterGraph::stop_processing()
{
    _processing = false;
    _ring_buffer.unblock_reader();
    _ring_buffer.unblock_writer();
    if(_output_thread.joinable())
        _output_thread.join();
}

MetaDataBatch * MasterGraph::create_coco_meta_data_reader(const char *source_path, bool is_output)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(MetaDataType::BoundingBox, MetaDataReaderType::COCO_META_DATA_READER, source_path);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    if(is_output)
    {
        if (_augmented_meta_data)
            THROW("Metadata output already defined, there can only be a single output for metadata augmentation")
        else
            _augmented_meta_data = _meta_data_reader->get_output();
    }
    return _meta_data_reader->get_output();
}

 MetaDataBatch * MasterGraph::create_tf_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type , MetaDataType label_type, std::map<std::string, std::string> feature_key_map)
 {
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(label_type, reader_type, source_path, feature_key_map);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    if (_augmented_meta_data)
        THROW("Metadata output already defined, there can only be a single output for metadata augmentation")
    else
        _augmented_meta_data = _meta_data_reader->get_output();
    return _meta_data_reader->get_output();
}

MetaDataBatch * MasterGraph::create_label_reader(const char *source_path, MetaDataReaderType reader_type)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(MetaDataType::Label, reader_type, source_path);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")
    else
        _augmented_meta_data = _meta_data_reader->get_output();
    return _meta_data_reader->get_output();
}


void MasterGraph::create_randombboxcrop_reader(RandomBBoxCrop_MetaDataReaderType reader_type, RandomBBoxCrop_MetaDataType label_type, bool all_boxes_overlap, bool no_crop, FloatParam* aspect_ratio, bool has_shape, int crop_width, int crop_height, int num_attempts, FloatParam* scaling, int total_num_attempts)
{
    if( _randombboxcrop_meta_data_reader)
        THROW("A metadata reader has already been created")
    _is_random_bbox_crop = true;
    RandomBBoxCrop_MetaDataConfig config(label_type, reader_type, all_boxes_overlap, no_crop, aspect_ratio, has_shape, crop_width, crop_height, num_attempts, scaling, total_num_attempts);
    _randombboxcrop_meta_data_reader = create_meta_data_reader(config);
    _randombboxcrop_meta_data_reader->set_meta_data(_meta_data_reader);
    _randombboxcrop_meta_data_reader->read_all();
    if (_random_bbox_crop_cords_data)
        THROW("Metadata can only have a single output")
    else
        _random_bbox_crop_cords_data = _randombboxcrop_meta_data_reader->get_output();
}

MetaDataBatch * MasterGraph::create_caffe2_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type , MetaDataType label_type)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(label_type, reader_type, source_path);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    if (_augmented_meta_data)
        THROW("Metadata output already defined, there can only be a single output for metadata augmentation")
    else
        _augmented_meta_data = _meta_data_reader->get_output();
    return _meta_data_reader->get_output();
}

MetaDataBatch * MasterGraph::create_caffe_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type , MetaDataType label_type)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(label_type, reader_type, source_path);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    if (_augmented_meta_data)
        THROW("Metadata output already defined, there can only be a single output for metadata augmentation")
    else
        _augmented_meta_data = _meta_data_reader->get_output();
    return _meta_data_reader->get_output();
}

MetaDataBatch * MasterGraph::create_cifar10_label_reader(const char *source_path, const char *file_prefix)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(MetaDataType::Label, MetaDataReaderType::CIFAR10_META_DATA_READER, source_path, std::map<std::string, std::string>(), file_prefix);    
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")
    else
        _augmented_meta_data = _meta_data_reader->get_output();
    return _meta_data_reader->get_output();
}


const std::pair<ImageNameBatch,pMetaDataBatch>& MasterGraph::meta_data()
{
    if(_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    return _ring_buffer.get_meta_data();
}

size_t MasterGraph::compute_optimum_internal_batch_size(size_t user_batch_size, RaliAffinity affinity)
{
    const unsigned MINIMUM_CPU_THREAD_COUNT = 2;
    const unsigned DEFAULT_SMT_COUNT = 2;


    if(affinity == RaliAffinity::GPU)
        return user_batch_size;
    
    unsigned THREAD_COUNT = std::thread::hardware_concurrency();
    if(THREAD_COUNT >= MINIMUM_CPU_THREAD_COUNT)
        INFO("Can run " + TOSTR(THREAD_COUNT) + " threads simultaneously on this machine")
    else
    {
        THREAD_COUNT = MINIMUM_CPU_THREAD_COUNT;
        WRN("hardware_concurrency() call failed assuming can run " + TOSTR(THREAD_COUNT) + " threads")
    }
    size_t ret = user_batch_size;
    size_t CORE_COUNT = THREAD_COUNT / DEFAULT_SMT_COUNT;

    if(CORE_COUNT <= 0)
        THROW("Wrong core count detected less than 0")

    for( size_t i = CORE_COUNT; i <= THREAD_COUNT; i++)
        if(user_batch_size % i == 0)
        {
            ret = i;
            break;
        }

    for(size_t i = CORE_COUNT; i > 1; i--)
        if(user_batch_size % i == 0)
        {
            ret = i;
            break;
        }
    INFO("User batch size "+ TOSTR(user_batch_size)+" Internal batch size set to "+ TOSTR(ret))
    return ret;
}

size_t MasterGraph::output_sample_size()
{
    return output_height() * output_width() * output_depth();
}

size_t MasterGraph::output_byte_size()
{
    return output_height() * output_width() * output_depth() * SAMPLE_SIZE;
}

size_t MasterGraph::output_depth()
{
    return _output_image_info.color_plane_count();
}

void MasterGraph::notify_user_thread()
{
    if(_output_routine_finished_processing)
        return;
    LOG("Output routine finished processing all images, no more image to be processed")
    _output_routine_finished_processing = true;
}

bool MasterGraph::no_more_processed_data()
{
    return (_output_routine_finished_processing && _ring_buffer.empty());
}

MasterGraph::Status
MasterGraph::copy_out_tensor_planar(void *out_ptr, RaliTensorFormat format, float multiplier0, float multiplier1,
                             float multiplier2, float offset0, float offset1, float offset2, bool reverse_channels, RaliTensorDataType output_data_type)
{
    if(no_more_processed_data())
        return MasterGraph::Status::NO_MORE_DATA;

    _convert_time.start();
    // Copies to the output context given by the user, each image is copied separate for planar
    const size_t w = output_width();
    const size_t h = _output_image_info.height_single();
    const size_t c = output_depth();
    const size_t n = _output_image_info.batch_size();

    const size_t single_output_image_size = output_byte_size();


    if(_output_image_info.mem_type() == RaliMemType::OCL)
    {
        THROW("copy_out_tensor_planar for GPU affinity is not implemented")
    }
    if(_output_image_info.mem_type() == RaliMemType::HOST)
    {
        float multiplier[3] = {multiplier0, multiplier1, multiplier2 };
        float offset[3] = {offset0, offset1, offset2 };
        size_t dest_buf_offset = 0;

        auto output_buffers =_ring_buffer.get_read_buffers();

        for( auto&& out_image: output_buffers)
        {
            for (unsigned batch = 0; batch < n ; batch++) {
                const size_t batch_offset = w*h*c*batch;
                auto in_buffer = (unsigned char *) out_image + batch_offset;
                if (format == RaliTensorFormat::NHWC) {
                    if (output_data_type == RaliTensorDataType::FP32) {
                        float *output_tensor_32 = static_cast<float *>(out_ptr) + batch_offset;
                        auto channel_size = w * h;
                        for (unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                            for (unsigned i = 0; i < channel_size; i++)
                                output_tensor_32[dest_buf_offset + channel_idx + i * c] =
                                        offset[channel_idx] + multiplier[channel_idx] *
                                                              (reverse_channels ? (float) (in_buffer[i +
                                                                                                     (c - channel_idx -
                                                                                                      1) *
                                                                                                     channel_size])
                                                                                : (float) (in_buffer[i + channel_idx *
                                                                                                         channel_size]));
                    } else if (output_data_type == RaliTensorDataType::FP16) {
                        half *output_tensor_16 = static_cast<half *>(out_ptr) + batch_offset;
                        auto channel_size = w * h;
                        for (unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                            for (unsigned i = 0; i < channel_size; i++)
                                output_tensor_16[dest_buf_offset + channel_idx + i * c] =
                                        offset[channel_idx] + multiplier[channel_idx] *
                                                              (reverse_channels ? (half) (in_buffer[
                                                                      (c - channel_idx - 1) * channel_size + i])
                                                                                : (half) (in_buffer[
                                                                              channel_idx * channel_size + i]));
                    }
                }
                if (format == RaliTensorFormat::NCHW) {
                    if (output_data_type == RaliTensorDataType::FP32) {
                        float *output_tensor_32 = static_cast<float *>(out_ptr) + batch_offset;
                        //output_tensor_32 += batch_offset;
                        auto channel_size = w * h;
                        if (c != 3) {
                            for (unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                                for (unsigned i = 0; i < channel_size; i++)
                                    output_tensor_32[dest_buf_offset + channel_idx * channel_size + i] =
                                            offset[channel_idx] + multiplier[channel_idx] *
                                                                  (reverse_channels ? (float) (in_buffer[
                                                                          (c - channel_idx - 1) * channel_size + i])
                                                                                    : (float) (in_buffer[
                                                                                  channel_idx * channel_size + i]));
                        } else {
#if (ENABLE_SIMD && __AVX2__)

                            float *B_buf = output_tensor_32 + dest_buf_offset;
                            float *G_buf = B_buf + channel_size;
                            float *R_buf = G_buf + channel_size;
                            unsigned char *in_buffer_R = in_buffer;
                            unsigned char *in_buffer_G = in_buffer + channel_size;
                            unsigned char *in_buffer_B = in_buffer_G + channel_size;

                            __m256 pmul0 = _mm256_set1_ps(multiplier0);
                            __m256 pmul1 = _mm256_set1_ps(multiplier1);
                            __m256 pmul2 = _mm256_set1_ps(multiplier2);
                            __m256 padd0 = _mm256_set1_ps(offset0);
                            __m256 padd1 = _mm256_set1_ps(offset1);
                            __m256 padd2 = _mm256_set1_ps(offset2);
                            unsigned int alignedLength = (channel_size & ~7);    // multiple of 8
                            unsigned int i = 0;

                            __m256 fR, fG, fB;
                            for (; i < alignedLength; i += 8) {
                                __m128i pixR, pixG, pixB;
                                if (reverse_channels) {
                                    pixB = _mm_loadl_epi64((const __m128i *) in_buffer_R);
                                    pixG = _mm_loadl_epi64((const __m128i *) in_buffer_G);
                                    pixR = _mm_loadl_epi64((const __m128i *) in_buffer_B);
                                } else {
                                    pixR = _mm_loadl_epi64((const __m128i *) in_buffer_R);
                                    pixG = _mm_loadl_epi64((const __m128i *) in_buffer_G);
                                    pixB = _mm_loadl_epi64((const __m128i *) in_buffer_B);
                                }
                                fB = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pixR));
                                fG = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pixG));
                                fR = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pixB));
                                fB = _mm256_mul_ps(fB, pmul0);
                                fG = _mm256_mul_ps(fG, pmul1);
                                fR = _mm256_mul_ps(fR, pmul2);
                                fB = _mm256_add_ps(fB, padd0);
                                fG = _mm256_add_ps(fG, padd1);
                                fR = _mm256_add_ps(fR, padd2);
                                _mm256_storeu_ps(B_buf, fB);
                                _mm256_storeu_ps(G_buf, fG);
                                _mm256_storeu_ps(R_buf, fR);
                                B_buf += 8;
                                G_buf += 8;
                                R_buf += 8;
                                in_buffer_R += 8, in_buffer_G += 8, in_buffer_B += 8;
                            }
                            for (; i < channel_size; i++) {
                                *B_buf++ = (*in_buffer_R++ * multiplier0) + offset0;
                                *G_buf++ = (*in_buffer_G++ * multiplier1) + offset1;
                                *R_buf++ = (*in_buffer_B++ * multiplier2) + offset1;
                            }

#else
                            for(unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                                for(unsigned i = 0; i < channel_size; i++)
                                    output_tensor_32[dest_buf_offset+channel_idx*channel_size + i] =
                                            offset[channel_idx] + multiplier[channel_idx]*(reverse_channels ? (float)(in_buffer[i+(c-channel_idx-1)*channel_size]) : (float)(in_buffer[i+channel_idx*channel_size]));
#endif
                        }
                    } else if (output_data_type == RaliTensorDataType::FP16) {
                        half *output_tensor_16 = static_cast<half *>(out_ptr) + batch_offset;
                        auto channel_size = w * h;
                        for (unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                            for (unsigned i = 0; i < channel_size; i++)
                                output_tensor_16[dest_buf_offset + channel_idx * channel_size + i] =
                                        offset[channel_idx] + multiplier[channel_idx] *
                                                              (reverse_channels ? (half) (in_buffer[i +
                                                                                                    (c - channel_idx -
                                                                                                     1) * channel_size])
                                                                                : (half) (in_buffer[i + channel_idx *
                                                                                                        channel_size]));
                    }
                }
            }
            dest_buf_offset += single_output_image_size;
        }
    }
    _convert_time.end();
    return Status::OK;
}
