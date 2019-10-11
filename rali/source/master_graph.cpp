#include <CL/cl.h>
#include <vx_ext_amd.h>
#include <VX/vx_types.h>
#include <cstring>
#include <sched.h>
#include "master_graph.h"
#include "parameter_factory.h"

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
    stop_processing();
    release();
}

MasterGraph::MasterGraph(size_t batch_size, RaliAffinity affinity, int gpu_id, size_t cpu_threads):
        _ring_buffer(OUTPUT_RING_BUFFER_DEPTH),
        _output_tensor(nullptr),
        _graph(nullptr),
        _affinity(affinity),
        _gpu_id(gpu_id),
        _convert_time("Conversion Time"),
        _batch_size(batch_size),
        _cpu_threads(cpu_threads),
        _process_time("Process Time"),
        _first_run(true),
        _processing(false),
        _in_process_count(0)
{
    try {
        vx_status status;
        _context = vxCreateContext();
        _mem_type = (_affinity == RaliAffinity::GPU) ? RaliMemType::OCL : RaliMemType::HOST;

        auto vx_affinity = get_ago_affinity_info(_affinity, 0, gpu_id);

        if ((status = vxGetStatus((vx_reference) _context)) != VX_SUCCESS)
            THROW("vxCreateContext failed" + TOSTR(status))

        // Setting attribute to run on CPU or GPU should be called before load kernel module
        if ((status = vxSetContextAttribute(_context,
                                            VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY,
                                            &vx_affinity,
                                            sizeof(vx_affinity))) != VX_SUCCESS)
            THROW("vxSetContextAttribute failed " + TOSTR(status))

        // loading OpenVX RPP modules
        if ((status = vxLoadKernels(_context, "vx_rpp")) != VX_SUCCESS)
            THROW("Cannot load OpenVX augmentation extension (vx_rpp), vxLoadKernels failed " + TOSTR(status))
#ifdef RALI_VIDEO
        // loading video decoder modules
        if ((status = vxLoadKernels(_context, "vx_media")) != VX_SUCCESS)
            WRN("Cannot load AMD's OpenVX media extension, video decode functionality will not be available")
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
    if(_first_run)
    {
        _first_run = false;
    } else {
        _ring_buffer.pop(); // Pop previously stored output from the ring buffer
        for (auto &&loader_image : _loader_image)
            loader_image->pop_name();
    }

    _ring_buffer.get_read_buffers();// make sure read buffers are ready, it'll wait here otherwise
    {
        std::unique_lock<std::mutex> lock(_count_lock);
        if (_in_process_count > 0)
            _in_process_count--;
    }
    return MasterGraph::Status::OK;
}

void
MasterGraph::create_single_graph()
{
    // Actual graph creating and calls into adding nodes to graph is deferred and is happening here to enable potential future optimizations
    _graph = std::make_shared<Graph>(_context, _affinity, 0, _gpu_id);
    for(auto node: _nodes)
    {
        // Any image not yet created can be created as virtual image
        for(auto image: node->output())
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
    size_t w = _output_image_info.width();
    size_t h = _output_image_info.height_batch();
    size_t c = _output_image_info.color_plane_count();
    _ring_buffer.init(_mem_type, _device.resources(), w * h * c, _output_images.size());
    create_single_graph();
    start_processing();
    return Status::OK;
}
Image *
MasterGraph::create_loader_output_image(const ImageInfo &info)
{
    /*
    *   NOTE: Output image for a source node needs to be created as a regular (non-virtual) image
    *   NOTE: allocate flag is not set for the create_from_handle function here since image's
    *       context will be swapped with the loader_module's internal buffer
    */
    auto output = new Image(info);

    if( output->create_from_handle(_context, ImageBufferAllocation::none) != 0)
        THROW("Creating output image for JPEG loader failed");

    _internal_images.push_back(output);

    return output;
}

Image *
MasterGraph::create_image(const ImageInfo &info, bool is_output)
{
    auto* output = new Image(info);

    if(is_output)
    {
        if (output->create_from_handle(_context, ImageBufferAllocation::none) != 0)
            THROW("Cannot create the image from handle")

        _output_images.push_back(output);
    }


    return output;
}

void MasterGraph::release()
{
    stop_processing();
    vx_status status;
    if(_graph != nullptr)
        _graph->release();

    if(_context && (status = vxReleaseContext(&_context)) != VX_SUCCESS)
        LOG ("Failed to call vxReleaseContext " + TOSTR(status))

    for(auto& image: _internal_images)
        delete image;// It will call the vxReleaseImage internally in the destructor

    for(auto& image: _output_images)
        delete image;// It will call the vxReleaseImage internally in the destructor


    deallocate_output_tensor();
}

MasterGraph::Status
MasterGraph::update_node_parameters()
{
    for(auto& node: _nodes)
        node->update_parameters();

    return Status::OK;
}

size_t
MasterGraph::output_image_count()
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
    return _output_image_info.height_batch();
}

MasterGraph::Status
MasterGraph::allocate_output_tensor()
{
    // creating a float buffer that can accommodates all output images
    size_t output_float_buffer_size = _output_image_info.width() *
                                      _output_image_info.height_batch() *
                                      _output_image_info.color_plane_count() *
                                      _output_images.size();

    if(_output_image_info.mem_type() == RaliMemType::OCL)
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
    if(_output_image_info.mem_type() == RaliMemType::OCL && _output_tensor != nullptr)
        clReleaseMemObject(_output_tensor );

    return Status::OK;
}

MasterGraph::Status
MasterGraph::reset_loaders()
{

    for(auto& loader_module: _loader_modules)
        loader_module->reset();

    _first_run = true;

    return Status::OK;
}

size_t
MasterGraph::internal_image_count()
{
    int ret = -1;
    for(auto& loader_module: _loader_modules) {
        int thisLoaderCount = loader_module->count();
        ret = (ret == -1 ) ? thisLoaderCount :
              ((thisLoaderCount < ret ) ? thisLoaderCount : ret);
    }
    return ret;
}
size_t
MasterGraph::remaining_images_count()
{
    std::unique_lock<std::mutex> lock(_count_lock);
    size_t ret = internal_image_count() + _in_process_count;
    return ret;
}

RaliMemType
MasterGraph::mem_type()
{
    return _mem_type;
}

std::vector<long long unsigned>
MasterGraph::timing()
{
    long long unsigned load_time = 0;
    long long unsigned decode_time = 0;
    for(auto& loader_module: _loader_modules)
    {
        auto ret = loader_module->timing();
        if(ret.size() < 2)
            continue;

        load_time += ret[0];
        decode_time += ret[1];
    }
    return {load_time, decode_time, _process_time.get_timing(), _convert_time.get_timing()};
}


MasterGraph::Status
MasterGraph::copy_output(
        cl_mem out_ptr,
        size_t out_size)
{
    return Status::NOT_IMPLEMENTED;
    _convert_time.start();
    _convert_time.end();
    return Status::OK;
}

#define CHECK_CL_CALL_RET(x) { cl_int ret; ret = x; if( ret != CL_SUCCESS) THROW("ocl call failed "+STR(#x)+" error "+TOSTR(ret)) }

MasterGraph::Status
MasterGraph::copy_out_tensor(float *out_ptr, RaliTensorFormat format, float multiplier0, float multiplier1,
                             float multiplier2, float offset0, float offset1, float offset2, bool reverse_channels)
{
    _convert_time.start();
    // Copies to the output context given by the user
    size_t w = _output_image_info.width();
    size_t h = _output_image_info.height_batch();
    size_t c = _output_image_info.color_plane_count();

    size_t single_output_image_size = w * h * c;

    if(_output_image_info.mem_type() == RaliMemType::OCL)
    {
        // OCL device memory
        cl_int status;

        size_t global_work_size = single_output_image_size;
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
                auto channel_size  = w * h;
                for(unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                    for(unsigned i = 0; i < channel_size; i++)
                        out_ptr[dest_buf_offset+channel_idx+ i*c] =
                                offset[channel_idx] + multiplier[channel_idx]*(reverse_channels ? (float)(in_buffer[i*c+c-channel_idx-1]) : (float)(in_buffer[i*c+channel_idx]));
            }
            if(format == RaliTensorFormat::NCHW)
            {
                auto channel_size  = w * h;
                for(unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                    for(unsigned i = 0; i < channel_size; i++)
                        out_ptr[dest_buf_offset+channel_idx*channel_size + i] =
                                offset[channel_idx] + multiplier[channel_idx]*(reverse_channels ? (float)(in_buffer[c*i+c-channel_idx-1]) : (float)(in_buffer[c*i+channel_idx]));


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
    _convert_time.start();
    // Copies to the output context given by the user
    size_t size = _output_image_info.width() *
                  _output_image_info.height_batch() *
                  _output_image_info.color_plane_count();

    size_t dest_buf_offset = 0;

    if(_output_image_info.mem_type() == RaliMemType::OCL)
    {
        //NOTE: the CL_TRUE flag is only used on the last buffer read call,
        // to avoid unnecessary sequence of synchronizations


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
        // host memory
        memcpy(out_ptr, _ring_buffer.get_host_master_read_buffer(), size * _output_images.size());
    }
    _convert_time.end();
    return Status::OK;
}

void MasterGraph::output_routine()
{
    while(_processing)
    {
        if(internal_image_count() <= 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Randomize parameters
        ParameterFactory::instance()->renew_parameters();
        _process_time.start();
        {
            std::unique_lock<std::mutex> lock(_count_lock);

            // Swap handles on the input images, so that new image is loaded to be processed
            for (auto &&loader_module: _loader_modules)
                if (loader_module->load_next() != LoaderModuleStatus::OK)
                    THROW("Loader module failed to laod next batch of images")

            _in_process_count++;
        }
        auto write_buffers = _ring_buffer.get_write_buffers();

        // Swap handles on the output images, so that new processed image will be written to the a new buffer
        for (size_t idx = 0; idx < _output_images.size(); idx++)
            _output_images[idx]->swap_handle(write_buffers[idx]);

        if (!_processing)
            break;
        update_node_parameters();
        _graph->process();
        _ring_buffer.push(); // After process returns, the spot in the circular buffer can be stored

        _process_time.end();
    }
}

void MasterGraph::start_processing()
{

    _processing = true;
    _output_thread = std::thread(&MasterGraph::output_routine, this);
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(_output_thread.native_handle(), SCHED_FIFO, &params);
#endif
}

void MasterGraph::stop_processing()
{
    _processing = false;
    _ring_buffer.cancel_reading();
    _ring_buffer.cancel_writing();
    try
    {
        _output_thread.join();
    } catch(...)
    {

    }
}
