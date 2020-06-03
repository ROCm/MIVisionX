/*
Copyright (c) 2015 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include "vx_amd_media.h"
#include "kernels.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <string>
#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

// OpenCL configuration
#define DUMP_DECODED_FRAME         0

#if DUMP_DECODED_FRAME
FILE *fpIn;
#endif

//#if DECODE_ENABLE_OPENCL
#if __APPLE__
#include <opencl.h>
#else
#include <CL/cl.h>
#endif
//#endif

#define DECODE_BUFFER_POOL_SIZE    2  // number of buffers in decoder queue: keep it atleast 2

typedef struct {
	vx_uint32 size;
	vx_uint32 type;
} AuxDataContainerHeader;

typedef struct {
	AuxDataContainerHeader h0;
	int outputFrameCount;
	int reserved[3];
	int64_t cpuTimestamp;
} LoomIoMediaDecoderAuxInfo;

class CLoomIoMediaDecoder {
public:
	CLoomIoMediaDecoder(vx_node node, vx_uint32 mediaCount, const char inputMediaFiles[], vx_uint32 width, vx_uint32 height, vx_df_image format, vx_uint32 stride, vx_uint32 offset);
	~CLoomIoMediaDecoder();
	vx_status Initialize();
	vx_status ProcessFrame(vx_image output, vx_array aux_data);
    vx_status SetRepeatMode(vx_int32 bRepeat);
    vx_status SetEnableUserBufferOpenCLMode(vx_bool bEnable);

protected:
	typedef enum { cmd_abort, cmd_decode } command;
	void DecodeLoop(int mediaIndex);
	void PushCommand(int mediaIndex, command cmd);
	void PushAck(int mediaIndex, int ack);
	command PopCommand(int mediaIndex);
	int PopAck(int mediaIndex);
    void PushFrame(int mediaIndex, AVFrame *frame);
    AVFrame * PopFrame(int mediaIndex);

private:
	vx_node node;
	int mediaCount;
	std::string inputMediaFiles;
    int width, stride;      // width and stride of output buffers
	int height;
	vx_df_image format;
	int decoderImageHeight;
    int clStride, clOffset;
    int offset;
    AVPixelFormat outputFormat, decoderFormat;
	vx_uint8 * decodeBuffer[DECODE_BUFFER_POOL_SIZE];
    vx_bool m_enableUserBufferOpenCL;
//#if DECODE_ENABLE_OPENCL
	cl_mem mem[DECODE_BUFFER_POOL_SIZE];
	cl_command_queue cmdq;
//#endif
	std::vector<std::string> inputMediaFileName;
    std::vector<int> useVaapi;
    std::vector<AVHWDeviceType> hwDeviceType;
	std::vector<AVFormatContext *> inputMediaFormatContext;
	std::vector<AVInputFormat *> inputMediaFormat;
	std::vector<AVCodecContext *> videoCodecContext;
	std::vector<SwsContext *> conversionContext;
    std::vector<std::deque<AVFrame *>> queueFrames;
    //std::vector<AVFrame *> swVideoFrame;
	std::vector<int> videoStreamIndex;
    std::vector<std::mutex> mutexCmd, mutexAck, mutexFrame;
    std::vector<std::condition_variable> cvCmd, cvAck, cvFrame;
	std::vector<std::deque<command>> queueCmd;
	std::vector<std::deque<int>> queueAck;
	std::vector<std::thread *> thread;
	std::vector<bool> eof;
	std::vector<int> decodeFrameCount;
    int outputFrameCount;
    std::vector<int> LoopDec;
};

static enum AVPixelFormat hwPixelFormat;

static int hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type, AVBufferRef *hw_device_ctx)
{
    int err = 0;

    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type,
                                      NULL, NULL, 0)) < 0) {
        return err;
    }
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    return err;
}

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts)
{
    const enum AVPixelFormat *p;

    for (p = pix_fmts; *p != -1; p++) {
        if (*p == hwPixelFormat)
            return *p;
    }
    //vxAddLogEntry((vx_reference)node, VX_ERROR_NOT_SUPPORTED, "ERROR: Failed to create specified HW device.\n");
    fprintf(stderr, "ERROR: Failed to get HW surface format.\n");

    return AV_PIX_FMT_NONE;
}

// helper function for spliting streams.
std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

void CLoomIoMediaDecoder::PushCommand(int mediaIndex, CLoomIoMediaDecoder::command cmd)
{
	std::unique_lock<std::mutex> lock(mutexCmd[mediaIndex]);
	queueCmd[mediaIndex].push_front(cmd);
	cvCmd[mediaIndex].notify_one();
}

CLoomIoMediaDecoder::command CLoomIoMediaDecoder::PopCommand(int mediaIndex)
{
	std::unique_lock<std::mutex> lock(mutexCmd[mediaIndex]);
	cvCmd[mediaIndex].wait(lock, [=] { return !queueCmd[mediaIndex].empty(); });
	command cmd = std::move(queueCmd[mediaIndex].back());
	queueCmd[mediaIndex].pop_back();
	return cmd;
}

void CLoomIoMediaDecoder::PushAck(int mediaIndex, int ack)
{
	std::unique_lock<std::mutex> lock(mutexAck[mediaIndex]);
    queueAck[mediaIndex].push_front(ack);
	cvAck[mediaIndex].notify_one();
}

int CLoomIoMediaDecoder::PopAck(int mediaIndex)
{
	std::unique_lock<std::mutex> lock(mutexAck[mediaIndex]);
	cvAck[mediaIndex].wait(lock, [=] { return !queueAck[mediaIndex].empty(); });
	int ack = std::move(queueAck[mediaIndex].back());
	queueAck[mediaIndex].pop_back();
	return ack;
}

void CLoomIoMediaDecoder::PushFrame(int mediaIndex, AVFrame *frame)
{
    std::unique_lock<std::mutex> lock(mutexFrame[mediaIndex]);
    queueFrames[mediaIndex].push_front(frame);
    cvFrame[mediaIndex].notify_one();
}

AVFrame * CLoomIoMediaDecoder::PopFrame(int mediaIndex)
{
    std::unique_lock<std::mutex> lock(mutexFrame[mediaIndex]);
    cvFrame[mediaIndex].wait(lock, [=] { return !queueFrames[mediaIndex].empty(); });
    AVFrame *frame = std::move(queueFrames[mediaIndex].back());
    queueFrames[mediaIndex].pop_back();
    return frame;
}


CLoomIoMediaDecoder::CLoomIoMediaDecoder(vx_node node_, vx_uint32 mediaCount_, const char inputMediaFiles_[], vx_uint32 width_, vx_uint32 height_, vx_df_image format_, vx_uint32 stride_, vx_uint32 offset_)
	: node{ node_ }, inputMediaFiles(inputMediaFiles_), mediaCount{ static_cast<int>(mediaCount_) }, width{ static_cast<int>(width_) },
      height{ static_cast<int>(height_) }, format{ format_ }, clStride{ static_cast<int>(stride_) }, clOffset{ static_cast<int>(offset_) },
      decoderImageHeight{ static_cast<int>(height_ / ((mediaCount_ <= 1) ? 1 : mediaCount_)) }, outputFormat{ AV_PIX_FMT_UYVY422 }, outputFrameCount{ 0 },
      inputMediaFileName(mediaCount_), inputMediaFormatContext(mediaCount_), inputMediaFormat(mediaCount_),
      videoCodecContext(mediaCount_), conversionContext(mediaCount_), videoStreamIndex(mediaCount_),
	  mutexCmd(mediaCount_), cvCmd(mediaCount_), queueCmd(mediaCount_), mutexAck(mediaCount_), cvAck(mediaCount_), queueAck(mediaCount_),
      thread(mediaCount_), eof(mediaCount_), decodeFrameCount(mediaCount_), useVaapi(mediaCount_), mutexFrame(mediaCount_), cvFrame(mediaCount_), queueFrames(mediaCount_), LoopDec(mediaCount_)
{
	memset(decodeBuffer, 0, sizeof(decodeBuffer));
    for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
        inputMediaFormat[mediaIndex] = NULL;
        videoCodecContext[mediaIndex] = NULL;
        inputMediaFormatContext[mediaIndex] = NULL;
        LoopDec[mediaIndex] = 0;
    }
    m_enableUserBufferOpenCL = false;   // use host buffers by default
    memset(mem, 0, sizeof(mem));
    cmdq = nullptr;
    // initialize freq inside GetTimeInMicroseconds()
	GetTimeInMicroseconds();    
#if DUMP_DECODED_FRAME
    fpIn = fopen("decoder_dump.yuv", "wb");
#endif
}

CLoomIoMediaDecoder::~CLoomIoMediaDecoder()
{
	// terminate the thread
	for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
		if (thread[mediaIndex]) {
			PushCommand(mediaIndex, cmd_abort);
			while (!eof[mediaIndex]) {
				if (PopAck(mediaIndex) < 0)
					break;
			}
			thread[mediaIndex]->join();
			delete thread[mediaIndex];
		}
	}

	// release buffers
    if (m_enableUserBufferOpenCL && cmdq) clReleaseCommandQueue(cmdq);

    for (int i = 0; i < DECODE_BUFFER_POOL_SIZE; i++) {
        if (m_enableUserBufferOpenCL && mem[i]) clReleaseMemObject(mem[i]);
		if (decodeBuffer[i]) aligned_free(decodeBuffer[i]);
	}

	// release media resources
	for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
        //if (swVideoFrame[mediaIndex]) av_frame_free(&videoFrame[mediaIndex]);
		if (conversionContext[mediaIndex]) av_free(conversionContext[mediaIndex]);
		if (inputMediaFormat[mediaIndex]) av_free(inputMediaFormat[mediaIndex]);
        if (videoCodecContext[mediaIndex]->hw_device_ctx) av_buffer_unref(&videoCodecContext[mediaIndex]->hw_device_ctx);
        if (videoCodecContext[mediaIndex]) av_free(videoCodecContext[mediaIndex]);
		if (inputMediaFormatContext[mediaIndex]) av_free(inputMediaFormatContext[mediaIndex]);
	}
#if DUMP_DECODED_FRAME
    if (fpIn) fclose(fpIn);
#endif
}

vx_status CLoomIoMediaDecoder::SetRepeatMode(vx_int32 bRepeat)
{
    for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
        LoopDec[mediaIndex] = bRepeat;
    }
    return VX_SUCCESS;
}

vx_status CLoomIoMediaDecoder::SetEnableUserBufferOpenCLMode(vx_bool bEnable)
{
    m_enableUserBufferOpenCL = bEnable;
    return VX_SUCCESS;
}


vx_status CLoomIoMediaDecoder::Initialize()
{
	// check for valid image type support and get stride in bytes (aligned to 16-byte boundary)
    if (format == VX_DF_IMAGE_NV12) {
        outputFormat = AV_PIX_FMT_NV12;
        stride = width;
    }
    else if (format == VX_DF_IMAGE_UYVY) {
		outputFormat = AV_PIX_FMT_UYVY422;
        stride = width*2;
    }
	else if (format == VX_DF_IMAGE_YUYV) {
		outputFormat = AV_PIX_FMT_YUYV422;
        stride = width*2;
    }
	else if (format == VX_DF_IMAGE_RGB) {
		outputFormat = AV_PIX_FMT_RGB24;
        stride = width*3;
    }
	else {
		vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_FORMAT, "ERROR: output image format %4.4s not supported", &format);
		return VX_ERROR_INVALID_FORMAT;
	}
	// check for validity of media count
	if (mediaCount < 1 || (height % mediaCount) != 0) {
		vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "ERROR: invalid mediaCount (%d) value", mediaCount);
		return VX_ERROR_INVALID_VALUE;
	}

	// get media count and filenames
	if (!inputMediaFiles.compare(inputMediaFiles.size() - 4, 4, ".txt")) {
		// read media filenames from text file
        std::ifstream infile(inputMediaFiles.c_str());
        std::string line;
        int mCount = 0;
        while(std::getline(infile, line) && mCount < mediaCount) {
            std::vector<std::string> streaminfo = split(line, ':');
            if (streaminfo.size() != 2) {
                vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: invalid input file format");
                return VX_ERROR_INVALID_LINK;
            }
            inputMediaFileName[mCount] = streaminfo[0];
            useVaapi[mCount++]         = atoi(streaminfo[1].c_str());
        }
	}
    else if (!inputMediaFiles.empty()) {
		// generate media filenames
        // split the string using ','
        std::vector<std::string> mediainfo = split(inputMediaFiles, ',');
        unsigned int mCount = mediainfo.size();
        if (mCount > mediaCount) mCount = mediaCount;
        for (int mediaIndex = 0; mediaIndex < mCount; mediaIndex++) {
            std::vector<std::string> streaminfo = split(mediainfo[mediaIndex], ':');
            if (streaminfo.size() != 2) {
                vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: invalid input file format");
                return VX_ERROR_INVALID_LINK;
            }
            inputMediaFileName[mediaIndex] = streaminfo[0];
            useVaapi[mediaIndex]           = atoi(streaminfo[1].c_str());
            //printf("mediaindex: %d inputMediaFileName: %s useVaapi: %d\n", mediaIndex, inputMediaFileName[mediaIndex].c_str(),  useVaapi[mediaIndex]);
        }
	}
	else {
        vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: invalid input format");
        return VX_ERROR_INVALID_LINK;
	}

	// open media file and initialize codec
	ERROR_CHECK_STATUS(initialize_ffmpeg());
	for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
		const char * mediaFileName = inputMediaFileName[mediaIndex].c_str();
		AVFormatContext * formatContext = nullptr;
		AVInputFormat * inputFormat = nullptr;
        AVCodec *decoder = NULL;
        AVStream *video = NULL;
        AVCodecContext * codecContext = nullptr;
        AVBufferRef *hw_device_ctx = NULL;
        int videostream;

        // find if hardware decode is available
        AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
        if (useVaapi[mediaIndex]) {
            hw_type = av_hwdevice_find_type_by_name("vaapi");
            if (hw_type == AV_HWDEVICE_TYPE_NONE) {
                vx_status status = VX_FAILURE;
                vxAddLogEntry((vx_reference)node, status, "ERROR: vaapi is not supported for this device\n");
                return status;
            }
            printf("Found vaapi device for %d\n", mediaIndex);
        }
        int err = avformat_open_input(&formatContext, mediaFileName, inputFormat, nullptr);
        if (err) {
            vx_status status = VX_FAILURE;
            vxAddLogEntry((vx_reference)node, status, "ERROR: avformat_open_input(%s) failed (%x)\n", mediaFileName, AVERROR(err));
            return status;
        }
        inputMediaFormatContext[mediaIndex] = formatContext;
		inputMediaFormat[mediaIndex] = inputFormat;
		err = avformat_find_stream_info(formatContext, nullptr);
		if (err) {
			vx_status status = VX_FAILURE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: avformat_find_stream_info() for %s failed (%d)\n", mediaFileName, err);
			return status;
		}        
        // find the video stream information
        err = av_find_best_stream(formatContext, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
        if (err < 0) {
            vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "ERROR: no video found in %s", mediaFileName);
            return VX_ERROR_INVALID_VALUE;
        }
        videostream = err;

        if (!useVaapi[mediaIndex]) {
            unsigned int streamIndex = -1;
            for (unsigned int si = 0; si < formatContext->nb_streams; si++) {
                AVCodecContext * vcc = formatContext->streams[si]->codec;
                if (vcc->codec_type == AVMEDIA_TYPE_VIDEO) {
                    // pick video stream index with larger dimensions
                    //printf("Using sw decoding: Found Video stream index:%d codecContext:%p\n", si, vcc);
                    if (!codecContext) {
                        codecContext = vcc;
                        streamIndex = si;
                    }
                    else if ((vcc->width > codecContext->width) && (vcc->height > codecContext->height)) {
                        codecContext = vcc;
                        streamIndex = si;
                    }
                }
            }
            if (!codecContext) {
                vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "ERROR: no video found in %s", mediaFileName);
                return VX_ERROR_INVALID_VALUE;
            }
        } else
        {
            // for hardware accelerated decoding, find config
            for (int i = 0; ; i++) {
                const AVCodecHWConfig *config = avcodec_get_hw_config(decoder, i);
                if (!config) {
                    vx_status status = VX_FAILURE;
                    vxAddLogEntry((vx_reference)node, status, "ERROR: decoder %s doesn't support device_type %s\n", decoder->name, av_hwdevice_get_type_name(hw_type) );
                    return status;
                }
                if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
                    config->device_type == hw_type) {
                    hwPixelFormat = config->pix_fmt;
                    break;
                }
            }
            if (!(codecContext = avcodec_alloc_context3(decoder))){
                vxAddLogEntry((vx_reference)node, VX_ERROR_NO_MEMORY, "ERROR: can't alloc codec context\n");
                return VX_ERROR_NO_MEMORY;
            }
        }
        videoCodecContext[mediaIndex] = codecContext;
        videoStreamIndex[mediaIndex] = videostream;
        video = formatContext->streams[videostream];
        decoderFormat = codecContext->pix_fmt;
        if (avcodec_parameters_to_context(codecContext, video->codecpar) < 0)
            return -1;
        if (useVaapi[mediaIndex]) {
            codecContext->get_format  = get_hw_format;
            if (hw_decoder_init(codecContext, hw_type, hw_device_ctx) < 0) {
                vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: Failed to create specified HW device.\n");
                return VX_FAILURE;
            }
            decoderFormat = AV_PIX_FMT_NV12; // nv12 for vaapi
        }

        ERROR_CHECK_STATUS(avcodec_open2(codecContext, decoder, nullptr));
		SwsContext * swsContext = NULL;
        if ((outputFormat != decoderFormat) || (codecContext->width != width) || (codecContext->height != decoderImageHeight)) {
            swsContext = sws_getContext(codecContext->width, codecContext->height, decoderFormat, width, decoderImageHeight, outputFormat, SWS_BILINEAR, NULL, NULL, NULL);
			ERROR_CHECK_NULLPTR(swsContext);
            printf("OK created sws context src: <%d %d %d> dst: <%d %d %d>\n", codecContext->width, codecContext->height, decoderFormat, width, decoderImageHeight, outputFormat);
        }
		conversionContext[mediaIndex] = swsContext;
#if 0
        AVFrame * frame = NULL, *sw_frame = NULL;
        if (!(frame = av_frame_alloc()) || !(sw_frame = av_frame_alloc())) {
            err = AVERROR(ENOMEM);
            vxAddLogEntry((vx_reference)node, VX_ERROR_NO_MEMORY, "ERROR: Can not alloc frame(%d)", err);
            return VX_ERROR_NO_MEMORY;
        }
		videoFrame[mediaIndex] = frame;
        swVideoFrame[mediaIndex] = sw_frame;
#endif
        // debug log
		vxAddLogEntry((vx_reference)node, VX_SUCCESS, "INFO: reading %dx%d into slice#%d from %s", width, decoderImageHeight, mediaIndex, mediaFileName);
	}

    if (m_enableUserBufferOpenCL) {
	// allocate OpenCL decode buffers
        cl_context context = nullptr;
        ERROR_CHECK_STATUS(vxQueryContext(vxGetContext((vx_reference)node), VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &context, sizeof(context)));
        cl_device_id device_id = nullptr;
        ERROR_CHECK_STATUS(clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, nullptr));
    #if defined(CL_VERSION_2_0)
        cmdq = clCreateCommandQueueWithProperties(context, device_id, 0, nullptr);
    #else
        cmdq = clCreateCommandQueue(context, device_id, 0, nullptr);
    #endif
        ERROR_CHECK_NULLPTR(cmdq);
        for (int i = 0; i < DECODE_BUFFER_POOL_SIZE; i++) {
            mem[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, clOffset + clStride * height, nullptr, nullptr);
            ERROR_CHECK_NULLPTR(mem[i]);
        }
    }

	// start decoder thread and wait until first frame is decoded
	outputFrameCount = 0;
	for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
		decodeFrameCount[mediaIndex] = 0;
		eof[mediaIndex] = false;
	}
	for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
		thread[mediaIndex] = new std::thread(&CLoomIoMediaDecoder::DecodeLoop, this, mediaIndex);
		ERROR_CHECK_NULLPTR(thread[mediaIndex]);
		// initial ACK to inform producer for readiness
		for (int i = 1; i < DECODE_BUFFER_POOL_SIZE; i++)
			PushCommand(mediaIndex, cmd_decode);
	}
    // do we need to do this here??
//	for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
//		PopAck(mediaIndex);
//	}

	return VX_SUCCESS;
}

static int frame_num = 0;

vx_status CLoomIoMediaDecoder::ProcessFrame(vx_image output, vx_array aux_data)
{
	// continue decoding another frame
	for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
		PushCommand(mediaIndex, cmd_decode);
	}
	// wait until next frame is available
	for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
		int ack = PopAck(mediaIndex);
		if ((ack < 0) || eof[mediaIndex]) {
			// nothing to process, so abandon the graph execution
			return VX_ERROR_GRAPH_ABANDONED;
		}
	}
	// set aux data
	if (aux_data) {
		// construct aux data
		LoomIoMediaDecoderAuxInfo haux = { 0 };
		haux.h0.size = sizeof(LoomIoMediaDecoderAuxInfo);
        haux.h0.type = AMDOVX_KERNEL_AMD_MEDIA_DECODE;
		haux.outputFrameCount = outputFrameCount;
		haux.cpuTimestamp = GetTimeInMicroseconds();
		// set aux data
		ERROR_CHECK_STATUS(vxTruncateArray(aux_data, 0));
		ERROR_CHECK_STATUS(vxAddArrayItems(aux_data, sizeof(haux), &haux, sizeof(uint8_t)));
	}
    if (m_enableUserBufferOpenCL) {
        // set the openCL buffer pointer for output buffer
        int bufId = outputFrameCount % DECODE_BUFFER_POOL_SIZE; outputFrameCount++;
        ERROR_CHECK_STATUS(vxSetImageAttribute(output, VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &mem[bufId], sizeof(cl_mem)));
    } else {
        for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
            AVFrame *frame = PopFrame(mediaIndex);       // assuming only one stream to decode
            if (conversionContext[mediaIndex] != NULL) {
                vx_rectangle_t rect = { 0, (vx_uint32)(mediaIndex * decoderImageHeight), (vx_uint32)width, (vx_uint32)(mediaIndex * decoderImageHeight + decoderImageHeight) };
                vx_map_id map_id, map_id1;
                vx_imagepatch_addressing_t addr = {0};
                uint8_t * ptr = nullptr;

                uint8_t *dst_data[4] = {0};
                int dst_linesize[4] = {0};
                ERROR_CHECK_STATUS(vxMapImagePatch(output, &rect, 0, &map_id, &addr, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
                dst_data[0] = ptr;
                dst_linesize[0] = addr.stride_y;
                if (outputFormat == AV_PIX_FMT_NV12) {
                    uint8_t *ptr_uv = nullptr;
                    vx_imagepatch_addressing_t addr1 = {0};
                    ERROR_CHECK_STATUS(vxMapImagePatch(output, &rect, 1, &map_id1, &addr1, (void **)&ptr_uv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
                    dst_data[1] = ptr_uv;
                    dst_linesize[1]  = addr1.stride_y;
                }
                // do sws_scale
                int ret = sws_scale(conversionContext[mediaIndex], frame->data, frame->linesize, 0, frame->height, dst_data, dst_linesize);
                if (ret < decoderImageHeight) {
                    fprintf(stderr, "Error in output image scaling using sws_scale\n");
                    return VX_FAILURE;
                }
                #if DUMP_DECODED_FRAME
                if (fpIn){
                    fwrite(dst_data[0], 1, decoderImageHeight*dst_linesize[0], fpIn);
                    if (outputFormat == AV_PIX_FMT_NV12)
                        fwrite(dst_data[1], 1, (decoderImageHeight>>1)*dst_linesize[1], fpIn);
                }
                #endif
                // commit image patch
                ERROR_CHECK_STATUS(vxUnmapImagePatch(output, map_id));
                if (outputFormat == AV_PIX_FMT_NV12) ERROR_CHECK_STATUS(vxUnmapImagePatch(output, map_id1));
            } else {
                // copy AV frame to output
                vx_rectangle_t rect = { 0, (vx_uint32)(mediaIndex * decoderImageHeight), (vx_uint32)width, (vx_uint32)decoderImageHeight };
                vx_rectangle_t rect1 = { 0, (vx_uint32)(mediaIndex * (decoderImageHeight>>1)), (vx_uint32)width, (vx_uint32)(decoderImageHeight) }; // UV
                vx_imagepatch_addressing_t addr = { 0 };
                addr.stride_x = stride / width;
                addr.stride_y = stride;
                ERROR_CHECK_STATUS(vxCopyImagePatch(output, &rect, 0, &addr, frame->data[0], VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                ERROR_CHECK_STATUS(vxCopyImagePatch(output, &rect1, 1, &addr, frame->data[1], VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            }
            av_frame_free(&frame);
        }
    }
    frame_num++;
    return VX_SUCCESS;
}

void CLoomIoMediaDecoder::DecodeLoop(int mediaIndex)
{
	// decode loop
	AVPacket avpkt = { 0 };
	int status;

	for (command cmd; !eof[mediaIndex] && ((cmd = PopCommand(mediaIndex)) != cmd_abort);) {
		int gotPicture = 0;
		while (!gotPicture && !eof[mediaIndex]) 
		{
			for (;;) {
				status = av_read_frame(inputMediaFormatContext[mediaIndex], &avpkt);
				if (status < 0) {
                    if ((status == AVERROR_EOF) && LoopDec[mediaIndex]) {
                        auto stream = inputMediaFormatContext[mediaIndex]->streams[videoStreamIndex[mediaIndex]];
                        avio_seek(inputMediaFormatContext[mediaIndex]->pb, 0, SEEK_SET);
                        avformat_seek_file(inputMediaFormatContext[mediaIndex], videoStreamIndex[mediaIndex], 0, 0, stream->duration, 0);
                        //printf("Reached EOF: Looping\n");
                        continue;
                    }
                    // no more packets: need to still flush decoder till we get eof
				    avpkt.data = NULL;
				    avpkt.size = 0;
				    status = avcodec_send_packet(videoCodecContext[mediaIndex], &avpkt);
				    if (status < 0) {
                        vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: Sending packet to video decoder");
				    }
                    eof[mediaIndex] = true;
                    PushAck(mediaIndex, -1);
                    av_packet_unref(&avpkt);
                    return;
				}
				else if (avpkt.stream_index == videoStreamIndex[mediaIndex]) {
                    // send packet to decoder
				    status = avcodec_send_packet(videoCodecContext[mediaIndex], &avpkt);
				    if (status < 0) {
                        vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: Sending packet to video decoder status:%x", AVERROR(status));
                        return;
				    }
				   break;
				}
			}
            AVFrame *frame = NULL, *sw_frame = NULL, *tmp_frame = NULL;
            if (!(frame = av_frame_alloc()) || !(sw_frame = av_frame_alloc())) {
                vxAddLogEntry((vx_reference)node, VX_ERROR_NO_MEMORY, "ERROR: Can not alloc frame(%d)");
                return;
	        }
            int status = avcodec_receive_frame(videoCodecContext[mediaIndex], frame);
            if (status == AVERROR(EAGAIN)) {
            	// output not available at this time: continue to send the next frame.
                av_frame_free(&frame);
                av_frame_free(&sw_frame);
                continue;
            } else if (status < 0) {
                vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: avcodec_receive_frame() failed (%x)\n", AVERROR(status));
                eof[mediaIndex] = true;
                PushAck(mediaIndex, -1);
                av_frame_free(&frame);
                av_frame_free(&sw_frame);
                return;
            }
            gotPicture = true;
            if (useVaapi[mediaIndex]) {
                /* retrieve data from GPU to CPU */
                if ((status = av_hwframe_transfer_data(sw_frame, frame, 0)) < 0) {
                    vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: avcodec_receive_frame() failed (%x)\n", AVERROR(status));
                    eof[mediaIndex] = true;
                    PushAck(mediaIndex, -1);
                    av_frame_free(&frame);
                    av_frame_free(&sw_frame);
                    return;
                }
                tmp_frame = sw_frame;
                av_frame_free(&frame);
            } else {
                tmp_frame = frame;
                av_frame_free(&sw_frame);
            }
            if (m_enableUserBufferOpenCL) {
            // do sw_scale for destination format
                int bufId = decodeFrameCount[mediaIndex] % DECODE_BUFFER_POOL_SIZE;
                if (conversionContext[mediaIndex] != NULL) {
                    cl_int err;
                    uint8_t * ptr = nullptr;
                    int mapHeight = (outputFormat == AV_PIX_FMT_NV12)? (decoderImageHeight + (decoderImageHeight>>1)) : decoderImageHeight;
                    void * mapped_ptr = (void *)clEnqueueMapBuffer(cmdq, mem[bufId], CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, clOffset + mediaIndex * mapHeight * clStride, mapHeight * clStride, 0, NULL, NULL, &err);
                    if(err) {
                        fprintf(stderr,"map output for sw_scale: clEnqueueMapBuffer failed (%d)", err);
                        continue;
                    }
                    uint8_t *dst_data[4] = {0};
                    int dst_linesize[4] = {0};
                    dst_data[0] = (uint8_t *)mapped_ptr;
                    dst_linesize[0] = clStride;
                    if (outputFormat == AV_PIX_FMT_NV12) {
                        dst_data[1] = (uint8_t *)mapped_ptr + decoderImageHeight*clStride;
                        dst_linesize[1]  = clStride;
                    }
                    // do sws_scale
                    int ret = sws_scale(conversionContext[mediaIndex], tmp_frame->data, tmp_frame->linesize, 0, tmp_frame->height, dst_data, dst_linesize);
                    if (ret < decoderImageHeight) {
                        fprintf(stderr, "Error in output image scaling using sws_scale\n");
                        continue;
                    }
                    #if DUMP_DECODED_FRAME
                    if (fpIn){
                        fwrite(dst_data[0], 1, decoderImageHeight*dst_linesize[0], fpIn);
                        if (outputFormat == AV_PIX_FMT_NV12)
                            fwrite(dst_data[1], 1, (decoderImageHeight>>1)*dst_linesize[1], fpIn);
                    }
                    #endif
                    // commit image patch
                    err = clEnqueueUnmapMemObject(cmdq, mem[bufId], mapped_ptr, 0, NULL, NULL);
                    if(err) {
                        fprintf(stderr,"map output for sw_scale: clEnqueueMapBuffer failed (%d)", err);
                        continue;
                    }
                    err = clFinish(cmdq);
                    if(err) {
                        fprintf(stderr,"map output for sw_scale: clFinish failed (%d)",  err);
                    }
                } else {
                    // copy AV frame to output
                    cl_int err = clEnqueueWriteBuffer(cmdq, mem[bufId], CL_TRUE, clOffset + mediaIndex * decoderImageHeight * clStride, decoderImageHeight * clStride, tmp_frame->data, 0, nullptr, nullptr);
                    if (err < 0) {
                        vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: clEnqueueWriteBuffer(buf[%d], slice[%d]) failed (%d)\n", bufId, mediaIndex, err);
                        continue;
                    }
                    clFinish(cmdq);
                }
            } else {
                PushFrame(mediaIndex, tmp_frame);
            }
            // update decoded frame count and send ACK
            decodeFrameCount[mediaIndex]++;
            PushAck(mediaIndex, 0);
        }
	}
end:
	// mark eof and send ACK
	eof[mediaIndex] = true;
	PushAck(mediaIndex, -1);
    av_packet_unref(&avpkt);
}


//! \brief The kernel execution.
static vx_status VX_CALLBACK amd_media_decode_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	// get decoder and output image
	CLoomIoMediaDecoder * decoder = nullptr;
	ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &decoder, sizeof(decoder)));
	if (!decoder) return VX_FAILURE;

	return decoder->ProcessFrame((vx_image)parameters[1], (vx_array)parameters[2]);
}

//! \brief The kernel initializer.
static vx_status VX_CALLBACK amd_media_decode_initialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	// get input parameters
	char inputMediaConfig[VX_MAX_STRING_BUFFER_SIZE_AMD];
	vx_uint32 width = 0, height = 0, stride = 0, offset = 0;
	vx_df_image format = VX_DF_IMAGE_VIRT;
	ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[0], inputMediaConfig, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
    vx_bool enableUserBufferOpenCL = false;
    if (parameters[4]) {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &enableUserBufferOpenCL, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER_STRIDE, &stride, sizeof(stride)));
        ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER_OFFSET, &offset, sizeof(offset)));
    }

	// create and initialize decoder
	const char * s = inputMediaConfig;
    vx_uint32 mediaCount = atoi(s);
	while (*s && *s != ',') s++;
	if (mediaCount < 1 || *s != ',') {
        printf("Got Mediacount %d next char %c\n", mediaCount, *s);
        vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "ERROR: invalid ioConfig: %s\nERROR: invalid ioConfig: valid syntax: <mediaCount>,(mediaList.txt|media%%d.mp4)|{file1.mp4,file2.mp4,...}\n", inputMediaConfig);
		return VX_ERROR_INVALID_VALUE;
	}
	if (*s == ',') s++;
    int loop = 0;
    if (parameters[3]) ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &loop, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    CLoomIoMediaDecoder * decoder = new CLoomIoMediaDecoder(node, mediaCount, s, width, height, format, stride, offset);
	ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &decoder, sizeof(decoder)));
    if (parameters[3]){
        ERROR_CHECK_STATUS(decoder->SetRepeatMode(loop));
    }
    if (parameters[4]) {
        ERROR_CHECK_STATUS(decoder->SetEnableUserBufferOpenCLMode(enableUserBufferOpenCL));
    }
    ERROR_CHECK_STATUS(decoder->Initialize());

	return VX_SUCCESS;
}

//! \brief The kernel deinitializer.
static vx_status VX_CALLBACK amd_media_decode_deinitialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	// get decoder
	CLoomIoMediaDecoder * decoder = nullptr;
	ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &decoder, sizeof(decoder)));

	if (decoder) {
		// release the resources
		delete decoder;
	}

	return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK amd_media_decode_validate(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
	// make sure input scalar contains num cameras and media file name
	vx_enum type;
	ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[0], VX_SCALAR_TYPE, &type, sizeof(type)));
	if (type != VX_TYPE_STRING_AMD)
		return VX_ERROR_INVALID_FORMAT;
    // make sure output format is UYVY/YUYV/RGB/NV12
	vx_uint32 width = 0, height = 0;
	vx_df_image format = VX_DF_IMAGE_VIRT;
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
    if (format != VX_DF_IMAGE_UYVY && format != VX_DF_IMAGE_YUYV && format != VX_DF_IMAGE_RGB && format != VX_DF_IMAGE_NV12)
		return VX_ERROR_INVALID_FORMAT;
	// set output image meta
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
    vx_bool enableUserBufferOpenCL = false;
    if (parameters[4]) {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &enableUserBufferOpenCL, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_ATTRIBUTE_AMD_ENABLE_USER_BUFFER_OPENCL, &enableUserBufferOpenCL, sizeof(enableUserBufferOpenCL)));
        printf("decoder validate:: set enableUserBufferOpenCL: %d\n", enableUserBufferOpenCL);
    }

	// check aux data parameter
	if (parameters[2]) {
		// make sure data type is UINT8
		vx_enum itemtype = VX_TYPE_INVALID;
		vx_size capacity = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ITEMTYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[2], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity)));
		if (itemtype != VX_TYPE_UINT8)
			return VX_ERROR_INVALID_TYPE;
		// set meta for aux data
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_ARRAY_ITEMTYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity)));
	}

	return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status amd_media_decode_publish(vx_context context)
{
	// add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.amd_media.decode", AMDOVX_KERNEL_AMD_MEDIA_DECODE,
                            amd_media_decode_kernel, 5, amd_media_decode_validate,
                            amd_media_decode_initialize, amd_media_decode_deinitialize);
	ERROR_CHECK_OBJECT(kernel);

	// set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // media config+filename: mediaCount,mediaList.txt|media%d.mp4|{file1.mp4:useVaapi,file2.mp4:useVaapi,...}
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED)); // output image
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_OPTIONAL)); // output auxiliary data (optional)
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL)); // input repeat decoding at eof (optional)
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL)); // input: to set enableUserBufferOpenCL flag

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL amdMediaDecoderNode(vx_graph graph, const char *input_str, vx_image output, vx_array aux_data, vx_int32 loop_decode, vx_bool enable_opencl_output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_input = vxCreateScalar(context, VX_TYPE_STRING_AMD, input_str);
        vx_scalar s_loop = vxCreateScalar(context, VX_TYPE_INT32, &loop_decode);
        vx_scalar s_enable_cl_out = vxCreateScalar(context, VX_TYPE_BOOL, &enable_opencl_output);
        vx_reference params[] = {
            (vx_reference)s_input,
            (vx_reference)output,
            (vx_reference)aux_data,
            (vx_reference)s_loop,
            (vx_reference)s_enable_cl_out,
        };
        if (vxGetStatus((vx_reference)s_input) == VX_SUCCESS) {
            node = createMediaNode(graph, "com.amd.amd_media.decode", params, sizeof(params) / sizeof(params[0])); // added node to graph
            vxReleaseScalar(&s_input);
            vxReleaseScalar(&s_loop);
            vxReleaseScalar(&s_enable_cl_out);
        }
    }
    return node;
}
