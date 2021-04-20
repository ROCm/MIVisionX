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

#include "commons.h"
#include "context.h"
#include "rali_api.h"
size_t RALI_API_CALL raliGetImageWidth(RaliImage p_image)
{
    auto image = static_cast<Image*>(p_image);
    return image->info().width();
}
size_t RALI_API_CALL raliGetImageHeight(RaliImage p_image)
{
    auto image = static_cast<Image*>(p_image);
    return image->info().height_batch();
}

size_t RALI_API_CALL raliGetImagePlanes(RaliImage p_image)
{
    auto image = static_cast<Image*>(p_image);
    return image->info().color_plane_count();
}

int RALI_API_CALL raliGetOutputWidth(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->output_width();
}

int RALI_API_CALL raliGetOutputHeight(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->output_height();
}

int RALI_API_CALL raliGetOutputColorFormat(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    auto translate_color_format = [](RaliColorFormat color_format)
    {
        switch(color_format){
            case RaliColorFormat::RGB24:
                return 0;
            case RaliColorFormat::BGR24:
                return 1;
            case RaliColorFormat::U8:
                return 2;
            case RaliColorFormat::RGB_PLANAR:
                return 3;
            default:
                THROW("Unsupported Image type" + TOSTR(color_format))
        }
    };

    return translate_color_format(context->master_graph->output_color_format());
}
size_t RALI_API_CALL raliGetAugmentationBranchCount(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->augmentation_branch_count();
}

size_t  RALI_API_CALL
raliGetRemainingImages(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    size_t count = 0;
    try
    {
        count = context->master_graph->remaining_images_count();
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return count;
}

RaliStatus RALI_API_CALL raliGetStatus(RaliContext p_context)
{
    if(!p_context)
        return RALI_CONTEXT_INVALID;
    auto context = static_cast<Context*>(p_context);

    if(context->no_error())
        return RALI_OK;

    return RALI_RUNTIME_ERROR;
}

const char* RALI_API_CALL raliGetErrorMessage(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    return context->error_msg();
}
TimingInfo
RALI_API_CALL raliGetTimingInfo(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    auto info = context->timing();
    //INFO("shuffle time "+ TOSTR(info.shuffle_time)); to display time taken for shuffling dataset
    return {info.image_read_time, info.image_decode_time, info.image_process_time, info.copy_to_output};
}

RaliMetaData
RALI_API_CALL raliCreateCaffe2LMDBLabelReader(RaliContext p_context, const char* source_path, bool is_output){

    if (!p_context)
        THROW("Invalid rali context passed to raliCreateCaffe2LMDBLabelReader")

    auto context = static_cast<Context*>(p_context);
    return context->master_graph->create_caffe2_lmdb_record_meta_data_reader(source_path , MetaDataReaderType::CAFFE2_META_DATA_READER , MetaDataType::Label);
}

RaliMetaData
RALI_API_CALL raliCreateCaffe2LMDBReaderDetection(RaliContext p_context, const char* source_path, bool is_output){
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateCaffe2LMDBReaderDetection")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_caffe2_lmdb_record_meta_data_reader(source_path , MetaDataReaderType::CAFFE2_DETECTION_META_DATA_READER,  MetaDataType::BoundingBox);

}

RaliMetaData
RALI_API_CALL raliCreateCaffeLMDBLabelReader(RaliContext p_context, const char* source_path){

    if (!p_context)
        THROW("Invalid rali context passed to raliCreateCaffeLMDBLabelReader")
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->create_caffe_lmdb_record_meta_data_reader(source_path , MetaDataReaderType::CAFFE_META_DATA_READER , MetaDataType::Label);
}

RaliMetaData
RALI_API_CALL raliCreateCaffeLMDBReaderDetection(RaliContext p_context, const char* source_path){
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateCaffeLMDBReaderDetection")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_caffe_lmdb_record_meta_data_reader(source_path, MetaDataReaderType::CAFFE_DETECTION_META_DATA_READER,  MetaDataType::BoundingBox);

}

size_t RALI_API_CALL raliIsEmpty(RaliContext p_context)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliIsEmpty")
    auto context = static_cast<Context*>(p_context);
    size_t ret = 0;
    try
    {
        ret = context->master_graph->empty();
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return ret;
}
