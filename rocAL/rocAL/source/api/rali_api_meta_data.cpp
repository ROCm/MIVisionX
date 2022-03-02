/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

//
// Created by mvx on 3/31/20.
//

#include "commons.h"
#include "context.h"
#include "rali_api.h"

void
RALI_API_CALL raliRandomBBoxCrop(RaliContext p_context, bool all_boxes_overlap, bool no_crop, RaliFloatParam p_aspect_ratio, bool has_shape, int crop_width, int crop_height, int num_attempts, RaliFloatParam p_scaling, int total_num_attempts, int64_t seed)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliRandomBBoxCrop")
    auto context = static_cast<Context*>(p_context);
    FloatParam *aspect_ratio;
    FloatParam *scaling;
    if(p_aspect_ratio == NULL)
    {
        aspect_ratio = ParameterFactory::instance()->create_uniform_float_rand_param(1.0, 1.0);
    }
    else
    {      
    
        aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
    }
    if(p_scaling == NULL)
    {
        scaling = ParameterFactory::instance()->create_uniform_float_rand_param(1.0, 1.0);
    }
    else
    {
        scaling = static_cast<FloatParam*>(p_scaling);        
    }
    context->master_graph->create_randombboxcrop_reader(RandomBBoxCrop_MetaDataReaderType::RandomBBoxCropReader, RandomBBoxCrop_MetaDataType::BoundingBox, all_boxes_overlap, no_crop, aspect_ratio, has_shape, crop_width, crop_height, num_attempts, scaling, total_num_attempts, seed);
}

RaliMetaData
RALI_API_CALL raliCreateLabelReader(RaliContext p_context, const char* source_path) {
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateLabelReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_label_reader(source_path, MetaDataReaderType::FOLDER_BASED_LABEL_READER);

}

RaliMetaData
RALI_API_CALL raliCreateVideoLabelReader(RaliContext p_context, const char* source_path, unsigned sequence_length, unsigned frame_step, unsigned frame_stride, bool file_list_frame_num) {
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateLabelReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_video_label_reader(source_path, MetaDataReaderType::VIDEO_LABEL_READER, sequence_length, frame_step, frame_stride, file_list_frame_num);

}

RaliMetaData
RALI_API_CALL raliCreateCOCOReader(RaliContext p_context, const char* source_path, bool is_output){
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateCOCOReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_coco_meta_data_reader(source_path, is_output, MetaDataReaderType::COCO_META_DATA_READER,  MetaDataType::BoundingBox);
}

RaliMetaData
RALI_API_CALL raliCreateCOCOReaderKeyPoints(RaliContext p_context, const char* source_path, bool is_output, float sigma, unsigned pose_output_width, unsigned pose_output_height){
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateCOCOReaderKeyPoints")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_coco_meta_data_reader(source_path, is_output, MetaDataReaderType::COCO_KEY_POINTS_META_DATA_READER, MetaDataType::KeyPoints, sigma, pose_output_width, pose_output_height);
}

RaliMetaData
RALI_API_CALL raliCreateTFReader(RaliContext p_context, const char* source_path, bool is_output,const char* user_key_for_label, const char* user_key_for_filename)
{    
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateTFReader")
    auto context = static_cast<Context*>(p_context);
    std::string user_key_for_label_str(user_key_for_label);
    std::string user_key_for_filename_str(user_key_for_filename);


    std::map<std::string, std::string> feature_key_map = {
        {"image/class/label", user_key_for_label_str},
        {"image/filename",user_key_for_filename_str}
    };
    return context->master_graph->create_tf_record_meta_data_reader(source_path , MetaDataReaderType::TF_META_DATA_READER , MetaDataType::Label, feature_key_map);}

RaliMetaData
RALI_API_CALL raliCreateTFReaderDetection(RaliContext p_context, const char* source_path, bool is_output,
    const char* user_key_for_label, const char* user_key_for_text, 
    const char* user_key_for_xmin, const char* user_key_for_ymin, const char* user_key_for_xmax, const char* user_key_for_ymax, 
    const char* user_key_for_filename)
{    
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateTFReaderDetection")
    auto context = static_cast<Context*>(p_context);

    std::string user_key_for_label_str(user_key_for_label);
    std::string user_key_for_text_str(user_key_for_text);
    std::string user_key_for_xmin_str(user_key_for_xmin);
    std::string user_key_for_ymin_str(user_key_for_ymin);
    std::string user_key_for_xmax_str(user_key_for_xmax);
    std::string user_key_for_ymax_str(user_key_for_ymax);
    std::string user_key_for_filename_str(user_key_for_filename);

    std::map<std::string, std::string> feature_key_map = {
        {"image/class/label", user_key_for_label_str},
        {"image/class/text", user_key_for_text_str},
        {"image/object/bbox/xmin", user_key_for_xmin_str},
        {"image/object/bbox/ymin", user_key_for_ymin_str},
        {"image/object/bbox/xmax", user_key_for_xmax_str},
        {"image/object/bbox/ymax", user_key_for_ymax_str},
        {"image/filename",user_key_for_filename_str}
    };

    return context->master_graph->create_tf_record_meta_data_reader(source_path , MetaDataReaderType::TF_DETECTION_META_DATA_READER,  MetaDataType::BoundingBox, feature_key_map);
}

RaliMetaData
RALI_API_CALL raliCreateMXNetReader(RaliContext p_context, const char* source_path, bool is_output)
{
    if (!p_context)
        ERR("Invalid rali context passed to raliCreateMXNetReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_mxnet_label_reader(source_path, is_output);

}

RaliMetaData
RALI_API_CALL raliCreateTextFileBasedLabelReader(RaliContext p_context, const char* source_path) {
    
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateTextFileBasedLabelReader")
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->create_label_reader(source_path, MetaDataReaderType::TEXT_FILE_META_DATA_READER);

}

void
RALI_API_CALL raliGetImageName(RaliContext p_context,  char* buf)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetImageName")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.first.size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    for(unsigned int i = 0; i < meta_data_batch_size; i++)
    {
        memcpy(buf, meta_data.first[i].c_str(), meta_data.first[i].size());
        buf += meta_data.first[i].size() * sizeof(char);
    }
}

unsigned
RALI_API_CALL raliGetImageNameLen(RaliContext p_context, int* buf)
{
    unsigned size = 0;
    if (!p_context)
        THROW("Invalid rali context passed to raliGetImageNameLen")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.first.size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    for(unsigned int i = 0; i < meta_data_batch_size; i++)
    {
        buf[i] = meta_data.first[i].size();
        size += buf[i];
    }
    return size;
}

void
RALI_API_CALL raliGetImageId(RaliContext p_context,  int* buf)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetImageId")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.first.size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    for(unsigned int i = 0; i < meta_data_batch_size; i++)
    {
        std::string str_id = meta_data.first[i].erase(0, meta_data.first[i].find_first_not_of('0'));
        buf[i] = stoi(str_id);
    }
}

void
RALI_API_CALL raliGetImageLabels(RaliContext p_context, int* buf)
{
    
    if (!p_context)
        THROW("Invalid rali context passed to raliGetImageLabels")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(!meta_data.second) {
        WRN("No label has been loaded for this output image")
        return;
    }
    size_t meta_data_batch_size = meta_data.second->get_label_batch().size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    memcpy(buf, meta_data.second->get_label_batch().data(),  sizeof(int)*meta_data_batch_size);
}

unsigned
RALI_API_CALL raliGetBoundingBoxCount(RaliContext p_context, int* buf)
{
    unsigned size = 0;
    if (!p_context)
        THROW("Invalid rali context passed to raliGetBoundingBoxCount")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.second->get_bb_labels_batch().size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    if(!meta_data.second)
        THROW("No label has been loaded for this output image")
    for(unsigned i = 0; i < meta_data_batch_size; i++)
    {
        buf[i] = meta_data.second->get_bb_labels_batch()[i].size();
        size += buf[i];
    }
    return size;
}

void
RALI_API_CALL raliGetBoundingBoxLabel(RaliContext p_context, int* buf)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetBoundingBoxLabel")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.second->get_bb_labels_batch().size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return;
    }
    for(unsigned i = 0; i < meta_data_batch_size; i++)
    { 
        unsigned bb_count = meta_data.second->get_bb_labels_batch()[i].size();
        memcpy(buf, meta_data.second->get_bb_labels_batch()[i].data(),  sizeof(int) * bb_count);
        buf += bb_count;
    }
}

void
RALI_API_CALL raliGetOneHotImageLabels(RaliContext p_context, int* buf, int numOfClasses)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetOneHotImageLabels")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(!meta_data.second) {
        WRN("No label has been loaded for this output image")
        return;
    }
    size_t meta_data_batch_size = meta_data.second->get_label_batch().size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))

    int labels_buf[meta_data_batch_size];
    int one_hot_encoded[meta_data_batch_size*numOfClasses];
    memset(one_hot_encoded, 0, sizeof(int) * meta_data_batch_size * numOfClasses);
    memcpy(labels_buf, meta_data.second->get_label_batch().data(),  sizeof(int)*meta_data_batch_size);
    
    for(uint i = 0; i < meta_data_batch_size; i++)
    {
        int label_index =  labels_buf[i];
        if (label_index >0 && label_index<= numOfClasses )
        {
        one_hot_encoded[(i*numOfClasses)+label_index-1]=1;

        }
        else if(label_index == 0)
        {
          one_hot_encoded[(i*numOfClasses)+numOfClasses-1]=1;  
        }

    }
    memcpy(buf,one_hot_encoded, sizeof(int) * meta_data_batch_size * numOfClasses);

}


void
RALI_API_CALL raliGetBoundingBoxCords(RaliContext p_context, float* buf)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetBoundingBoxCords")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.second->get_bb_cords_batch().size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return;
    }
    for(unsigned i = 0; i < meta_data_batch_size; i++)
    { 
        unsigned bb_count = meta_data.second->get_bb_cords_batch()[i].size();
        memcpy(buf, meta_data.second->get_bb_cords_batch()[i].data(), bb_count * sizeof(BoundingBoxCord));
        buf += (bb_count * 4);
    }
}

void
RALI_API_CALL raliGetImageSizes(RaliContext p_context, int* buf)
{   
    if (!p_context)
        THROW("Invalid rali context passed to raliGetImageSizes")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.second->get_img_sizes_batch().size();


    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return;
    }
    for(unsigned i = 0; i < meta_data_batch_size; i++)
    { 
        memcpy(buf, &(meta_data.second->get_img_sizes_batch()[i]), sizeof(ImgSize));
        buf += 2;
    }
}

RaliMetaData
RALI_API_CALL raliCreateTextCifar10LabelReader(RaliContext p_context, const char* source_path, const char* file_prefix) {
    
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateTextCifar10LabelReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_cifar10_label_reader(source_path, file_prefix);

}

void
RALI_API_CALL raliGetSequenceStartFrameNumber(RaliContext p_context,  unsigned int* buf)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetSequenceStartFrameNumber")
    auto context = static_cast<Context*>(p_context);
    std::vector<size_t> sequence_start_frame;
    context->master_graph->sequence_start_frame_number(sequence_start_frame);
    std::copy(sequence_start_frame.begin(), sequence_start_frame.end(), buf);
}

void
RALI_API_CALL raliGetSequenceFrameTimestamps(RaliContext p_context,  float* buf)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetSequenceFrameTimestamps")
    auto context = static_cast<Context*>(p_context);
    std::vector<std::vector<float>> sequence_frame_timestamps;
    context->master_graph->sequence_frame_timestamps(sequence_frame_timestamps);
    auto sequence_length = sequence_frame_timestamps[0].size();
    for (unsigned int i = 0; i < sequence_frame_timestamps.size(); i++)
    {
        std::copy(sequence_frame_timestamps[i].begin(), sequence_frame_timestamps[i].end(), buf);
        buf = buf + sequence_length;
    }
}

void RALI_API_CALL raliBoxEncoder(RaliContext p_context, std::vector<float>& anchors, float criteria,
                                  std::vector<float> &means, std::vector<float> &stds, bool offset, float scale)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliBoxEncoder")
    auto context = static_cast<Context *>(p_context);
    context->master_graph->box_encoder(anchors, criteria, means, stds, offset, scale);
}

void 
RALI_API_CALL raliCopyEncodedBoxesAndLables(RaliContext p_context, float* boxes_buf, int* labels_buf)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliCopyEncodedBoxesAndLables")
    auto context = static_cast<Context *>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.second->get_bb_labels_batch().size();
    if (context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != " + TOSTR(context->user_batch_size()))
    if (!meta_data.second)
    {
        WRN("No encoded labels and bounding boxes has been loaded for this output image")
        return;
    }
    unsigned sum = 0;
    unsigned sum_bb_count[meta_data_batch_size];
    for (unsigned i = 0; i < meta_data_batch_size; i++)
    {
        sum_bb_count[i] = sum;
        sum = sum + meta_data.second->get_bb_labels_batch()[i].size();
    }
    // copy labels buffer & bboxes buffer parallely
    #pragma omp parallel for
    for (unsigned i = 0; i < meta_data_batch_size; i++)
    {
        unsigned bb_count = meta_data.second->get_bb_labels_batch()[i].size();
        int *temp_labels_buf = labels_buf + sum_bb_count[i];
        float *temp_bbox_buf = boxes_buf + (sum_bb_count[i] * 4);
        memcpy(temp_labels_buf, meta_data.second->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
        memcpy(temp_bbox_buf, meta_data.second->get_bb_cords_batch()[i].data(), sizeof(BoundingBoxCord) * bb_count);
    }
}

void
RALI_API_CALL raliGetJointsDataPtr(RaliContext p_context, RaliJointsData **joints_data)
{  
    if (!p_context)
        THROW("Invalid rali context passed to raliGetBoundingBoxCords")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.second->get_joints_data_batch().center_batch.size();

    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return;
    }

    *joints_data = (RaliJointsData *)(&(meta_data.second->get_joints_data_batch()));
}

