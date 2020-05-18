//
// Created by mvx on 3/31/20.
//

#include "commons.h"
#include "context.h"
#include "rali_api.h"


RaliMetaData
RALI_API_CALL raliCreateLabelReader(RaliContext p_context, const char* source_path) {
    auto context = static_cast<Context*>(p_context);
    if (!context)
        THROW("Invalid rali context passed to raliCreateLabelReader")

    return context->master_graph->create_label_reader(source_path, MetaDataReaderType::FOLDER_BASED_LABEL_READER);

}

RaliMetaData
RALI_API_CALL raliCreateCOCOReader(RaliContext p_context, const char* source_path, bool is_output){
    auto context = static_cast<Context*>(p_context);
    if (!context)
        THROW("Invalid rali context passed to raliCreateLabelReader")

    return context->master_graph->create_coco_meta_data_reader(source_path, is_output);

}

RaliMetaData
RALI_API_CALL raliCreateTFReader(RaliContext p_context, const char* source_path, bool is_output){
    auto context = static_cast<Context*>(p_context);
    if (!context)
        THROW("Invalid rali context passed to raliCreateTFReader")

    return context->master_graph->create_tf_record_meta_data_reader(source_path);

}

RaliMetaData
RALI_API_CALL raliCreateTextFileBasedLabelReader(RaliContext p_context, const char* source_path) {
    auto context = static_cast<Context*>(p_context);
    if (!context)
        THROW("Invalid rali context passed to raliCreateTextFileBasedLabelReader")

    return context->master_graph->create_label_reader(source_path, MetaDataReaderType::TEXT_FILE_META_DATA_READER);

}


void
RALI_API_CALL raliGetImageName(RaliContext p_context,  char* buf, unsigned image_idx)
{
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(image_idx >= meta_data.first.size())
        THROW("Image idx is out of batch size range")
    memcpy(buf, meta_data.first[image_idx].c_str(), meta_data.first[image_idx].size());
}

unsigned
RALI_API_CALL raliGetImageNameLen(RaliContext p_context, unsigned image_idx)
{
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(image_idx >= meta_data.first.size())
        THROW("Image idx is out of batch size range")
    return meta_data.first[image_idx].size();
}

void
RALI_API_CALL raliGetImageLabels(RaliContext p_context, int* buf)
{
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
RALI_API_CALL raliGetBoundingBoxCount(RaliContext p_context, unsigned image_idx )
{
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return 0;
    }
    return meta_data.second->get_bb_labels_batch()[image_idx].size();
}

void
RALI_API_CALL raliGetBoundingBoxLabel(RaliContext p_context, int* buf, unsigned image_idx )
{
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return;
    }
    unsigned bb_count = meta_data.second->get_bb_labels_batch()[image_idx].size();
    memcpy(buf, meta_data.second->get_bb_labels_batch()[image_idx].data(),  sizeof(int)*bb_count);
}

void
RALI_API_CALL raliGetBoundingBoxCords(RaliContext p_context, float* buf, unsigned image_idx )
{
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();

    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return;
    }
    auto ptr = buf;
    memcpy(ptr,meta_data.second->get_bb_cords_batch()[image_idx].data(), meta_data.second->get_bb_cords_batch()[image_idx].size() * sizeof(BoundingBoxCord));
    ptr += sizeof(BoundingBoxCord)*sizeof(float);
}

RaliMetaData
RALI_API_CALL raliCreateTextCifar10LabelReader(RaliContext p_context, const char* source_path, const char* file_prefix) {
    auto context = static_cast<Context*>(p_context);
    if (!context)
        THROW("Invalid rali context passed to raliCreateTextFileBasedLabelReader")

    return context->master_graph->create_cifar10_label_reader(source_path, file_prefix);

}

