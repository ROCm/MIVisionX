/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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


#include <memory>
#include "label_reader_folders.h"
#include "meta_data_reader_factory.h"
#include "exception.h"
#include "coco_meta_data_reader.h"
#include "coco_meta_data_reader_key_points.h"
#include "text_file_meta_data_reader.h"
#include "cifar10_meta_data_reader.h"
#include "tf_meta_data_reader.h"
#include "caffe_meta_data_reader.h"
#include "caffe_meta_data_reader_detection.h"
#include "caffe2_meta_data_reader.h"
#include "caffe2_meta_data_reader_detection.h"
#include "tf_meta_data_reader_detection.h"
#include "video_label_reader.h"
#include "mxnet_meta_data_reader.h"

std::shared_ptr<MetaDataReader> create_meta_data_reader(const MetaDataConfig& config) {
    switch(config.reader_type()) {
        case MetaDataReaderType::FOLDER_BASED_LABEL_READER:
        {
            if(config.type() != MetaDataType::Label)
                THROW("FOLDER_BASED_LABEL_READER can only be used to load labels")
            auto ret = std::make_shared<LabelReaderFolders>();
            ret->init(config);
            return ret;
        }
            break;
#ifdef ROCAL_VIDEO
        case MetaDataReaderType::VIDEO_LABEL_READER:
        {
            if(config.type() != MetaDataType::Label)
                THROW("FOLDER_BASED_LABEL_READER can only be used to load labels")
            auto ret = std::make_shared<VideoLabelReader>();
            ret->init(config);
            return ret;
        }
            break;
#endif
        case MetaDataReaderType::TEXT_FILE_META_DATA_READER:
        {
            if(config.type() != MetaDataType::Label)
                THROW("TEXT_FILE_META_DATA_READER can only be used to load labels")
            auto ret = std::make_shared<TextFileMetaDataReader>();
            ret->init(config);
            return ret;
        }
            break;
        case MetaDataReaderType::TF_META_DATA_READER:
        {
            if(config.type() != MetaDataType::Label)
                THROW("TF_META_DATA_READER can only be used to load labels")
            auto ret = std::make_shared<TFMetaDataReader>();
            ret->init(config);
            return ret;
        }
            break;
        case MetaDataReaderType::TF_DETECTION_META_DATA_READER:
        {
            if(config.type() != MetaDataType::BoundingBox)
                THROW("FOLDER_BASED_LABEL_READER can only be used to load bounding boxes")
            auto ret = std::make_shared<TFMetaDataReaderDetection>();
            ret->init(config);
            return ret;
        }
            break;
        case MetaDataReaderType::COCO_META_DATA_READER:
        {
            if(config.type() != MetaDataType::BoundingBox)
                THROW("FOLDER_BASED_LABEL_READER can only be used to load bounding boxes")
            auto ret = std::make_shared<COCOMetaDataReader>();
            ret->init(config);
            return ret;
        }
            break;
        case MetaDataReaderType::COCO_KEY_POINTS_META_DATA_READER:
        {
            if(config.type() != MetaDataType::KeyPoints)
                THROW("COCO_KEY_POINTS_META_DATA_READER can only be used to load keypoints")
            auto ret = std::make_shared<COCOMetaDataReaderKeyPoints>();
            ret->init(config);
            return ret;
        }
            break;
        case MetaDataReaderType::CIFAR10_META_DATA_READER:
        {
            if(config.type() != MetaDataType::Label)
                THROW("TEXT_FILE_META_DATA_READER can only be used to load labels")
            auto ret = std::make_shared<Cifar10MetaDataReader>();
            ret->init(config);
            return ret;
        }
            break;
        case MetaDataReaderType::CAFFE_META_DATA_READER:
        {
            if(config.type() != MetaDataType::Label)
                THROW("CAFFE_META_DATA_READER can only be used to load labels")
            auto ret = std::make_shared<CaffeMetaDataReader>();
            ret->init(config);
            return ret;
        }
            break;
        case MetaDataReaderType::CAFFE_DETECTION_META_DATA_READER:
        {
            if(config.type() != MetaDataType::BoundingBox)
                THROW("CAFFE_DETECTION_META_DATA_READER can only be used to load labels")
            auto ret = std::make_shared<CaffeMetaDataReaderDetection>();
            ret->init(config);
            return ret;
        }
            break;
        case MetaDataReaderType::CAFFE2_META_DATA_READER:
        {
            if(config.type() != MetaDataType::Label)
                THROW("CAFFE2_META_DATA_READER can only be used to load labels")
            auto ret = std::make_shared<Caffe2MetaDataReader>();
            ret->init(config);
            return ret;
        }
            break;
        case MetaDataReaderType::CAFFE2_DETECTION_META_DATA_READER:
        {
            if(config.type() != MetaDataType::BoundingBox)
                THROW("CAFFE2_DETECTION_META_DATA_READER can only be used to load labels")
            auto ret = std::make_shared<Caffe2MetaDataReaderDetection>();
            ret->init(config);
            return ret;
        }
	    break;
        case MetaDataReaderType::MXNET_META_DATA_READER:
        {
            if(config.type() != MetaDataType::Label)
                THROW("MXNetMetaDataReader can only be used to load labels")
            auto ret = std::make_shared<MXNetMetaDataReader>();
            ret->init(config);
            return ret;
        }
        break;
        default:
            THROW("MetaDataReader type is unsupported : "+ TOSTR(config.reader_type()));
    }
}
