
#include <memory>
#include "label_reader_folders.h"
#include "meta_data_reader_factory.h"
#include "exception.h"
#include "coco_meta_data_reader.h"
#include "text_file_meta_data_reader.h"
#include "cifar10_meta_data_reader.h"


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
        case MetaDataReaderType::TEXT_FILE_META_DATA_READER:
        {
            if(config.type() != MetaDataType::Label)
                THROW("TEXT_FILE_META_DATA_READER can only be used to load labels")
            auto ret = std::make_shared<TextFileMetaDataReader>();
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
        case MetaDataReaderType::CIFAR10_META_DATA_READER:
        {
            if(config.type() != MetaDataType::Label)
                THROW("TEXT_FILE_META_DATA_READER can only be used to load labels")
            auto ret = std::make_shared<Cifar10MetaDataReader>();
            ret->init(config);
            return ret;
        }
            break;
        default:
            THROW("MetaDataReader type is unsupported : "+ TOSTR(config.reader_type()));
    }
}
