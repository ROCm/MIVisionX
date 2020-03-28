
#include <memory>
#include "label_reader_folders.h"
#include "meta_data_reader_factory.h"
#include "exception.h"
#include "bounding_box_reader.h"


std::shared_ptr<MetaDataReader> create_meta_data_manager(const MetaDataConfig& config) {
    switch(config.type()) {
        case MetaDataType::Label:
        {
            auto ret = std::make_shared<LabelReaderFolders>();
            ret->init(config);
            return ret;
        }
            break;
        case MetaDataType::BoundingBox:
        {
            auto ret = std::make_shared<BoundingBoxReader>();
            ret->init(config);
            return ret;
        }
            break;
        default:
            THROW("MetaDataReader type is unsupported");
    }
}
