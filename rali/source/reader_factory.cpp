#include <stdexcept>
#include <memory>
#include "reader_factory.h"
#include "tf_record_reader.h"
#include "file_source_reader.h"

std::shared_ptr<Reader> create_reader(ReaderConfig config) {
    switch(config.type()) {
        case StorageType ::FILE_SYSTEM:
        {
            auto ret = std::make_shared<FileSourceReader>();
            if(ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("File reader cannot access the storage");
            return ret;
        }
        case StorageType::TF_RECORD:
        {
            auto ret = std::make_shared<TFRecordReader>();
            if(ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("File reader cannot access the storage");
            return ret;
        }
        break;
        default:
            throw std::runtime_error ("Reader type is unsupported");
    }
}
