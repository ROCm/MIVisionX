#pragma once

#include <string>
#include <memory>
#include "meta_data.h"

enum class MetaDataReaderType
{
    FOLDER_BASED_LABEL_READER = 0,// Used for imagenet-like dataset
    TEXT_FILE_META_DATA_READER,// Used when metadata is stored in a text file
    COCO_META_DATA_READER
};
enum class MetaDataType
{
    Label,
    BoundingBox
};

struct MetaDataConfig
{
private:
    MetaDataType _type;
    MetaDataReaderType _reader_type;
    std::string _path;
public:
    MetaDataConfig(const MetaDataType& type, const MetaDataReaderType& reader_type, const std::string& path ):_type(type),  _path(path){}
    MetaDataConfig() = delete;
    MetaDataType type() const { return _type; }
    MetaDataReaderType reader_type() const { return _reader_type; }
    std::string path() const { return  _path; }
};


class MetaDataReader
{
public:
    virtual ~MetaDataReader()= default;
    virtual void init(const MetaDataConfig& cfg) = 0;
    virtual void read_all(const std::string& path) = 0;// Reads all the meta data information
    virtual void lookup(const std::vector<std::string>& image_names) = 0;// finds meta_data info associated with given names and fills the output
    virtual void release() = 0; // Deletes the loaded information
    virtual MetaDataBatch * get_output()= 0;

};

