#pragma once

#include <string>
#include <memory>
#include "meta_data.h"

enum class MetaDataReaderType
{
    FOLDER_BASED_LABEL_READER = 0,// Used for imagenet-like dataset
    TEXT_FILE_META_DATA_READER,// Used when metadata is stored in a text file
    COCO_META_DATA_READER,
    CIFAR10_META_DATA_READER,    // meta_data for cifar10 data which is store as part of bin file
    TF_META_DATA_READER
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
    std::string _file_prefix;           // if we want to read only filenames with prefix (needed for cifar10 meta data)
public:
    MetaDataConfig(const MetaDataType& type, const MetaDataReaderType& reader_type, const std::string& path, const std::string file_prefix=std::string())
                    :_type(type), _reader_type(reader_type),  _path(path), _file_prefix(file_prefix){}
    MetaDataConfig() = delete;
    MetaDataType type() const { return _type; }
    MetaDataReaderType reader_type() const { return _reader_type; }
    std::string path() const { return  _path; }
    std::string file_prefix() const { return  _file_prefix; }
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

