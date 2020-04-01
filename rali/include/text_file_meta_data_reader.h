#pragma once
#include "commons.h"
#include "meta_data.h"
#include "meta_data_reader.h"
class TextFileMetaDataReader: public MetaDataReader
{
public:
    void init(const MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    MetaDataBatch * get_output() override { return _output; }
    TextFileMetaDataReader();
    ~TextFileMetaDataReader() override { delete _output; }
private:
    LabelBatch* _output;
};
