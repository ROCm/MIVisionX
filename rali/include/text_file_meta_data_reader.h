#pragma once
#include <map>
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
    void read_files(const std::string& _path);
    bool exists(const std::string &image_name);
    void add(std::string image_name, int label);
    std::map<std::string, std::shared_ptr<Label>> _map_content;
    std::string _path;
};
