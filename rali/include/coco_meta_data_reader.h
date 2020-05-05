#pragma once
#include <map>
#include "commons.h"
#include "meta_data.h"
#include "meta_data_reader.h"
class COCOMetaDataReader: public MetaDataReader
{
public:
    void init(const MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    MetaDataBatch * get_output() override { return _output; }
    COCOMetaDataReader();
    ~COCOMetaDataReader() override { delete _output; }
private:
    BoundingBoxBatch* _output;
    std::string _path;
    void add(std::string image_name, BoundingBoxCords bbox, BoundingBoxLabels b_labels);
    bool exists(const std::string &image_name);
    std::map<std::string, std::shared_ptr<BoundingBox>> _map_content;
    std::map<std::string, std::shared_ptr<BoundingBox>>::iterator _itr;
};