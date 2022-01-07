#pragma once
#include <map>
#include <dirent.h>
#include <memory>
#include <list>
#include <variant>
#include <fstream>
#include <string>
#include "commons.h"
#include "meta_data.h"
#include "meta_data_reader.h"
#include "reader.h"

class MXNetMetaDataReader: public MetaDataReader
{
public :
    void init(const MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    bool set_timestamp_mode() override { return false; }
    MetaDataBatch * get_output() override { return _output; }
    MXNetMetaDataReader();
    ~MXNetMetaDataReader() override { delete _output; }
private:
    void read_images();
    bool exists(const std::string &image_name);
    void add(std::string image_name, int label);
    uint32_t DecodeFlag(uint32_t rec) {return (rec >> 29U) & 7U; };
    uint32_t DecodeLength(uint32_t rec) {return rec & ((1U << 29U) - 1U); };
    std::vector<std::tuple<int64_t, int64_t>> _indices; // used to store seek position and record size for a particular record.
    std::ifstream _file_contents;
    ImageRecordIOHeader _hdr;
    const uint32_t _kMagic = 0xced7230a;
    std::map<std::string, std::shared_ptr<Label>> _map_content;
    std::string _path;
    LabelBatch* _output;
    std::vector<std::string> _image_name;
};