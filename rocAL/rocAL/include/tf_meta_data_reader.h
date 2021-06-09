#pragma once
#include <map>
#include <dirent.h>
#include <memory>
#include <list>
#include <variant>
#include "commons.h"
#include "meta_data.h"
#include "meta_data_reader.h"

class TFMetaDataReader: public MetaDataReader
{
public :
    void init(const MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    MetaDataBatch * get_output() override { return _output; }
    TFMetaDataReader();
    ~TFMetaDataReader() override { delete _output; }
private:
    void read_files(const std::string& _path);
    bool exists(const std::string &image_name);
    void add(std::string image_name, int label);
    bool _last_rec;
    size_t _file_id = 0;
    //std::shared_ptr<TF_Read> _TF_read = nullptr;
    void read_record(std::ifstream &file_contents, uint file_size, std::vector<std::string> &image_name, std::string user_label_key, std::string user_filename_key);
    void incremenet_file_id() { _file_id++; }
    std::map<std::string, std::shared_ptr<Label>> _map_content;
    std::map<std::string, std::shared_ptr<Label>>::iterator _itr;
    std::string _path;
    std::map<std::string, std::string> _feature_key_map;
    LabelBatch* _output;
    DIR *_src_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names;
    std::vector<std::string> _subfolder_file_names;
    std::vector<std::string> _image_name;
};