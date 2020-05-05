#pragma once
#include <map>
#include <dirent.h>
#include "commons.h"
#include "meta_data.h"
#include "meta_data_reader.h"

class LabelReaderFolders: public MetaDataReader
{
public :
    void init(const MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    MetaDataBatch * get_output() override { return _output; }
    LabelReaderFolders();
    ~LabelReaderFolders() override { delete _output; }
private:
    void read_files(const std::string& _path);
    bool exists(const std::string &image_name);
    void add(std::string image_name, int label);
    std::map<std::string, std::shared_ptr<Label>> _map_content;
    std::map<std::string, std::shared_ptr<Label>>::iterator _itr;
    std::string _path;
    LabelBatch* _output;
    DIR *_src_dir, *_sub_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names;
    std::vector<std::string> _subfolder_file_names;
};