#pragma once
#include <map>
#include <dirent.h>
#include "commons.h"
#include "meta_data.h"
#include "meta_data_reader.h"
#include <lmdb.h>
#include "caffe_protos.pb.h"

class CaffeMetaDataReader: public MetaDataReader
{
public :
    void init(const MetaDataConfig& cfg) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    MetaDataBatch * get_output() override { return _output; }
    CaffeMetaDataReader();
    ~CaffeMetaDataReader() override { delete _output; }
private:
    void read_files(const std::string& _path);
    void read_lmdb_record(std::string _path, uint file_size);
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
    MDB_env* mdb_env;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_value;
    MDB_txn* mdb_txn;
    MDB_cursor* mdb_cursor;
 };
