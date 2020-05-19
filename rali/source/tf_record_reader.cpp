#include <cassert>
#include <commons.h>
#include "tf_record_reader.h"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdint.h>

namespace filesys = boost::filesystem;

TFRecordReader::TFRecordReader()
{
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _loop = false;
    _shuffle = false;
    _file_id = 0;
    _last_rec = false;
}

unsigned TFRecordReader::count()
{
    if(_loop)
        return _file_names.size();

    int ret = ((int)_file_names.size() -_read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status TFRecordReader::initialize(ReaderConfig desc)
{
    _file_id = 0;
    _folder_path = desc.path();
    _path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _loop = desc.loop();
    _shuffle = desc.shuffle();
    return folder_reading();
}

void TFRecordReader::incremenet_read_ptr()
{
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
}
size_t TFRecordReader::open()
{
    auto file_path = _file_names[_curr_file_idx];// Get next file name
    _last_id= file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
        _last_id.erase(0, last_slash_idx + 1);
    }
    _current_file_size = _file_size[_curr_file_idx];
    return _current_file_size;
}

size_t TFRecordReader::read(unsigned char* buf, size_t read_size)
{
    read_image(buf, _file_names[_curr_file_idx], _file_size[_curr_file_idx]);
    incremenet_read_ptr();
    return  read_size;

}

int TFRecordReader::close()
{
    return release();
}

TFRecordReader::~TFRecordReader()
{
    release();
}

int
TFRecordReader::release()
{
    return 0;
}

void TFRecordReader::reset()
{
    if(_shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());
    _read_counter = 0;
    _curr_file_idx = 0;
}

Reader::Status TFRecordReader::folder_reading()
{
    if ((_sub_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID ["+ TOSTR(_shard_id)+ "] ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;
    auto ret = Reader::Status::OK;
    while((_entity = readdir (_sub_dir)) != nullptr)
    {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
        // std::cerr<<"\n entry_name::"<<entry_name;
    }
    std::sort(entry_name_list.begin(), entry_name_list.end());
    for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
            std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
            _folder_path = subfolder_path;
            if(tf_record_reader() != Reader::Status::OK)
                WRN("FileReader ShardID ["+ TOSTR(_shard_id)+ "] File reader cannot access the storage at " + _folder_path);
    }
    if(_in_batch_read_count > 0 && _in_batch_read_count < _batch_count)
    {
        replicate_last_image_to_fill_last_shard();
        LOG("FileReader ShardID [" + TOSTR(_shard_id) + "] Replicated " + _folder_path+_last_file_name + " " + TOSTR((_batch_count - _in_batch_read_count) ) + " times to fill the last batch")
    }
    if(!_file_names.empty())
        LOG("FileReader ShardID ["+ TOSTR(_shard_id)+ "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path )
    closedir(_sub_dir);
    return ret;
}
void TFRecordReader::replicate_last_image_to_fill_last_shard()
{
    // std::cerr<<"\n Replicate last image";
    for(size_t i = _in_batch_read_count; i < _batch_count; i++)
    {
        _file_names.push_back(_last_file_name);
        _file_size.push_back(_last_file_size);
    }
}

Reader::Status TFRecordReader::tf_record_reader()
{
    std::string fname = _folder_path;

    uint file_size;
    std::ifstream file_contents(fname.c_str(),std::ios::binary);
    file_contents.seekg (0, std::ifstream::end);
    file_size = file_contents.tellg(); 
    // std::cerr<<"\n length of the file:: "<<length<<std::endl;
    file_contents.seekg (0, std::ifstream::beg);
    read_image_names(file_contents, file_size);
    _last_rec = false; 
    if(_file_names.size() != _file_size.size())
        std::cerr<<"\n Size of vectors are not same";
    file_contents.close();

    return Reader::Status::OK;
}

size_t TFRecordReader::get_file_shard_id()
{
    if(_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    return (_file_id / (_batch_count)) % _shard_count;
}


void TFRecordReader::read_image_names(std::ifstream &file_contents, uint file_size)
{
#if TFRECORD_ENABLE

    uint length;
    size_t uint64_size, uint32_size;
    uint64_t data_length;
    uint32_t length_crc, data_crc;
    uint64_size = sizeof(uint64_t); 
    uint32_size = sizeof(uint32_t);
    while(!_last_rec)
    {
        length = file_contents.tellg();
        char * header_length = new char [uint64_size];
        char * header_crc = new char [uint32_size];
        char * footer_crc = new char [uint32_size];
        file_contents.read(header_length, uint64_size);
        file_contents.read(header_crc, uint32_size);
        memcpy(&data_length, header_length, sizeof(data_length));
        memcpy(&length_crc, header_crc, sizeof(length_crc));
        if(uint(length + data_length + 16) == file_size)
        {
            _last_rec = true;

        }
        char *data = new char[data_length];
        file_contents.read(data,data_length);
        _single_example.ParseFromArray(data,data_length);
        _features = _single_example.features();
        auto feature = _features.feature();
        _single_feature = feature.at("image/filename");
        std::string fname = _single_feature.bytes_list().value()[0];
        _image_record_starting.insert(std::pair<std::string, uint>(fname, length));
        _in_batch_read_count++;
        _in_batch_read_count = (_in_batch_read_count%_batch_count == 0) ? 0 : _in_batch_read_count;
        std::string file_path = _folder_path;
        file_path.append("/");
        file_path.append(fname);
        _last_file_name = file_path;
        if(get_file_shard_id() != _shard_id )
        {
            incremenet_file_id();
            file_contents.read(footer_crc, sizeof(data_crc));
            continue;
        }
        _file_names.push_back(file_path);
        incremenet_file_id();

        _single_feature = feature.at("image/encoded");
        _last_file_size  = _single_feature.bytes_list().value()[0].size();
        _file_size.push_back(_last_file_size);
        file_contents.read(footer_crc, sizeof(data_crc));
        memcpy(&data_crc, footer_crc, sizeof(data_crc));
        free(header_length);
        free(header_crc);
        free(footer_crc);
        free(data);
    }
#else
    return;
#endif
}

void TFRecordReader::read_image(unsigned char* buff, std::string file_name, uint file_size)
{
#if TFRECORD_ENABLE
    std::string temp = file_name.substr(0, file_name.find_last_of("\\/"));
    const size_t last_slash_idx = file_name.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
        file_name.erase(0, last_slash_idx + 1);
    }  
    std::ifstream file_contents(temp.c_str(),std::ios::binary);
    auto it = _image_record_starting.find(file_name);
    if(_image_record_starting.end() == it)
    {
        THROW("ERROR: Given name not present in the map"+ file_name )
    }
    // std::cerr<<"\n image present at loc:: "<<it->second;
    file_contents.seekg (it->second, std::ifstream::beg);
    size_t uint64_size, uint32_size;
    uint64_t data_length;
    uint32_t length_crc, data_crc;
    uint64_size = sizeof(uint64_t); 
    uint32_size = sizeof(uint32_t); 
    char * header_length = new char [uint64_size];
    char * header_crc = new char [uint32_size];
    char * footer_crc = new char [uint32_size];
    file_contents.read(header_length, uint64_size);
    file_contents.read(header_crc, uint32_size);
    memcpy(&data_length, header_length, sizeof(data_length));
    memcpy(&length_crc, header_crc, sizeof(length_crc));
    char *data = new char[data_length];
    file_contents.read(data,data_length);
    _single_example.ParseFromArray(data,data_length);
    _features = _single_example.features();
    auto feature = _features.feature();
    _single_feature = feature.at("image/filename");
    std::string fname = _single_feature.bytes_list().value()[0];
    if(fname == file_name)
    {
        _single_feature = feature.at("image/encoded");
        memcpy(buff,_single_feature.bytes_list().value()[0].c_str(),_single_feature.bytes_list().value()[0].size());
    }        
    file_contents.read(footer_crc, sizeof(data_crc));
    memcpy(&data_crc, footer_crc, sizeof(data_crc));
    free(header_length);
    free(header_crc);
    free(footer_crc);
    file_contents.close();
    free(data);
#else
    return;
#endif
}
