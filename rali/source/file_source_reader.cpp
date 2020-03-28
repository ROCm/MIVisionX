#include <cassert>
#include <commons.h>
#include "file_source_reader.h"
#include <boost/filesystem.hpp>

namespace filesys = boost::filesystem;

FileSourceReader::FileSourceReader()
{
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _file_id = 0;
}

unsigned FileSourceReader::count()
{
    if(_loop)
        return _file_names.size();

    int ret = ((int)_file_names.size() -_read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status FileSourceReader::initialize(ReaderConfig desc)
{
    _file_id = 0;
    _folder_path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _loop = desc.loop();
    return subfolder_reading();
}

void FileSourceReader::incremenet_read_ptr()
{
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
}
size_t FileSourceReader::open()
{
    auto file_path = _file_names[_curr_file_idx];// Get next file name
    incremenet_read_ptr();
    _last_id= file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
        _last_id.erase(0, last_slash_idx + 1);
    }

    _current_fPtr = fopen(file_path.c_str(), "rb");// Open the file,

    if(!_current_fPtr) // Check if it is ready for reading
        return 0;

    fseek(_current_fPtr, 0 , SEEK_END);// Take the file read pointer to the end

    _current_file_size = ftell(_current_fPtr);// Check how many bytes are there between and the current read pointer position (end of the file)

    if(_current_file_size == 0)
    { // If file is empty continue
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        return 0;
    }

    fseek(_current_fPtr, 0 , SEEK_SET);// Take the file pointer back to the start

    return _current_file_size;
}

size_t FileSourceReader::read(unsigned char* buf, size_t read_size)
{
    if(!_current_fPtr)
        return 0;

    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > _current_file_size) ? _current_file_size : read_size;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
    return actual_read_size;
}

int FileSourceReader::close()
{
    return release();
}

FileSourceReader::~FileSourceReader()
{
    release();
}

int
FileSourceReader::release()
{
    if(!_current_fPtr)
        return 0;
    fclose(_current_fPtr);
    _current_fPtr = nullptr;
    return 0;
}

void FileSourceReader::reset()
{
    _read_counter = 0;
    _curr_file_idx = 0;
}

Reader::Status FileSourceReader::subfolder_reading()
{
    if ((_sub_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID ["+ TOSTR(_shard_id)+ "] ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while((_entity = readdir (_sub_dir)) != nullptr)
    {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
    }
    std::sort(entry_name_list.begin(), entry_name_list.end());

    std::string subfolder_path = _full_path + "/" + entry_name_list[0];

    filesys::path pathObj(subfolder_path);
    auto ret = Reader::Status::OK;
    if(filesys::exists(pathObj) && filesys::is_regular_file(pathObj))
    {
        ret = open_folder();
    }
    else if(filesys::exists(pathObj) && filesys::is_directory(pathObj))
    {
        for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
            std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
            _folder_path = subfolder_path;
            if(open_folder() != Reader::Status::OK)
                WRN("FileReader ShardID ["+ TOSTR(_shard_id)+ "] File reader cannot access the storage at " + _folder_path);
        }
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
void FileSourceReader::replicate_last_image_to_fill_last_shard()
{
    for(size_t i = _in_batch_read_count; i < _batch_count; i++)
        _file_names.push_back(_last_file_name);
}

Reader::Status FileSourceReader::open_folder()
{
    if ((_src_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID ["+ TOSTR(_shard_id)+ "] ERROR: Failed opening the directory at " + _folder_path);


    while((_entity = readdir (_src_dir)) != nullptr)
    {
        if(_entity->d_type != DT_REG)
            continue;

        if(get_file_shard_id() != _shard_id )
        {
            incremenet_file_id();
            continue;
        }
        _in_batch_read_count++;
        _in_batch_read_count = (_in_batch_read_count%_batch_count == 0) ? 0 : _in_batch_read_count;
        std::string file_path = _folder_path;
        file_path.append("/");
        file_path.append(_entity->d_name);
        _last_file_name = file_path;
        _file_names.push_back(file_path);
        incremenet_file_id();
    }
    if(_file_names.empty())
        WRN("FileReader ShardID ["+ TOSTR(_shard_id)+ "] Did not load any file from " + _folder_path)

    closedir(_src_dir);
    return Reader::Status::OK;
}

size_t FileSourceReader::get_file_shard_id()
{
    if(_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    return (_file_id / (_batch_count)) % _shard_count;
}
