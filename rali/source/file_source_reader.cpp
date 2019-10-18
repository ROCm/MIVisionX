#include <cassert>
#include <commons.h>
#include "file_source_reader.h"

FileSourceReader::FileSourceReader() 
{
    _src_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
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
    _folder_path = desc.path();
    _read_offset = desc.offset();
    _read_interval = desc.interval();
    _loop = desc.loop();
    return open_folder();
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

Reader::Status FileSourceReader::open_folder()
{
    if ((_src_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _folder_path);

    size_t read_counter = 0;
    while((_entity = readdir (_src_dir)) != nullptr)
    {
        if(_entity->d_type != DT_REG)
            continue;
            
        if(++read_counter <= _read_offset)
            continue;
        
        if((read_counter - _read_offset - 1)% _read_interval != 0)
            continue;

        std::string file_path = _folder_path;
        file_path.append("/");
        file_path.append(_entity->d_name);
        _file_names.push_back(file_path);
    }
    if(!_file_names.empty())
        LOG("Total of " + TOSTR(_file_names.size()) + " images loaded from " + _folder_path )
    else
        THROW("Could not find any file in " + _folder_path)

    _curr_file_idx = 0;
    closedir(_src_dir);
    return Reader::Status::OK;
}
