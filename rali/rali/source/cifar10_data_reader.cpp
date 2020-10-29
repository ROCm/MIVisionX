/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cassert>
#include <commons.h>
#include "cifar10_data_reader.h"
#include <boost/filesystem.hpp>
#include <file_source_reader.h>

namespace filesys = boost::filesystem;

CIFAR10DataReader::CIFAR10DataReader()
{
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _file_id = 0;
    _total_file_size = 0;
    _last_file_idx = 0;
}

unsigned CIFAR10DataReader::count()
{
    if(_loop)
        return _file_names.size();

    int ret = ((int)_file_names.size() -_read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status CIFAR10DataReader::initialize(ReaderConfig desc)
{
    _file_id = 0;
    _folder_path = desc.path();
    _batch_count = desc.get_batch_size();
    _loop = desc.loop();
   // _file_name_prefix = "data_batch_";
    return subfolder_reading();
}

void CIFAR10DataReader::incremenet_read_ptr()
{
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
}

size_t CIFAR10DataReader::open()
{
    auto file_path = _file_names[_curr_file_idx];// Get next file name
    auto file_offset = _file_offsets[_curr_file_idx];
    _last_file_idx = _file_idx[_curr_file_idx];
    incremenet_read_ptr();
    // update _last_id for the next record
    _last_id= file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
        _last_id.erase(0, last_slash_idx + 1);
    }
    // add file_idx to last_id so the loader knows the index within the same master file
    _last_id.append("_");
    _last_id.append(std::to_string(_last_file_idx));
    // compare the file_name with the last one opened
    if ( file_path.compare(_last_file_name) != 0) {
        if (_current_fPtr) {
            fclose(_current_fPtr);
            _current_fPtr = nullptr;
        }
        _current_fPtr = fopen(file_path.c_str(), "rb");// Open the file,
        _last_file_name = file_path;
        fseek(_current_fPtr, 0, SEEK_END);// Take the file read pointer to the end
        _total_file_size = ftell(_current_fPtr);
        fseek(_current_fPtr, 0, SEEK_SET);// Take the file read pointer to the beginning
    }

    if(!_current_fPtr) // Check if it is ready for reading
        return 0;

    fseek(_current_fPtr, file_offset, SEEK_END);// Take the file read pointer to the end

    _current_file_size = ftell(_current_fPtr);// Check how many bytes are there between and the current read pointer position (end of the file)

    if(_current_file_size < _raw_file_size)     // not enough data in the file to read
    { // If file is empty continue
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        return 0;
    }

    fseek(_current_fPtr, file_offset+1 , SEEK_SET);// Take the file pointer back to the fileoffset + 1 extra byte for label

    return (_raw_file_size-1);
}

size_t CIFAR10DataReader::read(unsigned char* buf, size_t read_size)
{
    if(!_current_fPtr)
        return 0;

    // Requested read size bigger than the raw file size? just read as many bytes as the raw file size
    read_size = (read_size > (_raw_file_size-1)) ? _raw_file_size-1 : read_size;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
    return actual_read_size;
}

int CIFAR10DataReader::close()
{
    return release();
}

CIFAR10DataReader::~CIFAR10DataReader()
{
    if(_current_fPtr) {
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
    }
}

int
CIFAR10DataReader::release()
{
    // do not need to close file here since data is read from the same file continuously
    return 0;
}

void CIFAR10DataReader::reset()
{
    _read_counter = 0;
    _curr_file_idx = 0;
}

Reader::Status CIFAR10DataReader::subfolder_reading()
{
    if ((_sub_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("CIFAR10DataReader ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while((_entity = readdir (_sub_dir)) != nullptr)
    {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
        LOG("CIFAR10DataReader  Got entry name " +  entry_name )
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
                WRN("CIFAR10DataReader: File reader cannot access the storage at " + _folder_path);
        }
    }
    if(!_file_names.empty())
        LOG("CIFAR10DataReader  Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path )

    closedir(_sub_dir);
    return ret;
}
/*
 //TODO..
void CIFAR10DataReader::replicate_last_image_to_fill_last_shard()
{
    for(size_t i = _in_batch_read_count; i < _batch_count; i++)
        _file_names.push_back(_last_file_name);
}
*/

Reader::Status CIFAR10DataReader::open_folder()
{
    if ((_src_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("CIFAR10DataReader ERROR: Failed opening the directory at " + _folder_path);


    while((_entity = readdir (_src_dir)) != nullptr)
    {
        if(_entity->d_type != DT_REG)
            continue;
        _in_batch_read_count++;
        _in_batch_read_count = (_in_batch_read_count%_batch_count == 0) ? 0 : _in_batch_read_count;
        std::string file_path = _folder_path;
        // check if the filename has the _file_name_prefix
        std::string  data_file_name = std::string (_entity->d_name);
        if  (data_file_name.find(_file_name_prefix) != std::string::npos) {
            file_path.append("/");
            file_path.append(_entity->d_name);
            FILE* fp = fopen(file_path.c_str(), "rb");// Open the file,
            fseek(fp, 0, SEEK_END);// Take the file read pointer to the end
            size_t total_file_size = ftell(fp);
            size_t num_of_raw_files = _raw_file_size? total_file_size / _raw_file_size: 0;
            unsigned file_offset = 0;
            for (unsigned i = 0; i < num_of_raw_files; i++) {
                _file_names.push_back(file_path);
                _file_offsets.push_back(file_offset);
                _file_idx.push_back(i);
                file_offset += _raw_file_size;
                incremenet_file_id();
            }
            fclose (fp);
        }
    }
    if(_file_names.empty())
        WRN("CIFAR10DataReader:: Did not load any file from " + _folder_path)

    closedir(_src_dir);
    return Reader::Status::OK;
}

