/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#include <algorithm>
#include "external_source_reader.h"
#include <boost/filesystem.hpp>

namespace filesys = boost::filesystem;

ExternalSourceReader::ExternalSourceReader() {
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _file_id = 0;
    _shuffle = false;
    _file_count_all_shards = 0;
    _loop = false;    // loop not supported for external source
    _end_of_sequence = false;
}

// return batch_size() for count_items unless end_of_sequence has been signalled.
unsigned ExternalSourceReader::count_items() {
    if (_file_mode == ExternalFileMode::FILENAME) {
        if (_end_of_sequence && _file_names_queue.empty()) {
        return 0;
        }
    } else {
        if (_end_of_sequence && _images_data_queue.empty()) {
        return 0;
        }
    }
    return _batch_count;
}

Reader::Status ExternalSourceReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _folder_path = desc.path();
    _file_id = 0;
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _file_mode = desc.mode();
    _end_of_sequence = false;
    _file_data.reserve(_batch_count);
    return ret;
}

void ExternalSourceReader::incremenet_read_ptr() {
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _batch_count;
}

size_t ExternalSourceReader::open() {
    if (_file_mode == ExternalFileMode::FILENAME) {
        std::string next_file_name;
        bool ret = pop_file_name(next_file_name);   // Get next file name: blocking call, will wait till next file is received from external source
        if (_end_of_sequence && !ret)
          return 0;
        _last_id= next_file_name;
        filesys::path pathObj(next_file_name);
        if(filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
          _current_fPtr = fopen(next_file_name.c_str(), "rb"); // Open the file,
          if(!_current_fPtr) // Check if it is ready for reading
              return 0;
          fseek(_current_fPtr, 0 , SEEK_END); // Take the file read pointer to the end
          _current_file_size = ftell(_current_fPtr); // Check how many bytes are there between and the current read pointer position (end of the file)
          if(_current_file_size == 0)
          { // If file is empty continue
              fclose(_current_fPtr);
              _current_fPtr = nullptr;
              return 0;
          }
          fseek(_current_fPtr, 0 , SEEK_SET); // Take the file pointer back to the start
          _file_data[_curr_file_idx] = std::make_tuple((unsigned char *)next_file_name.data(), (size_t)_current_file_size, 0, 0, 0);
          incremenet_read_ptr();
        }
    } else {
        std::tuple<unsigned char*, size_t, int, int, int> image;
        bool ret = pop_file_data(image);
        if (_end_of_sequence && !ret)
        {
            std::cerr<<"\n EOS || POP FAILED ";
            return 0;
        }
        _file_data[_curr_file_idx] = image;
        _current_file_size = std::get<1>(image);
    }
    return _current_file_size;
}

size_t ExternalSourceReader::read_data(unsigned char* buf, size_t read_size) {
    if (_file_mode == ExternalFileMode::FILENAME) {
        if(!_current_fPtr)
            return 0;

        // Requested read size bigger than the file size? just read as many bytes as the file size
        read_size = std::min(static_cast<unsigned int>(read_size),_current_file_size);
        size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
        return actual_read_size;
    } else {
        unsigned char *file_data_ptr = std::get<0>(_file_data[_curr_file_idx]);
        size_t size = _current_file_size;
        incremenet_read_ptr();
        if (size > read_size)
          THROW("Requested size doesn't match the actual size for file read")
        memcpy(buf, static_cast<void *>(file_data_ptr), size);
        return size;
    }
}

void ExternalSourceReader::get_dims(int cur_idx, int& width, int& height, int& channels) {
    if (cur_idx >= 0) {
      width = std::get<2>(_file_data[cur_idx]);
      height = std::get<3>(_file_data[cur_idx]);
      channels = std::get<4>(_file_data[cur_idx]);
    }
}

int ExternalSourceReader::close() {
    return release();
}

ExternalSourceReader::~ExternalSourceReader() {
    release();
}

int ExternalSourceReader::release() {
    if (_file_mode != ExternalFileMode::FILENAME) {
        if(!_current_fPtr)
            return 0;
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        _end_of_sequence = false;   // reset for looping
    }
    return 0;
}

void ExternalSourceReader::reset() {
    _read_counter = 0;
    _curr_file_idx = 0;
    _end_of_sequence = false;   // reset for looping
}

size_t ExternalSourceReader::get_file_shard_id() {
    if(_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    return _file_id  % _shard_count;
}

void ExternalSourceReader::push_file_name(const std::string& file_name) {
    std::unique_lock<std::mutex> lock(_lock);
    _file_names_queue.push(file_name);
    lock.unlock();
    // notify waiting thread of new data
    _wait_for_input.notify_all();
}

bool ExternalSourceReader::pop_file_name(std::string& file_name) {
    std::unique_lock<std::mutex> lock(_lock);
    if(_file_names_queue.empty() && !_end_of_sequence)
        _wait_for_input.wait(lock);
    if (!_file_names_queue.empty()) {
      file_name = _file_names_queue.front();
      _file_names_queue.pop();
      return true;
    } else
      return false;
}

void ExternalSourceReader::push_file_data(std::tuple<unsigned char*, size_t, int, int, int>& image) {
    std::unique_lock<std::mutex> lock(_lock);
    _images_data_queue.push(image);
    lock.unlock();
    // notify waiting thread of new data
    _wait_for_input.notify_all();
}

bool ExternalSourceReader::pop_file_data(std::tuple<unsigned char*,  size_t, int, int, int>& image) {
    std::unique_lock<std::mutex> lock(_lock);
    if(_images_data_queue.empty() && !_end_of_sequence)
        _wait_for_input.wait(lock);
    if (!_images_data_queue.empty()) {
      image = _images_data_queue.front();
      _images_data_queue.pop();
      return true;
    } else
      return false;
}

void ExternalSourceReader::feed_file_names(const std::vector<std::string>& file_names, size_t num_images, bool eos) {
    for (unsigned n = 0; n < num_images; n++) {
      push_file_name(file_names[n]);
    }
    _end_of_sequence = eos;
}

void ExternalSourceReader::feed_data(const std::vector<unsigned char *>& images, const std::vector<size_t>& image_size, ExternalFileMode mode, bool eos, int width, int height, int channels) {
    for (unsigned n = 0; n < images.size(); n++) {
        std::tuple<unsigned char*, size_t, int, int, int> image =  std::make_tuple(images[n], image_size[n], width, height, channels);
        push_file_data(image);
    }
    _end_of_sequence = eos;
}
