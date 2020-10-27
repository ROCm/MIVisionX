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

#pragma once
#include <string>
#include <map>
enum class StorageType
{
    FILE_SYSTEM = 0,
    TF_RECORD = 1,
    UNCOMPRESSED_BINARY_DATA = 2,        // experimental: added for supporting cifar10 data set
    CAFFE_LMDB_RECORD = 3,
    CAFFE2_LMDB_RECORD = 4,
    COCO_FILE_SYSTEM = 5
};

struct ReaderConfig
{
    explicit ReaderConfig(StorageType type, std::string path = "", std::string json_path = "", 
        const std::map<std::string, std::string> feature_key_map = std::map<std::string, std::string>(),
        bool shuffle = false, bool loop = false):_type(type), _path(path), _json_path(json_path), _feature_key_map(feature_key_map), _shuffle(shuffle), _loop(loop) {}
        virtual StorageType type() { return _type; };
    void set_path(const std::string& path) { _path = path; }
    void set_shard_id(size_t shard_id) { _shard_id = shard_id; }
    void set_shard_count(size_t shard_count) { _shard_count = shard_count; }
    void set_json_path(const std::string& json_path) { _json_path = json_path;}
    /// \param read_batch_count Tells the reader it needs to read the images in multiples of load_batch_count. If available images not divisible to load_batch_count,
    /// the reader will repeat images to make available images an even multiple of this load_batch_count
    void set_batch_count(size_t read_batch_count) { _batch_count = read_batch_count; }
    /// \param loop if True the reader's available images still the same no matter how many images have been read
    bool shuffle() { return _shuffle; }
    bool loop() { return _loop; }
    void set_shuffle( bool shuffle) { _shuffle = shuffle; }
    void set_loop( bool loop) { _loop = loop; }
    size_t get_shard_count() { return _shard_count; }
    size_t get_shard_id() { return _shard_id; }
    size_t get_batch_size() { return _batch_count; }
    std::string path() { return _path; }
    std::string json_path() { return _json_path;}
    std::map<std::string, std::string> feature_key_map() {return _feature_key_map; }
    void set_file_prefix(const std::string &prefix) {_file_prefix = prefix;}
    std::string file_prefix() {return _file_prefix;}
private:
    StorageType _type = StorageType::FILE_SYSTEM;
    std::string _path = "";
    std::string _json_path = "";
    std::map<std::string, std::string> _feature_key_map;
    size_t _shard_count= 1 ;
    size_t _shard_id = 0;
    size_t _batch_count = 1;//!< The reader will repeat images if necessary to be able to have images in multiples of the _batch_count.
    bool _shuffle = false;
    bool _loop = false;
    std::string _file_prefix = ""; //!< to read only files with prefix. supported only for cifar10_data_reader and tf_record_reader
};

class Reader {
public:
    enum class Status
    {
        OK = 0
    };

    // TODO: change method names to open_next, read_next , ...


    //! Initializes the resource which it's spec is defined by the desc argument
    /*!
     \param desc description of the resource infor. It's exact fields are defind by the derived class.
     \return status of the being able to locate the resource pointed to by the desc
    */
    virtual Status initialize(ReaderConfig desc) = 0;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded items	
     \return Size of the loaded resource
    */
 
       //! Opens the next item and returns it's size
    /*!
     \return Size of the item, if 0 failed to access it
    */
    virtual size_t open() = 0;
    
    //! Copies the data of the opened item to the buf
    virtual size_t read(unsigned char* buf, size_t read_size) = 0;

    //! Closes the opened item 
    virtual int close() = 0;

    //! Starts reading from the first item in the resource
    virtual void reset() = 0;

    //! Returns the name/identifier of the last item opened in this resource
    virtual std::string id() = 0;
    //! Returns the number of items remained in this resource
    virtual unsigned count() = 0;
    //! return shuffle_time if applicable
    virtual unsigned long long get_shuffle_time() = 0;

    virtual ~Reader() = default;

    #define E(expr) CHECK_CAFFE((rc = (expr)) == MDB_SUCCESS, #expr)
    #define CHECK_CAFFE(test, msg); ((test) ? (void)0 : ((void)fprintf(stderr, \
    "%s:%d: %s: %s\n", __FILE__, __LINE__, msg, mdb_strerror(rc)), abort()))
};

