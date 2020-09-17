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
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdint.h>
#include "caffe_lmdb_record_reader.h"

using namespace std;
using caffe_protos::Datum;

namespace filesys = boost::filesystem;

CaffeLMDBRecordReader::CaffeLMDBRecordReader():
_shuffle_time("shuffle_time", DBG_TIMING)
{
    _sub_dir = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _loop = false;
    _shuffle = false;
    _file_id = 0;
    _last_rec = false;
}

unsigned CaffeLMDBRecordReader::count()
{
    if (_loop)
        return _file_names.size();

    int ret = ((int)_file_names.size() - _read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status CaffeLMDBRecordReader::initialize(ReaderConfig desc)
{
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _folder_path = desc.path();
    _path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _loop = desc.loop();
    _shuffle = desc.shuffle();
    ret = folder_reading();
    //shuffle dataset if set
    _shuffle_time.start();
    if( ret==Reader::Status::OK && _shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());
    _shuffle_time.end();

    return ret;

}

void CaffeLMDBRecordReader::incremenet_read_ptr()
{
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
}
size_t CaffeLMDBRecordReader::open()
{
    auto file_path = _file_names[_curr_file_idx]; // Get next file name
    _last_id = file_path;
    _current_file_size = _file_size[_file_names[_curr_file_idx]];
    return _current_file_size;
}

size_t CaffeLMDBRecordReader::read(unsigned char *buf, size_t read_size)
{
    read_image(buf, _file_names[_curr_file_idx]);
    incremenet_read_ptr();
    return read_size;
}

int CaffeLMDBRecordReader::close()
{
    return release();
}

CaffeLMDBRecordReader::~CaffeLMDBRecordReader()
{
    _open_env = 0;
    mdb_txn_abort(_read_mdb_txn);
    mdb_close(_read_mdb_env, _read_mdb_dbi);
    mdb_env_close(_read_mdb_env);
    _read_mdb_txn = nullptr;
    _read_mdb_env = nullptr;
    release();
}

int CaffeLMDBRecordReader::release()
{
    mdb_cursor_close(_mdb_cursor);
    mdb_txn_abort(_mdb_txn);
    mdb_close(_mdb_env, _mdb_dbi);

    mdb_env_close(_mdb_env);
    _mdb_cursor = nullptr;
    _mdb_txn = nullptr;
    _mdb_env = nullptr;
    return 0;
}

void CaffeLMDBRecordReader::reset()
{
    _shuffle_time.start();
    if (_shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());
    _shuffle_time.end();
    _read_counter = 0;
    _curr_file_idx = 0;
}

Reader::Status CaffeLMDBRecordReader::folder_reading()
{
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("CaffeLMDBRecordReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

    std::string _full_path = _folder_path;
    auto ret = Reader::Status::OK;
    if (Caffe_LMDB_reader() != Reader::Status::OK)
        WRN("CaffeLMDBRecordReader ShardID [" + TOSTR(_shard_id) + "] CaffeLMDBRecordReader cannot access the storage at " + _folder_path);

    if (_in_batch_read_count > 0 && _in_batch_read_count < _batch_count)
    {
        replicate_last_image_to_fill_last_shard();
        std::cout << "CaffeLMDBRecordReader ShardID [" << TOSTR(_shard_id) << "] Replicated " << _folder_path + _last_file_name << " " << TOSTR((_batch_count - _in_batch_read_count)) << " times to fill the last batch" << std::endl;
    }
    if (!_file_names.empty())
        std::cout << "CaffeLMDBRecordReader ShardID [" << TOSTR(_shard_id) << "] Total of " << TOSTR(_file_names.size()) << " images loaded from " << _full_path << std::endl;
    closedir(_sub_dir);
    return ret;
}
void CaffeLMDBRecordReader::replicate_last_image_to_fill_last_shard()
{
    for (size_t i = _in_batch_read_count; i < _batch_count; i++)
    {
        _file_names.push_back(_last_file_name);
        _file_size.insert(pair<std::string, unsigned int>(_last_file_name, _last_file_size));
    }
}

Reader::Status CaffeLMDBRecordReader::Caffe_LMDB_reader()
{
    _open_env = 0;
    string tmp1 = _folder_path + "/data.mdb";
    string tmp2 = _folder_path + "/lock.mdb";
    uint file_size, file_size1;

    ifstream in_file(tmp1, ios::binary);
    in_file.seekg(0, ios::end);
    file_size = in_file.tellg();
    ifstream in_file1(tmp2, ios::binary);
    in_file1.seekg(0, ios::end);
    file_size1 = in_file1.tellg();
    _file_byte_size = file_size + file_size1;
    read_image_names();
    return Reader::Status::OK;
}

size_t CaffeLMDBRecordReader::get_file_shard_id()
{
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    return (_file_id / (_batch_count)) % _shard_count;
}

void CaffeLMDBRecordReader::read_image_names()
{
    int rc;
    // Creating an LMDB environment handle
    E(mdb_env_create(&_mdb_env));
    // Setting the size of the memory map to use for this environment.
    // The size of the memory map is also the maximum size of the database.
    E(mdb_env_set_mapsize(_mdb_env, _file_byte_size));
    // Opening an environment handle.
    E(mdb_env_open(_mdb_env, _path.c_str(), MDB_RDONLY, 0664));
    // Creating a transaction for use with the environment
    E(mdb_txn_begin(_mdb_env, NULL, MDB_RDONLY, &_mdb_txn));
    // Opening a database in the environment.
    E(mdb_open(_mdb_txn, NULL, 0, &_mdb_dbi));
    // Creating a cursor handle.
    // A cursor is associated with a specific transaction and database
    E(mdb_cursor_open(_mdb_txn, _mdb_dbi, &_mdb_cursor));

    // Retrieve by cursor. It retrieves key/data pairs from the database
    while ((rc = mdb_cursor_get(_mdb_cursor, &_mdb_key, &_mdb_value, MDB_NEXT)) == 0)
    {
        Datum datum;
        caffe_protos::AnnotatedDatum annotatedDatum_protos;
        annotatedDatum_protos.ParseFromArray((char *)_mdb_value.mv_data, _mdb_value.mv_size);
        // Checking image Datum
        int check_image_datum = annotatedDatum_protos.has_datum();

        if (check_image_datum)
            datum = annotatedDatum_protos.datum(); // parse datum for detection
        else
            datum.ParseFromArray((const void *)_mdb_value.mv_data, _mdb_value.mv_size); //parse datum for classification

        if (get_file_shard_id() != _shard_id)
        {
            incremenet_file_id();
            continue;
        }
        _in_batch_read_count++;
        _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;

        string image_key = string((char *)_mdb_key.mv_data);

        _file_names.push_back(image_key.c_str());
        _last_file_name = image_key.c_str();

        incremenet_file_id();

        _last_file_size = datum.data().size();
        _file_size.insert(pair<std::string, unsigned int>(_last_file_name, _last_file_size));
    }

    release();
}

void CaffeLMDBRecordReader::open_env_for_read_image() 
{
    // Creating an LMDB environment handle 
    E(mdb_env_create(&_read_mdb_env)); 
    // Setting the size of the memory map to use for this environment. 
    // The size of the memory map is also the maximum size of the database.
    E(mdb_env_set_mapsize(_read_mdb_env, _file_byte_size)); 
    // Opening an environment handle.
    E(mdb_env_open(_read_mdb_env, _path.c_str(), MDB_RDONLY, 0664)); 
    // Creating a transaction for use with the environment
    E(mdb_txn_begin(_read_mdb_env, NULL, MDB_RDONLY, &_read_mdb_txn));
    // Opening a database in the environment.
    E(mdb_open(_read_mdb_txn, NULL, 0, &_read_mdb_dbi));
    _open_env = 1;
}

void CaffeLMDBRecordReader::read_image(unsigned char *buff, std::string file_name)
{
    if(_open_env == 0)
    {
	open_env_for_read_image();
    }

    // Creating a cursor handle.
    // A cursor is associated with a specific transaction and database
    E(mdb_cursor_open(_read_mdb_txn, _read_mdb_dbi, &_read_mdb_cursor));

    string checkedStr = string((char *)file_name.c_str());
    string newStr = checkedStr.substr(0, checkedStr.find(".")) + ".JPEG";

    _read_mdb_key.mv_size = newStr.size();
    _read_mdb_key.mv_data = (char *)newStr.c_str();
    
    int _mdb_status = mdb_cursor_get(_read_mdb_cursor, &_read_mdb_key, &_read_mdb_value, MDB_SET_RANGE);
    if(_mdb_status == MDB_NOTFOUND) {
        THROW("\nKey Not found");
    }
    else
    {
        Datum datum;
        caffe_protos::AnnotatedDatum annotatedDatum_protos;
        annotatedDatum_protos.ParseFromArray((char *)_read_mdb_value.mv_data, _read_mdb_value.mv_size);

        // Checking image Datum
        int check_image_datum = annotatedDatum_protos.has_datum();

        if (check_image_datum)
            datum = annotatedDatum_protos.datum(); // parse datum for detection
        else
            datum.ParseFromArray((const void *)_read_mdb_value.mv_data, _read_mdb_value.mv_size); //parse datum for classification
            
        memcpy(buff, datum.data().c_str(), datum.data().size());

    }
    
    mdb_cursor_close(_read_mdb_cursor);
    _read_mdb_cursor = nullptr;
}
