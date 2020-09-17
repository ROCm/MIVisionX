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
#include "caffe2_lmdb_record_reader.h"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdint.h>
using namespace std;
namespace filesys = boost::filesystem;

Caffe2LMDBRecordReader::Caffe2LMDBRecordReader():
_shuffle_time("shuffle_time", DBG_TIMING)
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

unsigned Caffe2LMDBRecordReader::count()
{
    if(_loop)
        return _file_names.size();

    int ret = ((int)_file_names.size() -_read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status Caffe2LMDBRecordReader::initialize(ReaderConfig desc)
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

void Caffe2LMDBRecordReader::incremenet_read_ptr()
{
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
}
size_t Caffe2LMDBRecordReader::open()
{
    auto file_path = _file_names[_curr_file_idx];// Get next file name
    _last_id = file_path;
    _current_file_size = _file_size[_file_names[_curr_file_idx]];
    return _current_file_size;
}

size_t Caffe2LMDBRecordReader::read(unsigned char* buf, size_t read_size)
{
    read_image(buf, _file_names[_curr_file_idx]);
    incremenet_read_ptr();
    return  read_size;

}

int Caffe2LMDBRecordReader::close()
{
    return release();
}

Caffe2LMDBRecordReader::~Caffe2LMDBRecordReader()
{
    _open_env = 0; 
    mdb_txn_abort(_read_mdb_txn);
    mdb_close(_read_mdb_env, _read_mdb_dbi);
    mdb_env_close(_read_mdb_env);
    _read_mdb_txn = nullptr;
    _read_mdb_env = nullptr;
    release();
}

int
Caffe2LMDBRecordReader::release()
{
    return 0;
}

void Caffe2LMDBRecordReader::reset()
{
    _shuffle_time.start();
    if(_shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());
    _shuffle_time.end();
    _read_counter = 0;
    _curr_file_idx = 0;
}

Reader::Status Caffe2LMDBRecordReader::folder_reading()
{
    if ((_sub_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("Caffe2LMDBRecordReader ShardID ["+ TOSTR(_shard_id)+ "] ERROR: Failed opening the directory at " + _folder_path);

    std::string _full_path = _folder_path;
    auto ret = Reader::Status::OK;
    if(Caffe2_LMDB_reader() != Reader::Status::OK)
        WRN("Caffe2LMDBRecordReader ShardID ["+ TOSTR(_shard_id)+ "] Caffe2LMDBRecordReader cannot access the storage at " + _folder_path);

    if(_in_batch_read_count > 0 && _in_batch_read_count < _batch_count)
    {
        replicate_last_image_to_fill_last_shard();
        LOG("Caffe2LMDBRecordReader ShardID [" + TOSTR(_shard_id) + "] Replicated " + _folder_path+_last_file_name + " " + TOSTR((_batch_count - _in_batch_read_count) ) + " times to fill the last batch")
    }
    if(!_file_names.empty())
        LOG("Caffe2LMDBRecordReader ShardID ["+ TOSTR(_shard_id)+ "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path )
    closedir(_sub_dir);
    return ret;
}

void Caffe2LMDBRecordReader::replicate_last_image_to_fill_last_shard()
{
    for(size_t i = _in_batch_read_count; i < _batch_count; i++)
    {
        _file_names.push_back(_last_file_name);
        _file_size.insert(pair<std::string, unsigned int>(_last_file_name, _last_file_size));
    }
}

Reader::Status Caffe2LMDBRecordReader::Caffe2_LMDB_reader()
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

size_t Caffe2LMDBRecordReader::get_file_shard_id()
{
    if(_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    return (_file_id / (_batch_count)) % _shard_count;
}


void Caffe2LMDBRecordReader::read_image_names()
{
    int rc;
    MDB_env *env;
    MDB_dbi dbi;
    MDB_val key, data;
    MDB_txn *txn;
    MDB_cursor *cursor;
    string str_key, str_data;

    // Creating an LMDB environment handle
    E(mdb_env_create(&env));
    // Setting the size of the memory map to use for this environment.
    // The size of the memory map is also the maximum size of the database. 
    E(mdb_env_set_mapsize(env, _file_byte_size));
    // Opening an environment handle.
    E(mdb_env_open(env, _folder_path.c_str(), 0, 0664));
    // Creating a transaction for use with the environment. 
    E(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
    // Opening a database in the environment. 
    E(mdb_dbi_open(txn, NULL, 0, &dbi));

    // Creating a cursor handle.
    // A cursor is associated with a specific transaction and database
    E(mdb_cursor_open(txn, dbi, &cursor));

    // Retrieve by cursor. It retrieves key/data pairs from the database
    while((rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT)) == 0)
    {	
        str_key = string((char *) key.mv_data);
        if(get_file_shard_id() != _shard_id )
        {
            incremenet_file_id();
            continue;
        }
        _in_batch_read_count++;
        _in_batch_read_count = (_in_batch_read_count%_batch_count == 0) ? 0 : _in_batch_read_count;
        str_data = string((char *) data.mv_data);
        caffe2_protos::TensorProtos tens_protos;
        tens_protos.ParseFromArray((char *)data.mv_data, data.mv_size);
        int protos_size = tens_protos.protos_size();   
        if(protos_size != 0)
        {
            caffe2_protos::TensorProto image_proto = tens_protos.protos(0);
            bool chk_byte_data = image_proto.has_byte_data();
            if(chk_byte_data)
            {
                _file_names.push_back(str_key.c_str());
                _last_file_name = str_key.c_str();
                _last_file_size = image_proto.byte_data().size();
	        _file_size.insert(pair<std::string, unsigned int>(_last_file_name, _last_file_size));
            }
            else
            {
                THROW("\n Image parsing failed");
            }
        }   
        incremenet_file_id();
    }

    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    mdb_close(env, dbi);
    mdb_env_close(env);
}

void Caffe2LMDBRecordReader::open_env_for_read_image()
{
    // Creating an LMDB environment handle
    E(mdb_env_create(&_read_mdb_env));
    // Setting the size of the memory map to use for this environment.
    E(mdb_env_set_mapsize(_read_mdb_env, _file_byte_size)); 
    // The size of the memory map is also the maximum size of the database. 
    // Opening an environment handle.
    E(mdb_env_open(_read_mdb_env, _folder_path.c_str(), MDB_RDONLY, 0664));
    // Creating a transaction for use with the environment. 
    E(mdb_txn_begin(_read_mdb_env, NULL, MDB_RDONLY, &_read_mdb_txn));
    // Opening a database in the environment. 
    E(mdb_open(_read_mdb_txn, NULL, 0, &_read_mdb_dbi)); 
    _open_env = 1; 
}

void Caffe2LMDBRecordReader::read_image(unsigned char* buff, std::string file_name)
{

    if(_open_env == 0)
        open_env_for_read_image(); 

    // Creating a cursor handle.
    // A cursor is associated with a specific transaction and database
    E(mdb_cursor_open(_read_mdb_txn, _read_mdb_dbi, &_read_mdb_cursor)); 

    string checkedStr = string((char *)file_name.c_str());
    string newStr = checkedStr.substr(0, checkedStr.find(".")) + ".JPEG"; 
   
    _read_mdb_key.mv_size = newStr.size();
    _read_mdb_key.mv_data = (char *)newStr.c_str();
    
    int _mdb_status = mdb_cursor_get(_read_mdb_cursor, &_read_mdb_key, &_read_mdb_value, MDB_SET_RANGE); 
    if(_mdb_status == MDB_NOTFOUND) {
	    THROW("Key Not found");
    }
    else 
    {
            // Parsing Image and Label Protos using the key and data values
            // read from LMDB records
            caffe2_protos::TensorProtos tens_protos;

            tens_protos.ParseFromArray((char *)_read_mdb_value.mv_data, _read_mdb_value.mv_size);

            // Checking size of the protos
            int protos_size = tens_protos.protos_size();   
            if(protos_size != 0)
            { 
                caffe2_protos::TensorProto image_proto = tens_protos.protos(0);
                // Checking if image bytes is present or not
                bool chk_byte_data = image_proto.has_byte_data();
                
                if(chk_byte_data)
                {
                    memcpy(buff, image_proto.byte_data().c_str(), image_proto.byte_data().size());
                }
                else
                {
                    THROW("Image Parsing Failed");
                }
            }
            else
            {
                THROW("Parsing Protos Failed");
            }
    }
    // Closing cursor handles
    mdb_cursor_close(_read_mdb_cursor);
    _read_mdb_cursor = nullptr;
}
