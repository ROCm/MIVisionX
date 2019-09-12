#pragma once

#include "loader_module.h"
/*!
 * Implements the functionality required for the loading
 * input images from a folder on the file system
 */
class JpegFileLoaderConfig : public LoaderModuleConfig {
public:
    JpegFileLoaderConfig(size_t batch_size, RaliMemType mem_type):
    LoaderModuleConfig( batch_size, mem_type),
    path(""){}
    JpegFileLoaderConfig(size_t batch_size, RaliMemType mem_type, const std::string&  folder_path):
    LoaderModuleConfig( batch_size, mem_type),   
    path(folder_path) {}
    StorageType storage_type() override { return StorageType::FILE_SYSTEM;}
    DecoderType decoder_type() override { return DecoderType::TURBO_JPEG;}
    std::string path;
};

class RecordIOLoaderConfig : public LoaderModuleConfig {
public:	
    RecordIOLoaderConfig( size_t batch_size, RaliMemType mem_type, const std::string&  file_path):
    LoaderModuleConfig( batch_size, mem_type),
    path(file_path) {}
    StorageType storage_type() override { return StorageType::RECORDIO;}
    DecoderType decoder_type() override { return DecoderType::TURBO_JPEG;}
    std::string path;
    //TODO: When adding this type of reader add all required fields, if any
};

class TFRecordLoaderConfig : public LoaderModuleConfig {
public:	
    TFRecordLoaderConfig( size_t batch_size, RaliMemType mem_type, const std::string&  file_path):
    LoaderModuleConfig( batch_size, mem_type),
    path(file_path) {}
    StorageType storage_type() override { return StorageType::TFRecord;}
    DecoderType decoder_type() override { return DecoderType::TURBO_JPEG;}
    std::string path;
    //TODO: When adding this type of reader add all required fields
};

class LMDBLoaderConfig : public LoaderModuleConfig {
public:	
    LMDBLoaderConfig( size_t batch_size, RaliMemType mem_type, const std::string& file_path):
    LoaderModuleConfig( batch_size, mem_type), 
    path(file_path) {}
    StorageType storage_type() override { return StorageType::LMDB;}
    DecoderType decoder_type() override { return DecoderType::TURBO_JPEG;}
    std::string path;
    //TODO: When adding this type of reader add all required fields
};