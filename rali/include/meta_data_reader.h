#pragma once

#include <string>
#include <memory>
#include "meta_data.h"

class MetaDataReader
{
public:
    virtual ~MetaDataReader()= default;
    virtual void init(const MetaDataConfig& cfg) = 0;
    virtual void read_all(const std::string& path) = 0;// Reads all the meta data information
    virtual void lookup(const std::vector<std::string>& image_names) = 0;// finds meta_data info associated with given names and fills the output
    virtual void release() = 0; // Deletes the loaded information
    virtual MetaDataBatch * get_output()= 0;

};

