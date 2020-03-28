#pragma once
#include "meta_data_reader.h"
std::shared_ptr<MetaDataReader> create_meta_data_manager(const MetaDataConfig& config);
