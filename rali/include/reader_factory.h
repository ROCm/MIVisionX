#pragma once

#include <exception>
#include "file_source_reader.h"

std::shared_ptr<Reader> create_reader(ReaderConfig config);
