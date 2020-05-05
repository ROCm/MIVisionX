#pragma once

#include <exception>
#include "reader.h"

std::shared_ptr<Reader> create_reader(ReaderConfig config);
