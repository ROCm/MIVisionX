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
#include <vector>
#include <string>
#include <memory>
#include "reader.h"


class TFRecordReader : public Reader
        {
public:
    //! Reads the TFRecord File, and loads the image ids and other necessary info
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override { return Reader::Status::OK; }
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded images
     \return Size of the loaded resource
    */
    size_t read(unsigned char* buf, size_t max_size) override { return 0; }
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override { return 0; }

    //! Resets the object's state to read from the first file in the folder
    void reset() override { return ; }

    //! Returns the id of the latest file opened
    std::string id() override { return "0";}

    unsigned count() override { return 0; }

    ~TFRecordReader() override = default;

    int close() override{ return 0; };

    TFRecordReader()= default;

};