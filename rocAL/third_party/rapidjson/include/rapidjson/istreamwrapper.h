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
/*
Tencent is pleased to support the open source community by making RapidJSON available.

Copyright (C) 2015 THL A29 Limited, a Tencent company, and Milo Yip.

Licensed under the MIT License (the "License"); you may not use this file except
in compliance with the License. You may obtain a copy of the License at

http://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software distributed 
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the 
specific language governing permissions and limitations under the License.
*/
#ifndef RAPIDJSON_ISTREAMWRAPPER_H_
#define RAPIDJSON_ISTREAMWRAPPER_H_

#include "stream.h"
#include <iosfwd>
#include <ios>

#ifdef __clang__
RAPIDJSON_DIAG_PUSH
RAPIDJSON_DIAG_OFF(padded)
#elif defined(_MSC_VER)
RAPIDJSON_DIAG_PUSH
RAPIDJSON_DIAG_OFF(4351) // new behavior: elements of array 'array' will be default initialized
#endif

RAPIDJSON_NAMESPACE_BEGIN

//! Wrapper of \c std::basic_istream into RapidJSON's Stream concept.
/*!
    The classes can be wrapped including but not limited to:

    - \c std::istringstream
    - \c std::stringstream
    - \c std::wistringstream
    - \c std::wstringstream
    - \c std::ifstream
    - \c std::fstream
    - \c std::wifstream
    - \c std::wfstream

    \tparam StreamType Class derived from \c std::basic_istream.
*/
   
template <typename StreamType>
class BasicIStreamWrapper {
public:
    typedef typename StreamType::char_type Ch;

    //! Constructor.
    /*!
        \param stream stream opened for read.
    */
    BasicIStreamWrapper(StreamType &stream) : stream_(stream), buffer_(peekBuffer_), bufferSize_(4), bufferLast_(0), current_(buffer_), readCount_(0), count_(0), eof_(false) { 
        Read();
    }

    //! Constructor.
    /*!
        \param stream stream opened for read.
        \param buffer user-supplied buffer.
        \param bufferSize size of buffer in bytes. Must >=4 bytes.
    */
    BasicIStreamWrapper(StreamType &stream, char* buffer, size_t bufferSize) : stream_(stream), buffer_(buffer), bufferSize_(bufferSize), bufferLast_(0), current_(buffer_), readCount_(0), count_(0), eof_(false) { 
        RAPIDJSON_ASSERT(bufferSize >= 4);
        Read();
    }

    Ch Peek() const { return *current_; }
    Ch Take() { Ch c = *current_; Read(); return c; }
    size_t Tell() const { return count_ + static_cast<size_t>(current_ - buffer_); }

    // Not implemented
    void Put(Ch) { RAPIDJSON_ASSERT(false); }
    void Flush() { RAPIDJSON_ASSERT(false); } 
    Ch* PutBegin() { RAPIDJSON_ASSERT(false); return 0; }
    size_t PutEnd(Ch*) { RAPIDJSON_ASSERT(false); return 0; }

    // For encoding detection only.
    const Ch* Peek4() const {
        return (current_ + 4 - !eof_ <= bufferLast_) ? current_ : 0;
    }

private:
    BasicIStreamWrapper();
    BasicIStreamWrapper(const BasicIStreamWrapper&);
    BasicIStreamWrapper& operator=(const BasicIStreamWrapper&);

    void Read() {
        if (current_ < bufferLast_)
            ++current_;
        else if (!eof_) {
            count_ += readCount_;
            readCount_ = bufferSize_;
            bufferLast_ = buffer_ + readCount_ - 1;
            current_ = buffer_;

            if (!stream_.read(buffer_, static_cast<std::streamsize>(bufferSize_))) {
                readCount_ = static_cast<size_t>(stream_.gcount());
                *(bufferLast_ = buffer_ + readCount_) = '\0';
                eof_ = true;
            }
        }
    }

    StreamType &stream_;
    Ch peekBuffer_[4], *buffer_;
    size_t bufferSize_;
    Ch *bufferLast_;
    Ch *current_;
    size_t readCount_;
    size_t count_;  //!< Number of characters read
    bool eof_;
};

typedef BasicIStreamWrapper<std::istream> IStreamWrapper;
typedef BasicIStreamWrapper<std::wistream> WIStreamWrapper;

#if defined(__clang__) || defined(_MSC_VER)
RAPIDJSON_DIAG_POP
#endif

RAPIDJSON_NAMESPACE_END

#endif // RAPIDJSON_ISTREAMWRAPPER_H_
