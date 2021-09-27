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

#ifndef RAPIDJSON_FILEWRITESTREAM_H_
#define RAPIDJSON_FILEWRITESTREAM_H_

#include "stream.h"
#include <cstdio>

#ifdef __clang__
RAPIDJSON_DIAG_PUSH
RAPIDJSON_DIAG_OFF(unreachable-code)
#endif

RAPIDJSON_NAMESPACE_BEGIN

//! Wrapper of C file stream for output using fwrite().
/*!
    \note implements Stream concept
*/
class FileWriteStream {
public:
    typedef char Ch;    //!< Character type. Only support char.

    FileWriteStream(std::FILE* fp, char* buffer, size_t bufferSize) : fp_(fp), buffer_(buffer), bufferEnd_(buffer + bufferSize), current_(buffer_) { 
        RAPIDJSON_ASSERT(fp_ != 0);
    }

    void Put(char c) { 
        if (current_ >= bufferEnd_)
            Flush();

        *current_++ = c;
    }

    void PutN(char c, size_t n) {
        size_t avail = static_cast<size_t>(bufferEnd_ - current_);
        while (n > avail) {
            std::memset(current_, c, avail);
            current_ += avail;
            Flush();
            n -= avail;
            avail = static_cast<size_t>(bufferEnd_ - current_);
        }

        if (n > 0) {
            std::memset(current_, c, n);
            current_ += n;
        }
    }

    void Flush() {
        if (current_ != buffer_) {
            size_t result = std::fwrite(buffer_, 1, static_cast<size_t>(current_ - buffer_), fp_);
            if (result < static_cast<size_t>(current_ - buffer_)) {
                // failure deliberately ignored at this time
                // added to avoid warn_unused_result build errors
            }
            current_ = buffer_;
        }
    }

    // Not implemented
    char Peek() const { RAPIDJSON_ASSERT(false); return 0; }
    char Take() { RAPIDJSON_ASSERT(false); return 0; }
    size_t Tell() const { RAPIDJSON_ASSERT(false); return 0; }
    char* PutBegin() { RAPIDJSON_ASSERT(false); return 0; }
    size_t PutEnd(char*) { RAPIDJSON_ASSERT(false); return 0; }

private:
    // Prohibit copy constructor & assignment operator.
    FileWriteStream(const FileWriteStream&);
    FileWriteStream& operator=(const FileWriteStream&);

    std::FILE* fp_;
    char *buffer_;
    char *bufferEnd_;
    char *current_;
};

//! Implement specialized version of PutN() with memset() for better performance.
template<>
inline void PutN(FileWriteStream& stream, char c, size_t n) {
    stream.PutN(c, n);
}

RAPIDJSON_NAMESPACE_END

#ifdef __clang__
RAPIDJSON_DIAG_POP
#endif

#endif // RAPIDJSON_FILESTREAM_H_
