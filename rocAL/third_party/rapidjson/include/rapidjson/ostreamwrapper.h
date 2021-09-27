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

#ifndef RAPIDJSON_OSTREAMWRAPPER_H_
#define RAPIDJSON_OSTREAMWRAPPER_H_

#include "stream.h"
#include <iosfwd>

#ifdef __clang__
RAPIDJSON_DIAG_PUSH
RAPIDJSON_DIAG_OFF(padded)
#endif

RAPIDJSON_NAMESPACE_BEGIN

//! Wrapper of \c std::basic_ostream into RapidJSON's Stream concept.
/*!
    The classes can be wrapped including but not limited to:

    - \c std::ostringstream
    - \c std::stringstream
    - \c std::wpstringstream
    - \c std::wstringstream
    - \c std::ifstream
    - \c std::fstream
    - \c std::wofstream
    - \c std::wfstream

    \tparam StreamType Class derived from \c std::basic_ostream.
*/
   
template <typename StreamType>
class BasicOStreamWrapper {
public:
    typedef typename StreamType::char_type Ch;
    BasicOStreamWrapper(StreamType& stream) : stream_(stream) {}

    void Put(Ch c) {
        stream_.put(c);
    }

    void Flush() {
        stream_.flush();
    }

    // Not implemented
    char Peek() const { RAPIDJSON_ASSERT(false); return 0; }
    char Take() { RAPIDJSON_ASSERT(false); return 0; }
    size_t Tell() const { RAPIDJSON_ASSERT(false); return 0; }
    char* PutBegin() { RAPIDJSON_ASSERT(false); return 0; }
    size_t PutEnd(char*) { RAPIDJSON_ASSERT(false); return 0; }

private:
    BasicOStreamWrapper(const BasicOStreamWrapper&);
    BasicOStreamWrapper& operator=(const BasicOStreamWrapper&);

    StreamType& stream_;
};

typedef BasicOStreamWrapper<std::ostream> OStreamWrapper;
typedef BasicOStreamWrapper<std::wostream> WOStreamWrapper;

#ifdef __clang__
RAPIDJSON_DIAG_POP
#endif

RAPIDJSON_NAMESPACE_END

#endif // RAPIDJSON_OSTREAMWRAPPER_H_
