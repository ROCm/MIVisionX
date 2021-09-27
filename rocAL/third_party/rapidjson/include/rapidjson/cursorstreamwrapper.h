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

// Tencent is pleased to support the open source community by making RapidJSON available.
//
// Copyright (C) 2015 THL A29 Limited, a Tencent company, and Milo Yip.
//
// Licensed under the MIT License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// http://opensource.org/licenses/MIT
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef RAPIDJSON_CURSORSTREAMWRAPPER_H_
#define RAPIDJSON_CURSORSTREAMWRAPPER_H_

#include "stream.h"

#if defined(__GNUC__)
RAPIDJSON_DIAG_PUSH
RAPIDJSON_DIAG_OFF(effc++)
#endif

#if defined(_MSC_VER) && _MSC_VER <= 1800
RAPIDJSON_DIAG_PUSH
RAPIDJSON_DIAG_OFF(4702)  // unreachable code
RAPIDJSON_DIAG_OFF(4512)  // assignment operator could not be generated
#endif

RAPIDJSON_NAMESPACE_BEGIN


//! Cursor stream wrapper for counting line and column number if error exists.
/*!
    \tparam InputStream     Any stream that implements Stream Concept
*/
template <typename InputStream, typename Encoding = UTF8<> >
class CursorStreamWrapper : public GenericStreamWrapper<InputStream, Encoding> {
public:
    typedef typename Encoding::Ch Ch;

    CursorStreamWrapper(InputStream& is):
        GenericStreamWrapper<InputStream, Encoding>(is), line_(1), col_(0) {}

    // counting line and column number
    Ch Take() {
        Ch ch = this->is_.Take();
        if(ch == '\n') {
            line_ ++;
            col_ = 0;
        } else {
            col_ ++;
        }
        return ch;
    }

    //! Get the error line number, if error exists.
    size_t GetLine() const { return line_; }
    //! Get the error column number, if error exists.
    size_t GetColumn() const { return col_; }

private:
    size_t line_;   //!< Current Line
    size_t col_;    //!< Current Column
};

#if defined(_MSC_VER) && _MSC_VER <= 1800
RAPIDJSON_DIAG_POP
#endif

#if defined(__GNUC__)
RAPIDJSON_DIAG_POP
#endif

RAPIDJSON_NAMESPACE_END

#endif // RAPIDJSON_CURSORSTREAMWRAPPER_H_
