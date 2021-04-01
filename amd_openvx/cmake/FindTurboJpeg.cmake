################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 - 2020 Advanced Micro Devices, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
################################################################################
find_path(TurboJpeg_INCLUDE_DIRS
    NAMES turbojpeg.h
    PATHS
    /usr/include/
)
mark_as_advanced( TurboJpeg_INCLUDE_DIRS )

find_library( TurboJpeg_LIBRARIES
    NAMES libturbojpeg.so
    PATHS
    /usr/lib
)
mark_as_advanced( TurboJpeg_LIBRARIES_DIR )

find_path(TurboJpeg_LIBRARIES_DIR
    NAMES libturbojpeg.so
    PATHS
    /usr/lib
)

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( TurboJpeg 
    FOUND_VAR  TurboJpeg_FOUND 
    REQUIRED_VARS
        TurboJpeg_LIBRARIES 
        TurboJpeg_INCLUDE_DIRS
        TurboJpeg_LIBRARIES_DIR
)

set(TurboJpeg_FOUND ${TurboJpeg_FOUND} CACHE INTERNAL "")
set(TurboJpeg_LIBRARIES ${TurboJpeg_LIBRARIES} CACHE INTERNAL "")
set(TurboJpeg_INCLUDE_DIRS ${TurboJpeg_INCLUDE_DIRS} CACHE INTERNAL "")
set(TurboJpeg_LIBRARIES_DIR ${TurboJpeg_LIBRARIES_DIR} CACHE INTERNAL "")

if( NOT TurboJpeg_FOUND )
    message( "-- ${Yellow}FindTurboJpeg failed to find: turbojpeg${ColourReset}" )
endif()