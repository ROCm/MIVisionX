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
find_path(AMDRPP_INCLUDE_DIRS
    NAMES rpp.h
    PATHS
    /usr/include
    ${ROCM_PATH}/rpp/include
)
mark_as_advanced( AMDRPP_INCLUDE_DIRS )

find_library( AMDRPP_LIBRARIES
    NAMES amd_rpp
    PATHS
    /usr/lib
    ${ROCM_PATH}/rpp/lib
)
mark_as_advanced( AMDRPP_LIBRARIES_DIR )

find_path(AMDRPP_LIBRARIES_DIR
    NAMES libamd_rpp.so
    PATHS
    /usr/lib
    ${ROCM_PATH}/rpp/lib
)
    
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( AMDRPP 
    FOUND_VAR  AMDRPP_FOUND 
    REQUIRED_VARS
        AMDRPP_LIBRARIES 
        AMDRPP_INCLUDE_DIRS 
)

set(AMDRPP_FOUND ${AMDRPP_FOUND} CACHE INTERNAL "")
set(AMDRPP_LIBRARIES ${AMDRPP_LIBRARIES} CACHE INTERNAL "")
set(AMDRPP_INCLUDE_DIRS ${AMDRPP_INCLUDE_DIRS} CACHE INTERNAL "")
set(AMDRPP_LIBRARIES_DIR ${AMDRPP_LIBRARIES_DIR} CACHE INTERNAL "")

if( NOT AMDRPP_FOUND )
    message( "-- ${Yellow}FindAMDRPP failed to find: amd_rpp${ColourReset}" )
endif()