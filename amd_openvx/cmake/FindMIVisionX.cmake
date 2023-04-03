################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 - 2023 Advanced Micro Devices, Inc.
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
if(APPLE)
    set(SHARED_LIB_TYPE ".dylib")
else()
    set(SHARED_LIB_TYPE ".so")
endif()

find_path(MIVisionX_INCLUDE_DIRS
    NAMES vx_ext_amd.h
    HINTS
    $ENV{MIVisionX_PATH}/include/mivisionx
    PATHS
    ${MIVisionX_PATH}/include/mivisionx
    /usr/include
    ${ROCM_PATH}/include/mivisionx
)
mark_as_advanced(MIVisionX_INCLUDE_DIRS)

# OpenVX
find_library(OPENVX_LIBRARIES
    NAMES libopenvx${SHARED_LIB_TYPE}
    HINTS
    $ENV{MIVisionX_PATH}/lib
    PATHS
    ${MIVisionX_PATH}/lib
    /usr/lib
    ${ROCM_PATH}/lib
)
mark_as_advanced(OPENVX_LIBRARIES)

# VX_RPP
find_library(VXRPP_LIBRARIES
    NAMES libvx_rpp${SHARED_LIB_TYPE}
    HINTS
    $ENV{MIVisionX_PATH}/lib
    PATHS
    ${MIVisionX_PATH}/lib
    /usr/lib
    ${ROCM_PATH}/lib
)
mark_as_advanced(VXRPP_LIBRARIES)

find_path(MIVisionX_LIBRARIES_DIRS
    NAMES libopenvx${SHARED_LIB_TYPE}
    HINTS
    $ENV{ROCM_PATH}/lib
    $ENV{ROCM_PATH}/lib64
    $ENV{MIVisionX_PATH}/lib
    PATHS
    ${MIVisionX_PATH}/lib
    /usr/lib
    ${ROCM_PATH}/lib
)
mark_as_advanced(MIVisionX_LIBRARIES_DIRS)

if(OPENVX_LIBRARIES AND MIVisionX_INCLUDE_DIRS)
    set(MIVisionX_FOUND TRUE)
endif( )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( MIVisionX 
    FOUND_VAR  MIVisionX_FOUND 
    REQUIRED_VARS
        OPENVX_LIBRARIES
        VXRPP_LIBRARIES  
        MIVisionX_INCLUDE_DIRS
        MIVisionX_LIBRARIES_DIRS
)

set(MIVisionX_FOUND ${MIVisionX_FOUND} CACHE INTERNAL "")
set(OPENVX_LIBRARIES ${OPENVX_LIBRARIES} CACHE INTERNAL "")
set(VXRPP_LIBRARIES ${VXRPP_LIBRARIES} CACHE INTERNAL "")
set(MIVisionX_INCLUDE_DIRS ${MIVisionX_INCLUDE_DIRS} CACHE INTERNAL "")
set(MIVisionX_LIBRARIES_DIRS ${MIVisionX_LIBRARIES_DIRS} CACHE INTERNAL "")

if(MIVisionX_FOUND)
    message("-- ${White}Using MIVisionX -- \n\tLibraries:${OPENVX_LIBRARIES} \n\tIncludes:${MIVisionX_INCLUDE_DIRS}${ColourReset}")    
else()
    if(MIVisionX_FIND_REQUIRED)
        message(FATAL_ERROR "{Red}FindMIVisionX -- NOT FOUND${ColourReset}")
    endif()
    message( "-- ${Yellow}NOTE: FindMIVisionX failed to find -- openvx${ColourReset}" )
endif()