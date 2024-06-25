################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc.
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

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args( MIVISIONX 
    FOUND_VAR  MIVISIONX_FOUND 
    REQUIRED_VARS
        OPENVX_LIBRARY
        VX_RPP_LIBRARY  
        MIVISIONX_INCLUDE_DIR
        MIVISIONX_LIBRARIES
)

# find OpenVX
find_library(OPENVX_LIBRARY NAMES openvx HINTS ${ROCM_PATH}/lib)
find_library(OPENVXU_LIBRARY NAMES vxu HINTS ${ROCM_PATH}/lib)
find_path(OPENVX_INCLUDE_DIR NAMES VX/vx.h PATHS ${ROCM_PATH}/include/mivisionx)

if(OPENVX_LIBRARY AND OPENVXU_LIBRARY AND OPENVX_INCLUDE_DIR)
    set(OPENVX_FOUND TRUE)
    set(MIVISIONX_FOUND TRUE CACHE INTERNAL "")
    set(MIVISIONX_INCLUDE_DIR ${OPENVX_INCLUDE_DIR} CACHE INTERNAL "")
    message("-- ${White}FindMIVISIONX: Using OpenVX -- \n\tLibraries:${OPENVX_LIBRARY} \n\tIncludes:${OPENVX_INCLUDE_DIR}${ColourReset}")
else()
    message("-- ${Yellow}FindMIVISIONX: OpenVX Libraries Not Found")
endif()

# find VX_RPP
find_library(VX_RPP_LIBRARY NAMES vx_rpp HINTS ${ROCM_PATH}/lib)
find_path(VX_RPP_INCLUDE_DIR NAMES vx_ext_rpp.h PATHS ${ROCM_PATH}/include/mivisionx)

if(VX_RPP_LIBRARY AND VX_RPP_INCLUDE_DIR)
    set(VX_RPP_FOUND TRUE)
    message("-- ${White}FindMIVISIONX: Using VX_RPP -- \n\tLibraries:${VX_RPP_LIBRARY}${ColourReset}")
else()
    message("-- ${Yellow}FindMIVISIONX: VX_RPP Libraries Not Found")
endif()

set(MIVISIONX_FOUND ${MIVISIONX_FOUND} CACHE INTERNAL "")
set(OPENVX_LIBRARIES ${OPENVX_LIBRARIES} CACHE INTERNAL "")
set(VXRPP_LIBRARIES ${VXRPP_LIBRARIES} CACHE INTERNAL "")
set(MIVISIONX_INCLUDE_DIRS ${MIVISIONX_INCLUDE_DIRS} CACHE INTERNAL "")

if(MIVISIONX_FOUND)
    message("-- ${White}Using MIVISIONX -- \n\tLibraries:${OPENVX_LIBRARIES} \n\tIncludes:${MIVISIONX_INCLUDE_DIRS}${ColourReset}")    
else()
    if(MIVISIONX_FIND_REQUIRED)
        message(FATAL_ERROR "{Red}FindMIVISIONX -- NOT FOUND${ColourReset}")
    endif()
    message( "-- ${Yellow}NOTE: FindMIVISIONX failed to find -- openvx${ColourReset}" )
endif()