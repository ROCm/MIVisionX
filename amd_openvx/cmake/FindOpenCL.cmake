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

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args(
    OpenCL
    FOUND_VAR OpenCL_FOUND
    REQUIRED_VARS
        OpenCL_LIBRARIES
        OpenCL_INCLUDE_DIRS
        CL_TARGET_OPENCL_VERSION
    VERSION_VAR OpenCL_VERSION
)

if(OpenCL_LIBRARIES AND OpenCL_INCLUDE_DIRS)
    set(OpenCL_FOUND TRUE)
    add_definitions(-DCL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION})
else()
    find_path(OPENCL_INCLUDE_DIRS
        NAMES OpenCL/cl.h CL/cl.h
        HINTS
        ${OPENCL_ROOT}/include
        $ENV{AMDAPPSDKROOT}/include
        $ENV{CUDA_PATH}/include
        PATHS
        ${ROCM_PATH}/opencl/include
        /usr/include
        /usr/local/include
        /usr/local/cuda/include
        /opt/cuda/include
        DOC "OpenCL header file path"
    )
    mark_as_advanced( OPENCL_INCLUDE_DIRS )

    if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        find_library( OPENCL_LIBRARIES
            NAMES OpenCL
            HINTS
            ${OPENCL_ROOT}/lib
            $ENV{AMDAPPSDKROOT}/lib
            $ENV{CUDA_PATH}/lib
            DOC "OpenCL dynamic library path"
            PATH_SUFFIXES x86_64 x64 x86_64/sdk
            PATHS
            ${ROCM_PATH}/opencl/lib/
            /usr/lib
            /usr/local/cuda/lib
            /opt/cuda/lib
        )
    else( )
        find_library( OPENCL_LIBRARIES
            NAMES OpenCL
            HINTS
            ${OPENCL_ROOT}/lib
            $ENV{AMDAPPSDKROOT}/lib
            $ENV{CUDA_PATH}/lib
            DOC "OpenCL dynamic library path"
            PATH_SUFFIXES x86 Win32
            PATHS
            ${ROCM_PATH}/opencl/lib/
            /usr/lib
            /usr/local/cuda/lib
            /opt/cuda/lib
        )
    endif( )
    mark_as_advanced( OPENCL_LIBRARIES )

    if(OPENCL_LIBRARIES AND OPENCL_INCLUDE_DIRS)
        set(OPENCL_FOUND TRUE)
    endif( )

    set(OpenCL_FOUND ${OPENCL_FOUND} CACHE INTERNAL "")
    set(OpenCL_LIBRARIES ${OPENCL_LIBRARIES} CACHE INTERNAL "")
    set(OpenCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS} CACHE INTERNAL "")

    if(EXISTS "${ROCM_PATH}/opencl/lib/libOpenCL.so")
        if(NOT "${OPENCL_LIBRARIES}" STREQUAL "${ROCM_PATH}/opencl/lib/libOpenCL.so")
            message("-- ${Magenta}ROCm OpenCL Found - Force OpenCL_LIBRARIES & OpenCL_INCLUDE_DIRS to use ROCm OpenCL${ColourReset}")
            set(OpenCL_LIBRARIES ${ROCM_PATH}/opencl/lib/libOpenCL.so CACHE INTERNAL "")
            set(OpenCL_INCLUDE_DIRS ${ROCM_PATH}/opencl/include CACHE INTERNAL "")
        endif()
    endif()

    if(OpenCL_FOUND)
        execute_process(
            COMMAND bash -c "nm -gDC ${OpenCL_LIBRARIES} | grep OPENCL_2.2"
            OUTPUT_VARIABLE outVar
        )
        if(NOT ${outVar} STREQUAL "")
            set(CL_TARGET_OPENCL_VERSION 220 CACHE INTERNAL "")
        else()
            message( "-- ${Yellow}FindOpenCL failed to find: OpenCL 2.2${ColourReset}" )
            set(CL_TARGET_OPENCL_VERSION 120 CACHE INTERNAL "")
        endif()
        add_definitions(-DCL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION})
        message("-- ${Magenta}ROCm OpenCL Found - Setting CL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION}${ColourReset}")
    endif()

    if( NOT OpenCL_FOUND )
        message( "-- ${Yellow}FindOpenCL failed to find: OpenCL${ColourReset}" )
    endif()
endif()
