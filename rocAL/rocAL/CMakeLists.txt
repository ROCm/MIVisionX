# MIT License
#
# Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.5)
set(VERSION "1.1.0")
project(rocal VERSION ${VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(BUILD_ROCAL true)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)

find_package(AMDRPP QUIET)
find_package(TurboJpeg QUIET)
find_package(Protobuf QUIET)
find_package(FFmpeg QUIET)
find_package(OpenCV QUIET)
find_package(OpenMP QUIET)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads QUIET)
find_package(LMDB QUIET)
find_package(RapidJSON QUIET)

# Check for filesystem support
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
set(CMAKE_REQUIRED_INCLUDES "")
check_cxx_source_compiles("#include <filesystem>\nnamespace fs = std::filesystem;\nint main() { return 0; }" HAS_STD_FILESYSTEM)
check_cxx_source_compiles("#include <experimental/filesystem>\nnamespace fs = std::experimental::filesystem;\nint main() { return 0; }" HAS_EXP_FILESYSTEM)

if (HAS_STD_FILESYSTEM)
    set(FS_FOUND true)
elseif (HAS_EXP_FILESYSTEM)
    set(FS_FOUND true)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} stdc++fs)
else ()
    set(FS_FOUND false)
endif ()

if( GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    set(DEFAULT_AMDGPU_TARGETS "gfx803;gfx900;gfx906;gfx908;gfx90a;gfx940;gfx1030;gfx1031;gfx1032;gfx1100")
    if (BUILD_WITH_AMD_ADVANCE)
        set(DEFAULT_AMDGPU_TARGETS ${DEFAULT_AMDGPU_TARGETS} "gfx941;gfx942")
    endif()
    set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
    find_package(HIP QUIET)
    if(HIP_FOUND)
        SET(BUILD_ROCAL true)
        message("-- ${White}rocAL -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
    else()
        SET(BUILD_ROCAL false)
        message("-- ${Yellow}NOTE: rocAL library requires HIP for BACKEND=HIP, Not Found ${ColourReset}")
    endif()
endif()

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
  find_package(OpenCL QUIET)
  if(NOT OpenCL_FOUND)
      set(BUILD_ROCAL false)
      message("-- ${Yellow}NOTE: rocAL library requires OpenCL for BACKEND=OPENCL, Not Found ${ColourReset}")
  endif()
endif()

if(NOT AMDRPP_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires AMD's rpp library, Not Found ${ColourReset}")
else()
    #find the RPP backend type
    set(RPP_BACKEND_OPENCL_FOUND 0)
    set(RPP_BACKEND_HIP_FOUND 0)
    if(EXISTS ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h)
        file(READ ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h RPP_BACKEND_FILE)
        string(REGEX MATCH "RPP_BACKEND_OPENCL ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
        string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    else()
        message("-- ${Red}WARNING: ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h file Not Found. please run the setup script to install latest RPP package ${ColourReset}")
    endif()

    if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND)
        if (NOT RPP_BACKEND_OPENCL_FOUND)
            set(BUILD_ROCAL false)
            message("-- ${Red}WARNING: RPP with OpenCL backend Not Found (HINT: run setup script with the OCL backend first!) -- rocAL module excluded.${ColourReset}")
        endif()
    elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        if (NOT RPP_BACKEND_HIP_FOUND)
            set(BUILD_ROCAL false)
            message("-- ${Red}WARNING: RPP with the HIP backend Not Found (HINT: run setup script with the HIP backend first!) -- rocAL_rpp module excluded.${ColourReset}")
        endif()
    else()
        #set(BUILD_ROCAL false)
        message("-- ${Red}WARNING: GPU support with OpenCL or HIP Not Found -- rocAL CPU only build${ColourReset}")
    endif()
endif()
if(NOT TurboJpeg_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires TurboJpeg, Not Found ${ColourReset}")
endif()
if(NOT Protobuf_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires Protobuf, Not Found ${ColourReset}")
endif()
if(NOT OpenMP_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires OpenMP, Not Found ${ColourReset}")
endif()
if(NOT FS_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires FileSystem, Not Found ${ColourReset}")
endif()
if(NOT Threads_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires Threads, Not Found ${ColourReset}")
endif()
if(NOT LMDB_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires LMDB, Not Found ${ColourReset}")
endif()
if(NOT RapidJSON_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires RapidJSON, Not Found ${ColourReset}")
endif()

if(${BUILD_ROCAL})
    # AMD OpenVX & VX_RPP
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} openvx vx_rpp)
    # AMD RPP
    include_directories(${AMDRPP_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${AMDRPP_LIBRARIES})
    # OpenMP
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} OpenMP::OpenMP_CXX)
    # Threads
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} Threads::Threads)
    # TurboJPEG
    include_directories(${TurboJpeg_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${TurboJpeg_LIBRARIES})
    # Protobuf
    include_directories(${PROTOBUF_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${PROTOBUF_LIBRARIES})
    # LMDB
    include_directories(${LMDB_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LMDB_LIBRARIES})
    # RapidJSON
    include_directories(${RapidJSON_INCLUDE_DIRS})

    # rocAL uses C++ 17 features
    set(CMAKE_CXX_STANDARD 17)
    message("-- ${Green}ROCm Augmentation Library -- rocAL module added${ColourReset}")
    if(FFMPEG_FOUND)
        find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
        find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
        find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h)
        find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
    endif()
    include_directories(
                ${CMAKE_CURRENT_BINARY_DIR}
                ../../amd_openvx/openvx/include/
                ../../amd_openvx_extensions/amd_rpp/include/
                ../../amd_openvx_extensions/amd_media/include/
                include/api/
                include/augmentations/color_augmentations/
                include/augmentations/effects_augmentations/
                include/augmentations/geometry_augmentations/
                include/augmentations/
                include/decoders/image/
                include/decoders/video/
                include/device/
                include/loaders/
                include/loaders/image/
                include/loaders/video/
                include/meta_data/
                include/parameters/
                include/pipeline/
                include/readers/image/
                include/readers/video/
    )

    if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        link_directories(${HIP_PATH}/lib)
        include_directories(${ROCM_PATH}/include rocAL_hip)
        add_subdirectory(rocAL_hip)
    endif()

    protobuf_generate_cpp(TF_PROTO_SRCS TF_PROTO_HEADERS proto/example.proto proto/feature.proto)
    protobuf_generate_cpp(CAFFE2_PROTO_SRCS CAFFE2_PROTO_HEADERS proto/caffe2_protos.proto)
    protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HEADERS proto/caffe_protos.proto)
    link_directories(${AMDRPP_LIBRARIES_DIRS} ${TurboJpeg_LIBRARIES_DIR} ${PROTOBUF_LIBRARY_DIRS} /usr/local/lib/)

    file(GLOB_RECURSE SOURCES "source/*.cpp")
    add_library(${PROJECT_NAME} SHARED ${SOURCES} ${TF_PROTO_SRCS} ${TF_PROTO_HEADERS} ${CAFFE_PROTO_HEADERS} ${CAFFE_PROTO_SRCS} ${CAFFE2_PROTO_SRCS} ${CAFFE2_PROTO_HEADERS})

    if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        add_dependencies(${PROJECT_NAME} rocAL_hip)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} $<TARGET_OBJECTS:rocAL_hip>)
        target_compile_definitions(${PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=1)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=0)
        message("-- ${White}rocAL built with HIP Backend${ColourReset}")
    elseif("${BACKEND}" STREQUAL "OPENCL" AND OPENCL_FOUND)
        include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=0)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=1)
        message("-- ${White}rocAL built with OpenCL Backend${ColourReset}")
    else()
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=0)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=0)
        message("-- ${White}rocAL built with CPU only Backend${ColourReset}")
    endif()

    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBG_TIMING=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBGINFO=0)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBGLOG=0 WRNLOG=0)
    target_compile_definitions(${PROJECT_NAME} PUBLIC TFRECORD_ENABLE=0)
    message("-- ${White}rocAL built with ENABLE_SIMD")
    #OpenCV
    if (OpenCV_FOUND)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=1)
        include_directories(${OpenCV_INCLUDE_DIRS})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenCV_LIBRARIES})
        if(${OpenCV_VERSION_MAJOR} EQUAL 4)
            target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=1)
        else()
	        target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=0)
        endif()
    else()
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=0)
    endif()
    #FileSystem
    if (HAS_STD_FILESYSTEM)
        target_compile_definitions(${PROJECT_NAME} PUBLIC USE_STD_FILESYSTEM)
    elseif (HAS_EXP_FILESYSTEM)
        target_compile_definitions(${PROJECT_NAME} PUBLIC USE_EXP_FILESYSTEM)
    endif()
    # -Wall -- Enable most warning messages
    # -mavx2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and AVX2 built-in functions and code generation
    # -mfma -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and FMA built-in functions and code generation
    # -mf16c -- Support F16C built-in functions and code generation
    # -Wno-deprecated-declarations -- Do not warn about uses of functions, variables, and types marked as deprecated by using the deprecated attribute
    # -std=gnu++17 -- Conform to the ISO 2017 C++ standard with GNU extensions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mavx2 -mfma -mf16c -Wno-deprecated-declarations -std=gnu++17")
    message("-- ${White}rocAL -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")

    if(NOT FFMPEG_FOUND)
        message("-- ${Yellow}NOTE: rocAL library is going to be built without video decode functionality ${ColourReset}")
    else()
        include_directories(${AVUTIL_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR} ${SWSCALE_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${FFMPEG_LIBRARIES})
        message("-- ${White}rocAL library is going to be built with video decode functionality ${ColourReset}")
        target_compile_definitions(${PROJECT_NAME} PUBLIC -DROCAL_VIDEO)
    endif()
    target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})
    message("-- ${White}rocAL -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")
    set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

    # install MIVisionX libs -- {ROCM_PATH)/lib
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
    # install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
    install(FILES
                include/api/rocal_api.h
                include/api/rocal_api_info.h
                include/api/rocal_api_augmentation.h
                include/api/rocal_api_data_loaders.h
                include/api/rocal_api_types.h
                include/api/rocal_api_data_transfer.h
                include/api/rocal_api_parameters.h
                include/api/rocal_api_meta_data.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx/rocal COMPONENT dev
    )
else()
    message("-- ${Red}WARNING: rocAL module excluded ${ColourReset}")
endif()
