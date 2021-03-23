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
# - Try to find ffmpeg libraries (libavcodec, libavformat and libavutil)
# Once done this will define
#
# FFMPEG_FOUND - system has ffmpeg or libav
# FFMPEG_INCLUDE_DIR - the ffmpeg include directory
# FFMPEG_LIBRARIES - Link these to use ffmpeg
# FFMPEG_LIBAVCODEC
# FFMPEG_LIBAVFORMAT
# FFMPEG_LIBAVUTIL
#

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig")
include(FindPackageHandleStandardArgs)

    find_package_handle_standard_args(FFmpeg
        FOUND_VAR FFMPEG_FOUND
        REQUIRED_VARS
            FFMPEG_LIBRARY
            FFMPEG_INCLUDE_DIR
        VERSION_VAR FFMPEG_VERSION
    )

if(FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
  # in cache already
  set(FFMPEG_FOUND TRUE)
else()
  # use pkg-config to get the directories and then use these values
  # in the FIND_PATH() and FIND_LIBRARY() calls
  find_package(PkgConfig)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(_FFMPEG_AVCODEC libavcodec)
    pkg_check_modules(_FFMPEG_AVFORMAT libavformat)
    pkg_check_modules(_FFMPEG_AVUTIL libavutil)
  endif()

  find_path(FFMPEG_AVCODEC_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS ${_FFMPEG_AVCODEC_INCLUDE_DIRS}
      /usr/local/include
      /usr/include
      /opt/local/include
      /sw/include
    PATH_SUFFIXES ffmpeg libav)

  find_library(FFMPEG_LIBAVCODEC
    NAMES avcodec
    PATHS ${_FFMPEG_AVCODEC_LIBRARY_DIRS}
      /usr/local/lib
      /usr/lib
      /opt/local/lib
      /sw/lib)

  find_library(FFMPEG_LIBAVFORMAT
    NAMES avformat
    PATHS ${_FFMPEG_AVFORMAT_LIBRARY_DIRS}
      /usr/local/lib
      /usr/lib
      /opt/local/lib
      /sw/lib)

  find_library(FFMPEG_LIBAVUTIL
    NAMES avutil
    PATHS ${_FFMPEG_AVUTIL_LIBRARY_DIRS}
      /usr/local/lib
      /usr/lib
      /opt/local/lib
      /sw/lib)

  if(FFMPEG_LIBAVCODEC AND FFMPEG_LIBAVFORMAT)
    set(FFMPEG_FOUND TRUE)
  endif()
  
  if(_FFMPEG_AVCODEC_VERSION VERSION_LESS 58.18.100 OR _FFMPEG_AVFORMAT_VERSION VERSION_LESS 58.12.100 OR _FFMPEG_AVUTIL_VERSION VERSION_LESS 56.14.100)
    set(FFMPEG_FOUND FALSE)
    message("-- AVCODEC  required min version - 58.18.100")
    message("-- AVFORMAT required min version - 58.12.100")
    message("-- AVUTIL   required min version - 56.14.100")
    message("-- FFMPEG   required min version - 4.0.4")
    message("-- FFMPEG Marked Not Found - MIVisionX Modules requiring FFMPEG turned off")
  endif()
  
  if(FFMPEG_FOUND)
    set(FFMPEG_INCLUDE_DIR ${FFMPEG_AVCODEC_INCLUDE_DIR})
    set(FFMPEG_LIBRARIES
      ${FFMPEG_LIBAVCODEC}
      ${FFMPEG_LIBAVFORMAT}
      ${FFMPEG_LIBAVUTIL})
  endif()

  if(FFMPEG_FOUND)
    if(NOT FFMPEG_FIND_QUIETLY)
      message(STATUS
      "Found FFMPEG or Libav: ${FFMPEG_LIBRARIES}, ${FFMPEG_INCLUDE_DIR}")
    endif()
  else()
    if(FFMPEG_FIND_REQUIRED)
      message(FATAL_ERROR
      "Could not find libavcodec or libavformat or libavutil")
    endif()
  endif()
endif()
