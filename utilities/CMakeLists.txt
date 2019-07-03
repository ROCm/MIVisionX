# Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#  
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#  
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required (VERSION 2.8)
project (utilities)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../amd_openvx/cmake)

find_package(OpenCL QUIET)
find_package(FFmpeg QUIET)

if(OpenCL_FOUND)
    add_subdirectory (runvx)
    add_subdirectory (runcl)
    add_subdirectory (loom_shell)
    if(FFMPEG_FOUND)
        add_subdirectory (mv_deploy)
        message("-- ${Green}MIVisionX Utilities -- runvx, runcl, loom_shell, & mv_deploy modules added${ColourReset}")
        install (FILES ./mv_deploy/mvdeploy_api.h DESTINATION include)
        install (FILES ./mv_deploy/mv_extras_postproc.h DESTINATION model_compiler)
        install (FILES ./mv_deploy/mv_extras_postproc.cpp DESTINATION model_compiler)
    else()
        message("-- ${Green}MIVisionX Utilities -- runvx, runcl, & loom_shell modules added${ColourReset}")
        message("-- ${Red}WARNING:FFMPEG Not Found -- mv_deploy module excluded${ColourReset}")
    endif(FFMPEG_FOUND)
else()
    message("-- ${Red}WARNING:OpenCL/FFMPEG Not Found -- runvx, runcl, loom_shell, & mv_deploy modules excluded${ColourReset}")
endif(OpenCL_FOUND)
