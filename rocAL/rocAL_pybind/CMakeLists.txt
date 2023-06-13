# Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.4)
project(rocAL_pybind)

set(TARGET_NAME rocAL_pybind)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../amd_openvx/cmake)
set(ROCM_PATH /opt/rocm CACHE PATH "Deafult ROCm Installation Path")

set(CMAKE_CXX_STANDARD 14)
set(BUILD_ROCAL_PYBIND true)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    set(BUILD_ROCAL_PYBIND false)
    set(ROCAL_PYTHON OFF)
    message("-- ${Red}WARNING: rocAL pybind is not supported with OpenCL backend -- rocal_pybind module excluded${ColourReset}")
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    find_package(AMDRPP QUIET)
    find_package(TurboJpeg QUIET)
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message("-- ${White}rocAL PyBind -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
    else()
        message("-- ${Yellow}rocAL pybind requires HIP, Not Found${ColourReset}")
    endif()
endif()

if(ROCAL_PYTHON)
    if(NOT HIP_FOUND)
        set(BUILD_ROCAL_PYBIND false)
        message("-- ${Yellow}rocAL pybind requires HIP, Not Found${ColourReset}")
    endif()
    if(NOT AMDRPP_FOUND)
        set(BUILD_ROCAL_PYBIND false)
        message("-- ${Yellow}NOTE: rocAL pybind requires AMDRPP, Not Found${ColourReset}")
    else()
        #find the RPP backend type
        set(RPP_BACKEND_HIP_FOUND 0)
        if(EXISTS ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h)
            file(READ ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h RPP_BACKEND_FILE)
            string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
            set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
        else()
            message("-- ${Red}WARNING: ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h file Not Found. please run the setup script to install latest RPP package ${ColourReset}")
        endif()
        if(HIP_FOUND)
            if (NOT RPP_BACKEND_HIP_FOUND)
                set(BUILD_ROCAL_PYBIND false)
                message("-- ${Red}WARNING: RPP with the HIP backend Not Found (HINT: run setup script with the HIP backend first!) -- rocAL pybind module excluded${ColourReset}")
            endif()
        endif()
    endif()

    if(NOT TurboJpeg_FOUND)
        set(BUILD_ROCAL_PYBIND false)
        message("-- ${Yellow}NOTE: rocAL pybind requires TurboJpeg, Not Found${ColourReset}")
    endif()

    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 QUIET COMPONENTS Interpreter Development)
    include_directories(${Python3_INCLUDE_DIRS})
    message("-- ${Green}NOTE: rocAL pybind requires Python3, Found --  ${Python3_EXECUTABLE} ${ColourReset}")
    if(NOT Python3_FOUND)
        set(BUILD_ROCAL_PYBIND false)
        message("-- ${Yellow}NOTE: rocAL pybind requires Python3, Not Found${ColourReset}")
    endif()

    if(${BUILD_ROCAL_PYBIND})

        # avoid setting the default installation path to /usr/local
        if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
            message("-- ${Green}NOTE: rocAL default installation path -- ${Python3_STDARCH} ${ColourReset}")
            set(CMAKE_INSTALL_PREFIX_PYTHON ${Python3_STDARCH} CACHE PATH "rocAL default installation path" FORCE)
        endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        find_package(pybind11 REQUIRED CONFIG)
        if(NOT pybind11_FOUND)
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Yellow}NOTE: rocAL requires pybind11, Not Found${ColourReset}")
        endif()
        include_directories(${pybind11_INCLUDE_DIRS})
        # AMD RPP
        include_directories(${AMDRPP_INCLUDE_DIRS})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${AMDRPP_LIBRARIES})
        # MIVisionX
        include_directories(${MIVisionX_INCLUDE_DIRS})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OPENVX_LIBRARIES} ${VXRPP_LIBRARIES})
        # TurboJPEG
        include_directories(${TurboJpeg_INCLUDE_DIRS})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${TurboJpeg_LIBRARIES})
        # rocAL
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocal)
        include_directories(../rocAL/include/api/)

        file(GLOB_RECURSE pyfiles amd/*.py)
        file(GLOB_RECURSE sources *.cpp)

        pybind11_add_module(rocal_pybind MODULE ${sources})
        target_link_libraries(rocal_pybind PRIVATE ${LINK_LIBRARY_LIST})
        add_dependencies(rocal_pybind rocal)
        message("-- ${White}rocal_pybind -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")
        message("-- ${White}rocal_pybind -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")

        set_target_properties(rocal_pybind PROPERTIES
            PREFIX "${PYTHON_MODULE_PREFIX}"
            SUFFIX "${PYTHON_MODULE_EXTENSION}"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${TARGET_NAME}/${CMAKE_INSTALL_LIBDIR}")
        set(TARGET_PYMODULE_NAME "${PYTHON_MODULE_PREFIX}rocal_pybind${PYTHON_MODULE_EXTENSION}")

        CONFIGURE_FILE(amd/__init__.py
                ${CMAKE_BINARY_DIR}/${TARGET_NAME}/__init__.py @ONLY
                )
        CONFIGURE_FILE(setup.py
                ${CMAKE_BINARY_DIR}/setup.py @ONLY
                )
        CONFIGURE_FILE(getrocALwheelname.py
                ${CMAKE_BINARY_DIR}/getrocALwheelname.py @ONLY
                )
        CONFIGURE_FILE(rocal_pybind.cpp
                ${CMAKE_BINARY_DIR}/rocal_pybind.cpp @ONLY
                )

        foreach (filename ${pyfiles})
            get_filename_component(target "${filename}" REALPATH)
            #to maintain folder structure
            file(RELATIVE_PATH ITEM_PATH_REL ${CMAKE_CURRENT_SOURCE_DIR} ${filename})
            message(STATUS "Copying ${filename} to ${TARGET_NAME}/${ITEM_PATH_REL}")
            configure_file("${filename}" "${CMAKE_BINARY_DIR}/${TARGET_NAME}/${ITEM_PATH_REL}" COPYONLY)
        endforeach (filename)

        execute_process(COMMAND "${Python3_EXECUTABLE}" getrocALwheelname.py OUTPUT_VARIABLE PY_WHEELFILE
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
        message(STATUS "Detected rocAL Python wheel filename ${PY_WHEELFILE}")
        
        add_custom_target(PyPackageBuild
            COMMAND "${Python3_EXECUTABLE}" -m pip wheel .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            COMMENT "Building python wheel package"
            )

        add_dependencies(PyPackageBuild rocal_pybind)

        set_target_properties(PyPackageBuild PROPERTIES Package_Name ${PY_WHEELFILE})
        add_custom_target(PyPackageInstall
                COMMAND "${Python3_EXECUTABLE}" -m pip install --upgrade --force-reinstall --no-deps "${PY_WHEELFILE}"
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                COMMENT "Installing wheel file"
                )
        add_dependencies(PyPackageInstall PyPackageBuild)
        
        #install rocAL pybind libs
        install(FILES ${CMAKE_BINARY_DIR}/${TARGET_NAME}/${CMAKE_INSTALL_LIBDIR}/${TARGET_PYMODULE_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX_PYTHON})
        message("-- ${Green}Radeon Augmentation Library Python Binding - rocal_pybind module added ${ColourReset}")
    else()
        message("-- ${Red}WARNING: rocAL PyBind Module excluded - Dependency Failure${ColourReset}")
    endif()        
endif()
