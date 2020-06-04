# Copyright (c) 2012-2014 The Khronos Group Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and/or associated documentation files (the
# "Materials"), to deal in the Materials without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Materials, and to
# permit persons to whom the Materials are furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Materials.
#
# THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

set(KHRONOS_SRC_DIR ..)
set(OVX_TARGET_NAME c_model)
set(OVX_TARGET_CORES 4) #TODO: determine number of CPU cores for target platform

include(CheckCCompilerFlag)

macro(add_c_flag flag)
  string(TOUPPER "HAS${flag}" _HAS_VAR)
  check_c_compiler_flag(${flag} ${_HAS_VAR})
  if (${_HAS_VAR})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}")
  endif()
endmacro()

add_c_flag(-Wno-attributes)
add_c_flag(-std=gnu99) #sample implementation does use non-ansi extensions

if(CMAKE_COMPILER_IS_GNUCC)
  foreach(flags CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_DEBUG)
    string(REPLACE "-O3" "-O2" ${flags} "${${flags}}")
  endforeach()
endif()

# dirty hack to avoid early termination of sample implementation
add_definitions(-DDEBUG_BREAK=rand)

macro(openvx_target_setup target)
  if (WIN32)
    set_target_properties(${target} PROPERTIES PREFIX "")
  endif()
  target_include_directories(${target} PRIVATE ${KHRONOS_SRC_DIR}/include)
  target_compile_definitions(${target} PRIVATE OPENVX_BUILDING
                                               TARGET_NUM_CORES=${OVX_TARGET_CORES}
                                               OPENVX_USE_SMP
                            )
  if (WIN32)
    target_compile_definitions(${target} PRIVATE WINVER=0x501 _WIN32_WINNT=0x0600 "VX_API_ENTRY=__declspec(dllexport)")
  else()
    # _XOPEN_SOURCE=700 -> use POSIX 2008 (SUS v4)
    # _BSD_SOURCE=1     -> functionality derived from 4.3 BSD Unix is included as well as the ISO C, POSIX.1, and POSIX.2 material.
    # _GNU_SOURCE=1     -> If you define this macro, everything is included: ISO C89, ISO C99, POSIX.1, POSIX.2, BSD, SVID, X/Open, LFS, and GNU extensions. In the cases where POSIX.1 conflicts with BSD, the POSIX definitions take precedence.
    # _BSD_SOURCE is deprecated alias for _DEFAULT_SOURCE
    target_compile_definitions(${target} PRIVATE _XOPEN_SOURCE=700 _BSD_SOURCE=1 _GNU_SOURCE=1 _DEFAULT_SOURCE=1)
    if(APPLE)
        target_link_libraries(${target} PUBLIC pthread dl m)
    elseif(ANDROID)
        target_link_libraries(${target} PUBLIC dl m log)
    else()
        target_link_libraries(${target} PUBLIC pthread dl m rt)
    endif()
  endif()
  set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                                             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                                             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                                             PDB_OUTPUT_DIRECTORY     "${CMAKE_BINARY_DIR}/lib")
  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${target} PROPERTIES FOLDER "OpenVX - ${target}")
  endif()

  math(EXPR ARCH_BITS "${CMAKE_SIZEOF_VOID_P}*8")
  target_compile_definitions(${target} PRIVATE ARCH_${ARCH_BITS})

  if (MSVC)
    target_compile_options(${target} PRIVATE /wd4996 /wd4018 /wd4244 /wd4267)
  endif()
endmacro()

project(openvx-helper C)
file(GLOB OVX_SRC_HELPER "${KHRONOS_SRC_DIR}/helper/*.c" )
add_library(openvx-helper STATIC ${OVX_SRC_HELPER})
openvx_target_setup(openvx-helper)

project(vx_debug C)
file(GLOB OVX_SRC_DEBUG "${KHRONOS_SRC_DIR}/debug/*.c" )
add_library(vx_debug STATIC ${OVX_SRC_DEBUG})
target_include_directories(vx_debug PUBLIC ${KHRONOS_SRC_DIR}/debug)
openvx_target_setup(vx_debug)

project(openvx C)
file(GLOB OVX_SRC_FRAMEWORK "${KHRONOS_SRC_DIR}/sample/framework/*.c" )
add_library(openvx SHARED ${OVX_SRC_FRAMEWORK} "${KHRONOS_SRC_DIR}/sample/framework/openvx.def")
target_link_libraries(openvx PRIVATE openvx-helper vx_debug)
target_include_directories(openvx PUBLIC ${KHRONOS_SRC_DIR}/include PRIVATE ${KHRONOS_SRC_DIR}/sample/include)
openvx_target_setup(openvx)

project(vxu C)
file(GLOB OVX_SRC_VXU "${KHRONOS_SRC_DIR}/sample/vxu/*.c" )
add_library(vxu SHARED ${OVX_SRC_VXU} "${KHRONOS_SRC_DIR}/sample/vxu/vx_utility.def")
target_link_libraries(vxu PRIVATE openvx openvx-helper)
target_include_directories(vxu PUBLIC ${KHRONOS_SRC_DIR}/include)
openvx_target_setup(vxu)

project(openvx-c_model-lib C)
file(GLOB OVX_SRC_KERNELS_C_MODEL "${KHRONOS_SRC_DIR}/kernels/c_model/*.c")
add_library(openvx-c_model-lib STATIC ${OVX_SRC_KERNELS_C_MODEL})
target_link_libraries(openvx-c_model-lib PUBLIC openvx PRIVATE vx_debug)
target_include_directories(openvx-c_model-lib PRIVATE ${KHRONOS_SRC_DIR}/kernels/c_model)
openvx_target_setup(openvx-c_model-lib)

project(openvx-extras_k-lib C)
file(GLOB OVX_SRC_KERNELS_EXTRAS "${KHRONOS_SRC_DIR}/kernels/extras/*.c" )
add_library(openvx-extras_k-lib STATIC ${OVX_SRC_KERNELS_EXTRAS})
target_include_directories(openvx-extras_k-lib PUBLIC ${KHRONOS_SRC_DIR}/kernels/extras)
openvx_target_setup(openvx-extras_k-lib)

project(openvx-debug_k-lib C)
file(GLOB OVX_SRC_KERNELS_DEBUG "${KHRONOS_SRC_DIR}/kernels/debug/*.c" )
add_library(openvx-debug_k-lib STATIC ${OVX_SRC_KERNELS_DEBUG})
target_include_directories(openvx-debug_k-lib PUBLIC ${KHRONOS_SRC_DIR}/kernels/debug)
openvx_target_setup(openvx-debug_k-lib)

project(openvx-extras-lib C)
add_library(openvx-extras-lib STATIC "${KHRONOS_SRC_DIR}/libraries/extras/vx_extras_lib.c")
target_link_libraries(openvx-extras-lib PUBLIC openvx-helper)
openvx_target_setup(openvx-extras-lib)

project(openvx-extras C)
file(GLOB OVX_SRC_LIB_EXTRAS "${KHRONOS_SRC_DIR}/libraries/extras/*.c" )
list(REMOVE_ITEM OVX_SRC_LIB_EXTRAS "${CMAKE_CURRENT_SOURCE_DIR}/${KHRONOS_SRC_DIR}/libraries/extras/vx_extras_lib.c")
add_library(openvx-extras SHARED ${OVX_SRC_LIB_EXTRAS} "${KHRONOS_SRC_DIR}/libraries/extras/openvx-extras.def")
target_link_libraries(openvx-extras PRIVATE openvx-helper openvx-extras_k-lib PUBLIC openvx)
target_include_directories(openvx-extras PRIVATE ${KHRONOS_SRC_DIR}/libraries/extras)
openvx_target_setup(openvx-extras)

project(openvx-debug-lib C)
add_library(openvx-debug-lib STATIC "${KHRONOS_SRC_DIR}/libraries/debug/vx_debug_lib.c")
openvx_target_setup(openvx-debug-lib)

project(openvx-debug C)
file(GLOB OVX_SRC_LIB_DEBUG "${KHRONOS_SRC_DIR}/libraries/debug/*.c" )
list(REMOVE_ITEM OVX_SRC_LIB_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/${KHRONOS_SRC_DIR}/libraries/debug/vx_debug_lib.c")
add_library(openvx-debug SHARED ${OVX_SRC_LIB_DEBUG} "${KHRONOS_SRC_DIR}/libraries/debug/openvx-debug.def")
target_link_libraries(openvx-debug PRIVATE openvx-helper openvx-debug_k-lib PUBLIC openvx)
target_include_directories(openvx-debug PRIVATE ${KHRONOS_SRC_DIR}/libraries/debug)
openvx_target_setup(openvx-debug)

project(openvx-c_model C)
file(GLOB OVX_SRC_TARGETS_C_MODEL "${KHRONOS_SRC_DIR}/sample/targets/c_model/*.c")
add_library(openvx-c_model SHARED ${OVX_SRC_TARGETS_C_MODEL} "${KHRONOS_SRC_DIR}/sample/targets/c_model/openvx-target.def")
target_link_libraries(openvx-c_model PRIVATE openvx-debug-lib openvx-extras-lib openvx-helper openvx-c_model-lib PUBLIC openvx vxu)
target_include_directories(openvx-c_model PRIVATE ${KHRONOS_SRC_DIR}/sample/include ${KHRONOS_SRC_DIR}/kernels/c_model ${KHRONOS_SRC_DIR}/sample/targets/c_model ${KHRONOS_SRC_DIR}/debug)
openvx_target_setup(openvx-c_model)

if (MSVC)
  # compile that file as C++ because MSVC in not C99 compliant
  set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/${KHRONOS_SRC_DIR}/sample/targets/c_model/vx_optpyrlk.c"
                              PROPERTIES COMPILE_FLAGS /TP)
endif()

if (0)
  project(vx_conformance C)
  file(GLOB OVX_CONFORMANCE "${KHRONOS_SRC_DIR}/conformance/*.c")
  add_executable(vx_conformance ${OVX_CONFORMANCE})
  target_include_directories(vx_conformance PRIVATE "${KHRONOS_SRC_DIR}/conformance")
  target_link_libraries(vx_conformance PRIVATE openvx-debug-lib openvx-helper openvx vxu)
  openvx_target_setup(vx_conformance)
endif()
