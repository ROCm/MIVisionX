cmake_minimum_required(VERSION 3.10)
set(MIN_CXX_VER_REQUIRED 7.4)
#set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
project(rali_pybind)
link_directories(${ROCM_PATH}/mivisionx/lib/
                ${ROCM_PATH}/rpp/lib)
include_directories ( ../include
                      ../../amd_openvx/openvx/include/
                      ${ROCM_PATH}/mivisionx/include/)
add_subdirectory(pybind11)
pybind11_add_module(rali_pybind rali_pybind.cpp)
target_link_libraries(rali_pybind PRIVATE rali vx_rpp amd_rpp )
