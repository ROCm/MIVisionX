# Install script for directory: /home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/opt/rocm/mivisionx")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./rali/python" TYPE FILE FILES
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/./python/rali.py"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/python/rali_lib.py"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/./python/rali_common.py"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/./python/rali_image.py"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/./python/rali_parameter.py"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/./python/rali_torch.py"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/./python/rali_image_iterator.py"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./include" TYPE FILE FILES
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/include/rali_api.h"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/include/rali_api_info.h"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/include/rali_api_augmentation.h"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/include/rali_api_data_loaders.h"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/include/rali_api_types.h"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/include/rali_api_data_transfer.h"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/include/rali_api_parameters.h"
    "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/rali/include/rali_api_meta_data.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/librali.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/librali.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/librali.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/buid/lib/librali.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/librali.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/librali.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/librali.so"
         OLD_RPATH "/home/lokesh/AMD/GPUOpen/forked/MIVisionX/buid/lib:/usr/local/lib:/opt/rocm/rpp/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/librali.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

