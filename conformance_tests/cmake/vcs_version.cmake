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

message(STATUS "Get version information")
function(fill_version)
  find_package(Git QUIET)

  if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
      WORKING_DIRECTORY "."
      OUTPUT_VARIABLE VCSVERSION
      RESULT_VARIABLE GIT_RESULT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GIT_RESULT EQUAL 0)
      unset(VCSVERSION)
    endif()
  else()
    # We don't have git:
    unset(VCSVERSION)
  endif()

  if(DEFINED VCSVERSION)
    set(VERSION "${VCSVERSION}")
  else()
    set(VERSION "unknown")
  endif()

  message(STATUS "Version: ${VERSION}")

  set(RESULT "#define VCS_VERSION_STR \"${VERSION}\"")
  set(OUTPUT ${OUTPUT_DIR}/vcs_version.inc)

  if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" lines)
  endif()
  if("${lines}" STREQUAL "${RESULT}")
    #message(STATUS "${OUTPUT} contains same content")
  else()
    file(WRITE "${OUTPUT}" "${RESULT}")
  endif()
endfunction()
fill_version()
