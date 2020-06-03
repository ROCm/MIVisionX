/*
 * Copyright (c) 2012-2015 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

#ifndef _OPENVX_H_
#define _OPENVX_H_

/*!
 * \file
 * \brief The top level OpenVX Header.
 */

/*! \brief Defines the maximum number of characters in a implementation string.
 * \ingroup group_context
 */
#define VX_MAX_IMPLEMENTATION_NAME (64)

/*! \brief Defines the maximum string length of a kernel name to be added to OpenVX.
 * \ingroup group_kernel
 */
#define VX_MAX_KERNEL_NAME (256)

/*! \brief Defines the maximum length of a message buffer to copy from the log.
 * \ingroup group_basic_features
 */
#define VX_MAX_LOG_MESSAGE_LEN (1024)

#include <VX/vx_vendors.h>
#include <VX/vx_types.h>
#include <VX/vx_kernels.h>
#include <VX/vx_api.h>
#include <VX/vx_nodes.h>

/*! Defines the major version number macro.
 * \ingroup group_basic_features
 */
#define VX_VERSION_MAJOR(x) ((x & 0xFF) << 8)

/*! Defines the minor version number macro.
 * \ingroup group_basic_features
 */
#define VX_VERSION_MINOR(x) ((x & 0xFF) << 0)

/*! \brief Defines the predefined version number for 1.0.
 * \ingroup group_basic_features
 */
#define VX_VERSION_1_0      (VX_VERSION_MAJOR(1) | VX_VERSION_MINOR(0))

/*! Defines the OpenVX Version Number.
 * \ingroup group_basic_features
 */
#define VX_VERSION          VX_VERSION_1_0

#endif
