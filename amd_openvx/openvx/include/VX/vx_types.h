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

#ifndef _OPENVX_TYPES_H_
#define _OPENVX_TYPES_H_

/*!
 * \file vx_types.h
 * \brief The type definitions required by OpenVX Library.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/*!
 * \internal
 * \def VX_API_ENTRY
 * \brief This is a tag used to identify exported, public API functions as
 * distinct from internal functions, helpers, and other non-public interfaces.
 * It can optionally be defined in the make system according the the compiler and intent.
 * \ingroup group_basic_features
 */
#ifndef VX_API_ENTRY
#define VX_API_ENTRY
#endif
#ifndef VX_API_CALL
#if defined(_WIN32)
#define VX_API_CALL __stdcall
#else
#define VX_API_CALL
#endif
#endif
#ifndef VX_CALLBACK
#if defined(_WIN32)
#define VX_CALLBACK __stdcall
#else
#define VX_CALLBACK
#endif
#endif

/*! \brief An 8 bit ASCII character.
 * \ingroup group_basic_features
 */
typedef char     vx_char;

/*! \brief An 8-bit unsigned value.
 * \ingroup group_basic_features
 */
typedef uint8_t  vx_uint8;

/*! \brief A 16-bit unsigned value.
 * \ingroup group_basic_features
 */
typedef uint16_t vx_uint16;

/*! \brief A 32-bit unsigned value.
 * \ingroup group_basic_features
 */
typedef uint32_t vx_uint32;

/*! \brief A 64-bit unsigned value.
 * \ingroup group_basic_features
 */
typedef uint64_t vx_uint64;

/*! \brief An 8-bit signed value.
 * \ingroup group_basic_features
 */
typedef int8_t   vx_int8;

/*! \brief A 16-bit signed value.
 * \ingroup group_basic_features
 */
typedef int16_t  vx_int16;

/*! \brief A 32-bit signed value.
 * \ingroup group_basic_features
 */
typedef int32_t  vx_int32;

/*! \brief A 64-bit signed value.
 * \ingroup group_basic_features
 */
typedef int64_t  vx_int64;

#if defined(EXPERIMENTAL_PLATFORM_SUPPORTS_16_FLOAT)

/*! \brief A 16-bit float value.
 * \ingroup group_basic_features
 */
typedef hfloat   vx_float16;
#endif

/*! \brief A 32-bit float value.
 * \ingroup group_basic_features
 */
typedef float    vx_float32;

/*! \brief A 64-bit float value (aka double).
 * \ingroup group_basic_features
 */
typedef double   vx_float64;

/*! \brief A generic opaque reference to any object within OpenVX.
 * \details A user of OpenVX should not assume that this can be cast directly to anything;
 * however, any object in OpenVX can be cast back to this for the purposes of
 * querying attributes of the object or for passing the object as a parameter to
 * functions that take a <tt>\ref vx_reference</tt> type.
 * If the API does not take that specific type but may take others, an
 * error may be returned from the API.
 * \ingroup group_reference
 */
typedef struct _vx_reference *vx_reference;

/*! \brief Sets the standard enumeration type size to be a fixed quantity.
 * \details All enumerable fields must use this type as the container to
 * enforce enumeration ranges and sizeof() operations.
 * \ingroup group_basic_features
 */
typedef int32_t vx_enum;

/*! \brief A wrapper of <tt>size_t</tt> to keep the naming convention uniform.
 * \ingroup group_basic_features
 */
typedef size_t vx_size;

/*! \brief Used to hold a VX_DF_IMAGE code to describe the pixel format and color space.
 * \ingroup group_basic_features
 */
typedef uint32_t vx_df_image;

/*! \brief An opaque reference to a scalar.
 * \details A scalar can be up to 64 bits wide.
 * \see vxCreateScalar
 * \ingroup group_scalar
 * \extends vx_reference
 */
typedef struct _vx_scalar *vx_scalar;

/*! \brief An opaque reference to an image.
 * \see vxCreateImage
 * \ingroup group_image
 * \extends vx_reference
 */
typedef struct _vx_image *vx_image;

/*! \brief An opaque reference to the descriptor of a kernel.
 * \see vxGetKernelByName
 * \see vxGetKernelByEnum
 * \ingroup group_kernel
 * \extends vx_reference
 */
typedef struct _vx_kernel *vx_kernel;

/*! \brief An opaque reference to a single parameter.
 * \see vxGetParameterByIndex
 * \ingroup group_parameter
 * \extends vx_reference
 */
typedef struct _vx_parameter *vx_parameter;

/*! \brief An opaque reference to a kernel node.
 * \see vxCreateGenericNode
 * \ingroup group_node
 * \extends vx_reference
 */
typedef struct _vx_node *vx_node;

/*! \brief An opaque reference to a graph
 * \see vxCreateGraph
 * \ingroup group_graph
 * \extends vx_reference
 */
typedef struct _vx_graph *vx_graph;

/*! \brief An opaque reference to the implementation context.
 * \see vxCreateContext
 * \ingroup group_context
 * \extends vx_reference
 */
typedef struct _vx_context *vx_context;

/*! \brief The delay object. This is like a ring buffer of objects that is
 * maintained by the OpenVX implementation.
 * \see vxCreateDelay
 * \extends vx_reference
 * \ingroup group_delay
 */
typedef struct _vx_delay *vx_delay;

/*! \brief The Look-Up Table (LUT) Object.
 * \extends vx_reference
 * \ingroup group_lut
 */
typedef struct _vx_lut *vx_lut;

/*! \brief The Distribution object. This has a user-defined number of bins over
 * a user-defined range (within a uint32_t range).
 * \extends vx_reference
 * \ingroup group_distribution
 */
typedef struct _vx_distribution *vx_distribution;

/*! \brief The Matrix Object. An MxN matrix of some unit type.
 * \extends vx_reference
 * \ingroup group_matrix
 */
typedef struct _vx_matrix *vx_matrix;

/*! \brief The Image Pyramid object. A set of scaled images.
 * \extends vx_reference
 * \ingroup group_pyramid
 */
typedef struct _vx_pyramid *vx_pyramid;

/*! \brief The Threshold Object. A thresholding object contains the types and
 * limit values of the thresholding required.
 * \extends vx_reference
 * \ingroup group_threshold
 */
typedef struct _vx_threshold *vx_threshold;

/*! \brief The Convolution Object. A user-defined convolution kernel of MxM elements.
 * \extends vx_reference
 * \ingroup group_convolution
 */
typedef struct _vx_convolution *vx_convolution;

/*! \brief The remap table Object. A remap table contains per-pixel mapping of
 * output pixels to input pixels.
 * \ingroup group_remap
 */
typedef struct _vx_remap *vx_remap;

/*! \brief The Array Object. Array is a strongly-typed container for other data structures.
 * \ingroup group_array
 */
typedef struct _vx_array *vx_array;

/*! \brief A Boolean value.
 * This allows 0 to be FALSE, as it is in C, and any non-zero to be TRUE.
 * \code
 * vx_bool ret = vx_true_e;
 * if (ret) printf("true!\n");
 * ret = vx_false_e;
 * if (!ret) printf("false!\n");
 * \endcode
 * This would print both strings.
 * \ingroup group_basic_features
 */
typedef enum _vx_bool_e {
    /*! \brief The "false" value. */
    vx_false_e = 0,
    /*! \brief The "true" value. */
    vx_true_e,
} vx_bool;

/*!
 * \brief This object is used by output validation functions to specify the meta data 
 * of the expected output data object. If the output object is an image, 
 * the vx_meta_format object can additionally store the valid region delta rectangle.
 * \note when the actual output object of the user node is virtual, the information 
 * given through the vx_meta_format object allows the OpenVX framework to automatically 
 * create the data object when meta data were not specified by the application at object 
 * creation time. 
 * \ingroup group_user_kernels
 */
typedef struct _vx_meta_format* vx_meta_format;

/*! \brief The type enumeration lists all the known types in OpenVX.
 * \ingroup group_basic_features
 */
enum vx_type_e {
    VX_TYPE_INVALID         = 0x000,/*!< \brief An invalid type value. When passed an error must be returned. */
    VX_TYPE_CHAR            = 0x001,/*!< \brief A <tt>\ref vx_char</tt>. */
    VX_TYPE_INT8            = 0x002,/*!< \brief A <tt>\ref vx_int8</tt>. */
    VX_TYPE_UINT8           = 0x003,/*!< \brief A <tt>\ref vx_uint8</tt>. */
    VX_TYPE_INT16           = 0x004,/*!< \brief A <tt>\ref vx_int16</tt>. */
    VX_TYPE_UINT16          = 0x005,/*!< \brief A <tt>\ref vx_uint16</tt>. */
    VX_TYPE_INT32           = 0x006,/*!< \brief A <tt>\ref vx_int32</tt>. */
    VX_TYPE_UINT32          = 0x007,/*!< \brief A <tt>\ref vx_uint32</tt>. */
    VX_TYPE_INT64           = 0x008,/*!< \brief A <tt>\ref vx_int64</tt>. */
    VX_TYPE_UINT64          = 0x009,/*!< \brief A <tt>\ref vx_uint64</tt>. */
    VX_TYPE_FLOAT32         = 0x00A,/*!< \brief A <tt>\ref vx_float32</tt>. */
    VX_TYPE_FLOAT64         = 0x00B,/*!< \brief A <tt>\ref vx_float64</tt>. */
    VX_TYPE_ENUM            = 0x00C,/*!< \brief A <tt>\ref vx_enum</tt>. Equivalent in size to a <tt>\ref vx_int32</tt>. */
    VX_TYPE_SIZE            = 0x00D,/*!< \brief A <tt>\ref vx_size</tt>. */
    VX_TYPE_DF_IMAGE        = 0x00E,/*!< \brief A <tt>\ref vx_df_image</tt>. */
#if defined(EXPERIMENTAL_PLATFORM_SUPPORTS_16_FLOAT)
    VX_TYPE_FLOAT16         = 0x00F,/*!< \brief A <tt>\ref vx_float16</tt>. */
#endif
    VX_TYPE_BOOL            = 0x010,/*!< \brief A <tt>\ref vx_bool</tt>. */

    /* add new scalar types here */

    VX_TYPE_SCALAR_MAX,     /*!< \brief A floating value for comparison between OpenVX scalars and OpenVX structs. */

    VX_TYPE_RECTANGLE       = 0x020,/*!< \brief A <tt>\ref vx_rectangle_t</tt>. */
    VX_TYPE_KEYPOINT        = 0x021,/*!< \brief A <tt>\ref vx_keypoint_t</tt>. */
    VX_TYPE_COORDINATES2D   = 0x022,/*!< \brief A <tt>\ref vx_coordinates2d_t</tt>. */
    VX_TYPE_COORDINATES3D   = 0x023,/*!< \brief A <tt>\ref vx_coordinates3d_t</tt>. */
    VX_TYPE_USER_STRUCT_START = 0x100, 
                                    /*!< \brief A floating value for user-defined struct base index.*/
    VX_TYPE_STRUCT_MAX      = VX_TYPE_USER_STRUCT_START - 1,     
                                    /*!< \brief A floating value for comparison between OpenVX 
                                          structs and user structs. */
    VX_TYPE_VENDOR_STRUCT_START = 0x400, 
                                    /*!< \brief A floating value for vendor-defined struct base index.*/
    VX_TYPE_USER_STRUCT_END = VX_TYPE_VENDOR_STRUCT_START - 1, 
                                    /*!< \brief A floating value for comparison between user structs and 
                                          vendor structs. */
    VX_TYPE_VENDOR_STRUCT_END = 0x7FF,   
                                    /*!< \brief A floating value for comparison between vendor 
                                          structs and OpenVX objects. */
    VX_TYPE_REFERENCE       = 0x800,/*!< \brief A <tt>\ref vx_reference</tt>. */
    VX_TYPE_CONTEXT         = 0x801,/*!< \brief A <tt>\ref vx_context</tt>. */
    VX_TYPE_GRAPH           = 0x802,/*!< \brief A <tt>\ref vx_graph</tt>. */
    VX_TYPE_NODE            = 0x803,/*!< \brief A <tt>\ref vx_node</tt>. */
    VX_TYPE_KERNEL          = 0x804,/*!< \brief A <tt>\ref vx_kernel</tt>. */
    VX_TYPE_PARAMETER       = 0x805,/*!< \brief A <tt>\ref vx_parameter</tt>. */
    VX_TYPE_DELAY           = 0x806,/*!< \brief A <tt>\ref vx_delay</tt>. */
    VX_TYPE_LUT             = 0x807,/*!< \brief A <tt>\ref vx_lut</tt>. */
    VX_TYPE_DISTRIBUTION    = 0x808,/*!< \brief A <tt>\ref vx_distribution</tt>. */
    VX_TYPE_PYRAMID         = 0x809,/*!< \brief A <tt>\ref vx_pyramid</tt>. */
    VX_TYPE_THRESHOLD       = 0x80A,/*!< \brief A <tt>\ref vx_threshold</tt>. */
    VX_TYPE_MATRIX          = 0x80B,/*!< \brief A <tt>\ref vx_matrix</tt>. */
    VX_TYPE_CONVOLUTION     = 0x80C,/*!< \brief A <tt>\ref vx_convolution</tt>. */
    VX_TYPE_SCALAR          = 0x80D,/*!< \brief A <tt>\ref vx_scalar</tt>. when needed to be completely generic for kernel validation. */
    VX_TYPE_ARRAY           = 0x80E,/*!< \brief A <tt>\ref vx_array</tt>. */
    VX_TYPE_IMAGE           = 0x80F,/*!< \brief A <tt>\ref vx_image</tt>. */
    VX_TYPE_REMAP           = 0x810,/*!< \brief A <tt>\ref vx_remap</tt>. */
    VX_TYPE_ERROR           = 0x811,/*!< \brief An error object which has no type. */
    VX_TYPE_META_FORMAT     = 0x812,/*!< \brief A <tt>\ref vx_meta_format</tt>. */

    /* \todo add new object types here */

    VX_TYPE_VENDOR_OBJECT_START  = 0xC00,/*!< \brief A floating value for vendor defined object base index. */
    VX_TYPE_OBJECT_MAX      = VX_TYPE_VENDOR_OBJECT_START - 1,/*!< \brief A value used for bound checking the OpenVX object types. */
    VX_TYPE_VENDOR_OBJECT_END   = 0xFFF,/*!< \brief A value used for bound checking of vendor objects */
};

/*! \brief The enumeration of all status codes.
 * \see vx_status.
 * \ingroup group_basic_features
 */
enum vx_status_e {
    VX_STATUS_MIN                       = -25,/*!< \brief Indicates the lower bound of status codes in VX. Used for bounds checks only. */
    /* add new codes here */
    VX_ERROR_REFERENCE_NONZERO          = -24,/*!< \brief Indicates that an operation did not complete due to a reference count being non-zero. */
    VX_ERROR_MULTIPLE_WRITERS           = -23,/*!< \brief Indicates that the graph has more than one node outputting to the same data object. This is an invalid graph structure. */
    VX_ERROR_GRAPH_ABANDONED            = -22,/*!< \brief Indicates that the graph is stopped due to an error or a callback that abandoned execution. */
    VX_ERROR_GRAPH_SCHEDULED            = -21,/*!< \brief Indicates that the supplied graph already has been scheduled and may be currently executing. */
    VX_ERROR_INVALID_SCOPE              = -20,/*!< \brief Indicates that the supplied parameter is from another scope and cannot be used in the current scope. */
    VX_ERROR_INVALID_NODE               = -19,/*!< \brief Indicates that the supplied node could not be created.*/
    VX_ERROR_INVALID_GRAPH              = -18,/*!< \brief Indicates that the supplied graph has invalid connections (cycles). */
    VX_ERROR_INVALID_TYPE               = -17,/*!< \brief Indicates that the supplied type parameter is incorrect. */
    VX_ERROR_INVALID_VALUE              = -16,/*!< \brief Indicates that the supplied parameter has an incorrect value. */
    VX_ERROR_INVALID_DIMENSION          = -15,/*!< \brief Indicates that the supplied parameter is too big or too small in dimension. */
    VX_ERROR_INVALID_FORMAT             = -14,/*!< \brief Indicates that the supplied parameter is in an invalid format. */
    VX_ERROR_INVALID_LINK               = -13,/*!< \brief Indicates that the link is not possible as specified. The parameters are incompatible. */
    VX_ERROR_INVALID_REFERENCE          = -12,/*!< \brief Indicates that the reference provided is not valid. */
    VX_ERROR_INVALID_MODULE             = -11,/*!< \brief This is returned from <tt>\ref vxLoadKernels</tt> when the module does not contain the entry point. */
    VX_ERROR_INVALID_PARAMETERS         = -10,/*!< \brief Indicates that the supplied parameter information does not match the kernel contract. */
    VX_ERROR_OPTIMIZED_AWAY             = -9,/*!< \brief Indicates that the object refered to has been optimized out of existence. */
    VX_ERROR_NO_MEMORY                  = -8,/*!< \brief Indicates that an internal or implicit allocation failed. Typically catastrophic. After detection, deconstruct the context. \see vxVerifyGraph. */
    VX_ERROR_NO_RESOURCES               = -7,/*!< \brief Indicates that an internal or implicit resource can not be acquired (not memory). This is typically catastrophic. After detection, deconstruct the context. \see vxVerifyGraph. */
    VX_ERROR_NOT_COMPATIBLE             = -6,/*!< \brief Indicates that the attempt to link two parameters together failed due to type incompatibilty. */
    VX_ERROR_NOT_ALLOCATED              = -5,/*!< \brief Indicates to the system that the parameter must be allocated by the system.  */
    VX_ERROR_NOT_SUFFICIENT             = -4,/*!< \brief Indicates that the given graph has failed verification due to an insufficient number of required parameters, which cannot be automatically created. Typically this indicates required atomic parameters. \see vxVerifyGraph. */
    VX_ERROR_NOT_SUPPORTED              = -3,/*!< \brief Indicates that the requested set of parameters produce a configuration that cannot be supported. Refer to the supplied documentation on the configured kernels. \see vx_kernel_e. */
    VX_ERROR_NOT_IMPLEMENTED            = -2,/*!< \brief Indicates that the requested kernel is missing. \see vx_kernel_e vxGetKernelByName. */
    VX_FAILURE                          = -1,/*!< \brief Indicates a generic error code, used when no other describes the error. */
    VX_SUCCESS                          =  0,/*!< \brief No error. */
};

/*! \brief A formal status type with known fixed size.
 * \see vx_status_e
 * \ingroup group_basic_features
 */
typedef vx_enum vx_status;

/*! \brief The formal typedef of the response from the callback.
 * \see vx_action_e
 * \ingroup group_node_callback
 */
typedef vx_enum vx_action;

/*! \brief A callback to the client after a particular node has completed.
 * \see vx_action
 * \see vxAssignNodeCallback
 * \param [in] node The node to which the callback was attached.
 * \return An action code from <tt>\ref vx_action_e</tt>.
 * \ingroup group_node_callback
 */
typedef vx_action (VX_CALLBACK *vx_nodecomplete_f)(vx_node node);

/*! \brief Vendor IDs are 2 nibbles in size and are located in the upper byte of
 * the 4 bytes of an enumeration.
 * \ingroup group_basic_features
 */
#define VX_VENDOR_MASK                      (0xFFF00000)

/*! \brief A type mask removes the scalar/object type from the attribute.
 * It is 3 nibbles in size and is contained between the third and second byte.
 * \see vx_type_e
 * \ingroup group_basic_features
 */
#define VX_TYPE_MASK                        (0x000FFF00)

/*! \brief A library is a set of vision kernels with its own ID supplied by a vendor.
 * The vendor defines the library ID. The range is \f$ [0,2^{8}-1] \f$ inclusive.
 * \ingroup group_basic_features
 */
#define VX_LIBRARY_MASK                     (0x000FF000)

/*! \brief An individual kernel in a library has its own unique ID within \f$ [0,2^{12}-1] \f$ (inclusive).
 * \ingroup group_basic_features
 */
#define VX_KERNEL_MASK                      (0x00000FFF)

/*! \brief An object's attribute ID is within the range of \f$ [0,2^{8}-1] \f$ (inclusive).
 * \ingroup group_basic_features
 */
#define VX_ATTRIBUTE_ID_MASK                (0x000000FF)

/*! \brief A type of enumeration. The valid range is between \f$ [0,2^{8}-1] \f$ (inclusive).
 * \ingroup group_basic_features
 */
#define VX_ENUM_TYPE_MASK                   (0x000FF000)

/*! \brief A generic enumeration list can have values between \f$ [0,2^{12}-1] \f$ (inclusive).
 * \ingroup group_basic_features
 */
#define VX_ENUM_MASK                        (0x00000FFF)

/*! \brief A macro to extract the vendor ID from the enumerated value.
 * \ingroup group_basic_features
 */
#define VX_VENDOR(e)                        (((vx_uint32)e & VX_VENDOR_MASK) >> 20)

/*! \brief A macro to extract the type from an enumerated attribute value.
 * \ingroup group_basic_features
 */
#define VX_TYPE(e)                          (((vx_uint32)e & VX_TYPE_MASK) >> 8)

/*! \brief A macro to extract the enum type from an enumerated value.
 * \ingroup group_basic_features
 */
#define VX_ENUM_TYPE(e)                     (((vx_uint32)e & VX_ENUM_TYPE_MASK) >> 12)

/*! \brief A macro to extract the kernel library enumeration from a enumerated kernel value.
 * \ingroup group_basic_features
 */
#define VX_LIBRARY(e)                       (((vx_uint32)e & VX_LIBRARY_MASK) >> 12)

#if defined(_LITTLE_ENDIAN_) || (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) || defined(_WIN32)
#define VX_DF_IMAGE(a,b,c,d)                  ((a) | (b << 8) | (c << 16) | (d << 24))
#define VX_ATTRIBUTE_BASE(vendor, object)   (((vendor) << 20) | (object << 8))
#define VX_KERNEL_BASE(vendor, lib)         (((vendor) << 20) | (lib << 12))
#define VX_ENUM_BASE(vendor, id)            (((vendor) << 20) | (id << 12))
#elif defined(_BIG_ENDIAN_) || (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define VX_DF_IMAGE(a,b,c,d)                  ((d) | (c << 8) | (b << 16) | (a << 24))
#define VX_ATTRIBUTE_BASE(vendor, object)   ((vendor) | (object << 12))
#define VX_KERNEL_BASE(vendor, lib)         ((vendor) | (lib << 12))
#define VX_ENUM_BASE(vendor, id)            ((vendor) | (id << 12))
#else
#error "Endian-ness must be defined!"
#endif

/*! \def VX_DF_IMAGE
 * \brief Converts a set of four chars into a \c uint32_t container of a VX_DF_IMAGE code.
 * \note Use a <tt>\ref vx_df_image</tt> variable to hold the value.
 * \ingroup group_basic_features
 */

/*! \def VX_ATTRIBUTE_BASE
 * \brief Defines the manner in which to combine the Vendor and Object IDs to get
 * the base value of the enumeration.
 * \ingroup group_basic_features
 */

/*! \def VX_KERNEL_BASE
 * \brief Defines the manner in which to combine the Vendor and Library IDs to get
 * the base value of the enumeration.
 * \ingroup group_basic_features
 */

/*! \def VX_ENUM_BASE
 * \brief Defines the manner in which to combine the Vendor and Object IDs to get
 * the base value of the enumeration.
 * \details From any enumerated value (with exceptions), the vendor, and enumeration
 * type should be extractable. Those types that are exceptions are
 * <tt>\ref vx_vendor_id_e</tt>, <tt>\ref vx_type_e</tt>, <tt>\ref vx_enum_e</tt>, <tt>\ref vx_df_image_e</tt>, and \c vx_bool.
 * \ingroup group_basic_features
 */

/*! \brief The set of supported enumerations in OpenVX.
 * \details These can be extracted from enumerated values using <tt>\ref VX_ENUM_TYPE</tt>.
 * \ingroup group_basic_features
 */
enum vx_enum_e {
    VX_ENUM_DIRECTION       = 0x00, /*!< \brief Parameter Direction. */
    VX_ENUM_ACTION          = 0x01, /*!< \brief Action Codes. */
    VX_ENUM_HINT            = 0x02, /*!< \brief Hint Values. */
    VX_ENUM_DIRECTIVE       = 0x03, /*!< \brief Directive Values. */
    VX_ENUM_INTERPOLATION   = 0x04, /*!< \brief Interpolation Types. */
    VX_ENUM_OVERFLOW        = 0x05, /*!< \brief Overflow Policies. */
    VX_ENUM_COLOR_SPACE     = 0x06, /*!< \brief Color Space. */
    VX_ENUM_COLOR_RANGE     = 0x07, /*!< \brief Color Space Range. */
    VX_ENUM_PARAMETER_STATE = 0x08, /*!< \brief Parameter State. */
    VX_ENUM_CHANNEL         = 0x09, /*!< \brief Channel Name. */
    VX_ENUM_CONVERT_POLICY  = 0x0A, /*!< \brief Convert Policy. */
    VX_ENUM_THRESHOLD_TYPE  = 0x0B, /*!< \brief Threshold Type List. */
    VX_ENUM_BORDER_MODE     = 0x0C, /*!< \brief Border Mode List. */
    VX_ENUM_COMPARISON      = 0x0D, /*!< \brief Comparison Values. */
    VX_ENUM_IMPORT_MEM      = 0x0E, /*!< \brief The memory import enumeration. */
    VX_ENUM_TERM_CRITERIA   = 0x0F, /*!< \brief A termination criteria. */
    VX_ENUM_NORM_TYPE       = 0x10, /*!< \brief A norm type. */
    VX_ENUM_ACCESSOR        = 0x11, /*!< \brief An accessor flag type. */
    VX_ENUM_ROUND_POLICY    = 0x12, /*!< \brief Rounding Policy. */
};

/*! \brief A return code enumeration from a <tt>\ref vx_nodecomplete_f</tt> during execution.
 * \see <tt>vxAssignNodeCallback</tt>
 * \ingroup group_node_callback
 */
enum vx_action_e {
    /*! \brief Continue executing the graph with no changes. */
    VX_ACTION_CONTINUE = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_ACTION) + 0x0,
    /*! \brief Stop executing the graph. */
    VX_ACTION_ABANDON  = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_ACTION) + 0x1,
};

/*! \brief An indication of how a kernel will treat the given parameter.
 * \ingroup group_parameter
 */
enum vx_direction_e {
    /*! \brief The parameter is an input only. */
    VX_INPUT = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_DIRECTION) + 0x0,
    /*! \brief The parameter is an output only. */
    VX_OUTPUT = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_DIRECTION) + 0x1,
    /*! \brief The parameter is both an input and output. */
    VX_BIDIRECTIONAL = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_DIRECTION) + 0x2,
};

/*! \brief These enumerations are given to the \c vxHint API to enable/disable platform
 * optimizations and/or features. Hints are optional and usually are vendor-specific.
 * \see <tt>vxHint</tt>
 * \ingroup group_hint
 */
enum vx_hint_e {
    /*! \brief Indicates to the implementation that the user wants to disable
     * any parallelization techniques. Implementations may not be parallelized,
     * so this is a hint only.
     */
    VX_HINT_SERIALIZE = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_HINT) + 0x0,
};

/*! \brief These enumerations are given to the \c vxDirective API to enable/disable
 * platform optimizations and/or features. Directives are not optional and
 * usually are vendor-specific, by defining a vendor range of directives and
 * starting their enumeration from there.
 * \see <tt>vxDirective</tt>
 * \ingroup group_directive
 */
enum vx_directive_e {
    /*! \brief Disables recording information for graph debugging. */
    VX_DIRECTIVE_DISABLE_LOGGING = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_DIRECTIVE) + 0x0,
    /*! \brief Enables recording information for graph debugging. */
    VX_DIRECTIVE_ENABLE_LOGGING = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_DIRECTIVE) + 0x1,
};

/*! \brief The Conversion Policy Enumeration.
 * \ingroup group_basic_features
 */
enum vx_convert_policy_e {
    /*! \brief Results are the least significant bits of the output operand, as if
     * stored in two's complement binary format in the size of its bit-depth.
     */
    VX_CONVERT_POLICY_WRAP = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CONVERT_POLICY) + 0x0,
    /*! \brief Results are saturated to the bit depth of the output operand. */
    VX_CONVERT_POLICY_SATURATE = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CONVERT_POLICY) + 0x1,
};

/*! \brief Based on the VX_DF_IMAGE definition.
 * \note Use <tt>\ref vx_df_image</tt> to contain these values.
 * \ingroup group_basic_features
 */
enum vx_df_image_e {
    /*! \brief A virtual image of no defined type. */
    VX_DF_IMAGE_VIRT = VX_DF_IMAGE('V','I','R','T'),
    /*! \brief A single plane of 24-bit pixel as 3 interleaved 8-bit units of
     * R then G then B data. This uses the BT709 full range by default.
     */
    VX_DF_IMAGE_RGB  = VX_DF_IMAGE('R','G','B','2'),
    /*! \brief A single plane of 32-bit pixel as 4 interleaved 8-bit units of
     * R then G then B data, then a <i>don't care</i> byte.
     * This uses the BT709 full range by default.
     */
    VX_DF_IMAGE_RGBX = VX_DF_IMAGE('R','G','B','A'),
    /*! \brief A 2-plane YUV format of Luma (Y) and interleaved UV data at
     * 4:2:0 sampling. This uses the BT709 full range by default.
     */
    VX_DF_IMAGE_NV12 = VX_DF_IMAGE('N','V','1','2'),
    /*! \brief A 2-lane YUV format of Luma (Y) and interleaved VU data at
     * 4:2:0 sampling. This uses the BT709 full range by default.
     */
    VX_DF_IMAGE_NV21 = VX_DF_IMAGE('N','V','2','1'),
    /*! \brief A single plane of 32-bit macro pixel of U0, Y0, V0, Y1 bytes.
     * This uses the BT709 full range by default.
     */
    VX_DF_IMAGE_UYVY = VX_DF_IMAGE('U','Y','V','Y'),
    /*! \brief A single plane of 32-bit macro pixel of Y0, U0, Y1, V0 bytes.
     * This uses the BT709 full range by default.
     */
    VX_DF_IMAGE_YUYV = VX_DF_IMAGE('Y','U','Y','V'),
    /*! \brief A 3 plane of 8-bit 4:2:0 sampled Y, U, V planes.
     * This uses the BT709 full range by default.
     */
    VX_DF_IMAGE_IYUV = VX_DF_IMAGE('I','Y','U','V'),
    /*! \brief A 3 plane of 8 bit 4:4:4 sampled Y, U, V planes.
     * This uses the BT709 full range by default.
     */
    VX_DF_IMAGE_YUV4 = VX_DF_IMAGE('Y','U','V','4'),
    /*! \brief A single plane of unsigned 8-bit data.
     * The range of data is not specified, as it may be extracted from a YUV or
     * generated.
     */
    VX_DF_IMAGE_U8 = VX_DF_IMAGE('U','0','0','8'),
    /*! \brief A single plane of unsigned 16-bit data.
     * The range of data is not specified, as it may be extracted from a YUV or
     * generated.
     */
    VX_DF_IMAGE_U16  = VX_DF_IMAGE('U','0','1','6'),
    /*! \brief A single plane of signed 16-bit data.
     * The range of data is not specified, as it may be extracted from a YUV or
     * generated.
     */
    VX_DF_IMAGE_S16  = VX_DF_IMAGE('S','0','1','6'),
    /*! \brief A single plane of unsigned 32-bit data.
     * The range of data is not specified, as it may be extracted from a YUV or
     * generated.
     */
    VX_DF_IMAGE_U32  = VX_DF_IMAGE('U','0','3','2'),
    /*! \brief A single plane of unsigned 32-bit data.
     * The range of data is not specified, as it may be extracted from a YUV or
     * generated.
     */
    VX_DF_IMAGE_S32  = VX_DF_IMAGE('S','0','3','2'),
};

/*! \brief The reference attributes list.
 * \ingroup group_reference
 */
enum vx_reference_attribute_e {
    /*! \brief Returns the reference count of the object. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_REF_ATTRIBUTE_COUNT = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_REFERENCE) + 0x0,
    /*! \brief Returns the <tt>\ref vx_type_e</tt> of the reference. Use a <tt>\ref vx_enum</tt> parameter. */
    VX_REF_ATTRIBUTE_TYPE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_REFERENCE) + 0x1,
};

/*! \brief A list of context attributes.
 * \ingroup group_context
 */
enum vx_context_attribute_e {
    /*! \brief Queries the unique vendor ID. Use a <tt>\ref vx_uint16</tt>. */
    VX_CONTEXT_ATTRIBUTE_VENDOR_ID = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0x0,
    /*! \brief Queries the OpenVX Version Number. Use a <tt>\ref vx_uint16</tt> */
    VX_CONTEXT_ATTRIBUTE_VERSION = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0x1,
    /*! \brief Queries the context for the number of \e unique kernels. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNELS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0x2,
    /*! \brief Queries the context for the number of active modules. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_CONTEXT_ATTRIBUTE_MODULES = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0x3,
    /*! \brief Queries the context for the number of active references. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_CONTEXT_ATTRIBUTE_REFERENCES = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0x4,
    /*! \brief Queries the context for it's implementation name. Use a <tt>\ref vx_char</tt>[<tt>\ref VX_MAX_IMPLEMENTATION_NAME</tt>] array */
    VX_CONTEXT_ATTRIBUTE_IMPLEMENTATION = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0x5,
    /*! \brief Queries the number of bytes in the extensions string. Use a <tt>\ref vx_size</tt> parameter. */
    VX_CONTEXT_ATTRIBUTE_EXTENSIONS_SIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0x6,
    /*! \brief Retrieves the extensions string. This is a space-separated string of extension names. Use a <tt>\ref vx_char</tt> pointer allocated to the size returned from <tt>\ref VX_CONTEXT_ATTRIBUTE_EXTENSIONS_SIZE</tt>. */
    VX_CONTEXT_ATTRIBUTE_EXTENSIONS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0x7,
    /*! \brief The maximum width or height of a convolution matrix.
     * Use a <tt>\ref vx_size</tt> parameter.
     * Each vendor must support centered kernels of size w X h, where both w
     * and h are odd numbers, 3 <= w <= n and 3 <= h <= n, where n is the value of the
     * <tt>\ref VX_CONTEXT_ATTRIBUTE_CONVOLUTION_MAXIMUM_DIMENSION</tt> attribute. n is an odd
     * number that should not be smaller than 9. w and h may or may not be equal to
     * each other. All combinations of w and h meeting the conditions above must be
     * supported. The behavior of <tt>\ref vxCreateConvolution</tt> is undefined for values
     * larger than the value returned by this attribute.
     */
    VX_CONTEXT_ATTRIBUTE_CONVOLUTION_MAXIMUM_DIMENSION = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0x8,
    /*! \brief The maximum window dimension of the OpticalFlowPyrLK kernel.
     * \see <tt>\ref VX_KERNEL_OPTICAL_FLOW_PYR_LK</tt>. Use a <tt>\ref vx_size</tt> parameter.
     */
    VX_CONTEXT_ATTRIBUTE_OPTICAL_FLOW_WINDOW_MAXIMUM_DIMENSION = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0x9,
    /*! \brief The border mode for immediate mode functions.
     * \details Graph mode functions are unaffected by this attribute. Use a pointer to a <tt>\ref vx_border_mode_t</tt> structure as parameter.
     * \note The assumed default value for immediate mode functions is <tt>\ref VX_BORDER_MODE_UNDEFINED</tt>.
     */
    VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_MODE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0xA,
    /*! \brief Returns the table of all unique the kernels that exist in the context.
     *  Use a <tt>\ref vx_kernel_info_t</tt> array.
     * \pre You must call <tt>\ref vxQueryContext</tt> with <tt>\ref VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNELS</tt>
     * to compute the necessary size of the array.
     */
    VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNEL_TABLE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0xB,
};

/*! \brief The kernel attributes list
 * \ingroup group_kernel
 */
enum vx_kernel_attribute_e {
    /*! \brief Queries a kernel for the number of parameters the kernel
     * supports. Use a <tt>\ref vx_uint32</tt> parameter.
     */
    VX_KERNEL_ATTRIBUTE_PARAMETERS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_KERNEL) + 0x0,
    /*! \brief Queries the name of the kernel. Not settable.
     * Use a <tt>\ref vx_char</tt>[<tt>\ref VX_MAX_KERNEL_NAME</tt>] array (not a <tt>\ref vx_array</tt>).
     */
    VX_KERNEL_ATTRIBUTE_NAME = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_KERNEL) + 0x1,
    /*! \brief Queries the enum of the kernel. Not settable.
     * Use a <tt>\ref vx_enum</tt> parameter.
     */
    VX_KERNEL_ATTRIBUTE_ENUM = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_KERNEL) + 0x2,
    /*! \brief The local data area allocated with each kernel when it becomes a
     * node. Use a <tt>\ref vx_size</tt> parameter.
     * \note If not set it will default to zero.
     */
    VX_KERNEL_ATTRIBUTE_LOCAL_DATA_SIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_KERNEL) + 0x3,
    /*! \brief The local data pointer allocated with each kernel when it becomes
     * a node. Use a void pointer parameter.
     * Use a <tt>\ref vx_size</tt> parameter.
     */
    VX_KERNEL_ATTRIBUTE_LOCAL_DATA_PTR = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_KERNEL) + 0x4,
};

/*! \brief The node attributes list.
 * \ingroup group_node
 */
enum vx_node_attribute_e {
    /*! \brief Queries the status of node execution. Use a <tt>\ref vx_status</tt> parameter. */
    VX_NODE_ATTRIBUTE_STATUS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_NODE) + 0x0,
    /*! \brief Queries the performance of the node execution. Use a <tt>\ref vx_perf_t</tt> parameter. */
    VX_NODE_ATTRIBUTE_PERFORMANCE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_NODE) + 0x1,
    /*! \brief Gets or sets the border mode of the node.
     * Use a <tt>\ref vx_border_mode_t</tt> structure.
     */
    VX_NODE_ATTRIBUTE_BORDER_MODE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_NODE) + 0x2,
    /*! \brief Indicates the size of the kernel local memory area.
     * Use a <tt>\ref vx_size</tt> parameter.
     */
    VX_NODE_ATTRIBUTE_LOCAL_DATA_SIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_NODE) + 0x3,
    /*! \brief Indicates the pointer kernel local memory area.
     * Use a void * parameter.
     */
    VX_NODE_ATTRIBUTE_LOCAL_DATA_PTR = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_NODE) + 0x4,
};

/*! \brief The parameter attributes list
 * \ingroup group_parameter
 */
enum vx_parameter_attribute_e {
    /*! \brief Queries a parameter for its index value on the kernel with which it is associated. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_PARAMETER_ATTRIBUTE_INDEX = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_PARAMETER) + 0x0,
    /*! \brief Queries a parameter for its direction value on the kernel with which it is associated. Use a <tt>\ref vx_enum</tt> parameter. */
    VX_PARAMETER_ATTRIBUTE_DIRECTION = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_PARAMETER) + 0x1,
    /*! \brief Queries a parameter for its type, \ref vx_type_e is returned. The size of the parameter is implied for plain data objects. For opaque data objects like images and arrays a query to their attributes has to be called to determine the size. */
    VX_PARAMETER_ATTRIBUTE_TYPE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_PARAMETER) + 0x2,
    /*! \brief Queries a parameter for its state. A value in <tt>\ref vx_parameter_state_e</tt> is returned. Use a <tt>\ref vx_enum</tt> parameter. */
    VX_PARAMETER_ATTRIBUTE_STATE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_PARAMETER) + 0x3,
    /*! \brief Use to extract the reference contained in the parameter. Use a <tt>\ref vx_reference</tt> parameter.  */
    VX_PARAMETER_ATTRIBUTE_REF = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_PARAMETER) + 0x4,
};

/*! \brief The image attributes list.
 * \ingroup group_image
 */
enum vx_image_attribute_e {
    /*! \brief Queries an image for its height. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_IMAGE_ATTRIBUTE_WIDTH = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0x0,
    /*! \brief Queries an image for its width. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_IMAGE_ATTRIBUTE_HEIGHT = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0x1,
    /*! \brief Queries an image for its format. Use a <tt>\ref vx_df_image</tt> parameter. */
    VX_IMAGE_ATTRIBUTE_FORMAT = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0x2,
    /*! \brief Queries an image for its number of planes. Use a <tt>\ref vx_size</tt> parameter. */
    VX_IMAGE_ATTRIBUTE_PLANES = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0x3,
    /*! \brief Queries an image for its color space (see <tt>\ref vx_color_space_e</tt>). Use a <tt>\ref vx_enum</tt> parameter. */
    VX_IMAGE_ATTRIBUTE_SPACE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0x4,
    /*! \brief Queries an image for its channel range (see <tt>\ref vx_channel_range_e</tt>). Use a <tt>\ref vx_enum</tt> parameter. */
    VX_IMAGE_ATTRIBUTE_RANGE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0x5,
    /*! \brief Queries an image for its total number of bytes. Use a <tt>\ref vx_size</tt> parameter. */
    VX_IMAGE_ATTRIBUTE_SIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0x6,
};

/*! \brief The scalar attributes list.
 * \ingroup group_scalar
 */
enum vx_scalar_attribute_e {
    /*! \brief Queries the type of atomic that is contained in the scalar. Use a <tt>\ref vx_enum</tt> parameter.*/
    VX_SCALAR_ATTRIBUTE_TYPE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_SCALAR) + 0x0,
};

/*! \brief The graph attributes list.
 * \ingroup group_graph
 */
enum vx_graph_attribute_e {
    /*! \brief Returns the number of nodes in a graph. Use a <tt>\ref vx_uint32</tt> parameter.*/
    VX_GRAPH_ATTRIBUTE_NUMNODES = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_GRAPH) + 0x0,
    /*! \brief Returns the overall status of the graph. Use a <tt>\ref vx_status</tt> parameter.*/
    VX_GRAPH_ATTRIBUTE_STATUS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_GRAPH) + 0x1,
    /*! \brief Returns the overall performance of the graph. Use a <tt>\ref vx_perf_t</tt> parameter. */
    VX_GRAPH_ATTRIBUTE_PERFORMANCE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_GRAPH) + 0x2,
    /*! \brief Returns the number of explicitly declared parameters on the graph. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_GRAPH_ATTRIBUTE_NUMPARAMETERS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_GRAPH) + 0x3,
};

/*! \brief The Look-Up Table (LUT) attribute list.
 * \ingroup group_lut
 */
enum vx_lut_attribute_e {
    /*! \brief Indicates the value type of the LUT. Use a <tt>\ref vx_enum</tt>. */
    VX_LUT_ATTRIBUTE_TYPE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS,VX_TYPE_LUT) + 0x0,
    /*! \brief Indicates the number of elements in the LUT. Use a <tt>\ref vx_size</tt>. */
    VX_LUT_ATTRIBUTE_COUNT = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS,VX_TYPE_LUT) + 0x1,
    /*! \brief Indicates the total size of the LUT in bytes. Uses a <tt>\ref vx_size</tt>. */
    VX_LUT_ATTRIBUTE_SIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS,VX_TYPE_LUT) + 0x2,
};

/*! \brief The distribution attribute list.
 * \ingroup group_distribution
 */
enum vx_distribution_attribute_e {
    /*! \brief Indicates the number of dimensions in the distribution. Use a <tt>\ref vx_size</tt> parameter. */
    VX_DISTRIBUTION_ATTRIBUTE_DIMENSIONS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_DISTRIBUTION) + 0x0,
    /*! \brief Indicates the start of the values to use (inclusive). Use a <tt>\ref vx_int32</tt> parameter. */
    VX_DISTRIBUTION_ATTRIBUTE_OFFSET = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_DISTRIBUTION) + 0x1,
    /*! \brief Indicates end value to use as the range. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_DISTRIBUTION_ATTRIBUTE_RANGE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_DISTRIBUTION) + 0x2,
    /*! \brief Indicates the number of bins. Use a <tt>\ref vx_size</tt> parameter. */
    VX_DISTRIBUTION_ATTRIBUTE_BINS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_DISTRIBUTION) + 0x3,
    /*! \brief Indicates the range of a bin. Use a <tt>\ref vx_uint32</tt> parameter.  */
    VX_DISTRIBUTION_ATTRIBUTE_WINDOW = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_DISTRIBUTION) + 0x4,
    /*! \brief Indicates the total size of the distribution in bytes. Use a <tt>\ref vx_size</tt> parameter. */
    VX_DISTRIBUTION_ATTRIBUTE_SIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_DISTRIBUTION) + 0x5,
};

/*! \brief The Threshold types.
 * \ingroup group_threshold
 */
enum vx_threshold_type_e {
    /*! \brief A threshold with only 1 value. */
    VX_THRESHOLD_TYPE_BINARY = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_THRESHOLD_TYPE) + 0x0,
    /*! \brief A threshold with 2 values (upper/lower). Use with Canny Edge Detection. */
    VX_THRESHOLD_TYPE_RANGE = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_THRESHOLD_TYPE) + 0x1,
};

/*! \brief The threshold attributes.
 * \ingroup group_threshold
 */
enum vx_threshold_attribute_e {
    /*! \brief The value type of the threshold. Use a <tt>\ref vx_enum</tt> parameter. Will contain a <tt>\ref vx_threshold_type_e</tt>. */
    VX_THRESHOLD_ATTRIBUTE_TYPE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_THRESHOLD) + 0x0,
    /*! \brief The value of the single threshold. Use a <tt>\ref vx_int32</tt> parameter. */
    VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_THRESHOLD) + 0x1,
    /*! \brief The value of the lower threshold. Use a <tt>\ref vx_int32</tt> parameter. */
    VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_THRESHOLD) + 0x2,
    /*! \brief The value of the higher threshold. Use a <tt>\ref vx_int32</tt> parameter. */
    VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_THRESHOLD) + 0x3,
    /*! \brief The value of the TRUE threshold. Use a <tt>\ref vx_int32</tt> parameter. */
    VX_THRESHOLD_ATTRIBUTE_TRUE_VALUE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_THRESHOLD) + 0x4,
    /*! \brief The value of the FALSE threshold. Use a <tt>\ref vx_int32</tt> parameter. */
    VX_THRESHOLD_ATTRIBUTE_FALSE_VALUE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_THRESHOLD) + 0x5,
    /*! \brief The data type of the threshold's value. Use a <tt>\ref vx_enum</tt> parameter. Will contain a <tt>\ref vx_type_e</tt>.*/
    VX_THRESHOLD_ATTRIBUTE_DATA_TYPE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_THRESHOLD) + 0x6,
};

/*! \brief The matrix attributes.
 * \ingroup group_matrix
 */
enum vx_matrix_attribute_e {
    /*! \brief The value type of the matrix. Use a <tt>\ref vx_enum</tt> parameter. */
    VX_MATRIX_ATTRIBUTE_TYPE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_MATRIX) + 0x0,
    /*! \brief The M dimension of the matrix. Use a <tt>\ref vx_size</tt> parameter. */
    VX_MATRIX_ATTRIBUTE_ROWS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_MATRIX) + 0x1,
    /*! \brief The N dimension of the matrix. Use a <tt>\ref vx_size</tt> parameter. */
    VX_MATRIX_ATTRIBUTE_COLUMNS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_MATRIX) + 0x2,
    /*! \brief The total size of the matrix in bytes. Use a <tt>\ref vx_size</tt> parameter. */
    VX_MATRIX_ATTRIBUTE_SIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_MATRIX) + 0x3,
};

/*! \brief The convolution attributes.
 * \ingroup group_convolution
 */
enum vx_convolution_attribute_e {
    /*! \brief The number of rows of the convolution matrix. Use a <tt>\ref vx_size</tt> parameter. */
    VX_CONVOLUTION_ATTRIBUTE_ROWS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONVOLUTION) + 0x0,
    /*! \brief The number of columns of the convolution matrix. Use a <tt>\ref vx_size</tt> parameter. */
    VX_CONVOLUTION_ATTRIBUTE_COLUMNS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONVOLUTION) + 0x1,
    /*! \brief The scale of the convolution matrix. Use a <tt>\ref vx_uint32</tt> parameter.
     * \if OPENVX_STRICT_1_0
     * \note For 1.0, only powers of 2 are supported up to 2^31.
     * \endif
     */
    VX_CONVOLUTION_ATTRIBUTE_SCALE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONVOLUTION) + 0x2,
    /*! \brief The total size of the convolution matrix in bytes. Use a <tt>\ref vx_size</tt> parameter. */
    VX_CONVOLUTION_ATTRIBUTE_SIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONVOLUTION) + 0x3,
};

/*! \brief The pyramid object attributes.
 * \ingroup group_pyramid
 */
enum vx_pyramid_attribute_e {
    /*! \brief The number of levels of the pyramid. Use a <tt>\ref vx_size</tt> parameter. */
    VX_PYRAMID_ATTRIBUTE_LEVELS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_PYRAMID) + 0x0,
    /*! \brief The scale factor between each level of the pyramid. Use a <tt>\ref vx_float32</tt> parameter. */
    VX_PYRAMID_ATTRIBUTE_SCALE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_PYRAMID) + 0x1,
    /*! \brief The width of the 0th image in pixels. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_PYRAMID_ATTRIBUTE_WIDTH = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_PYRAMID) + 0x2,
    /*! \brief The height of the 0th image in pixels. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_PYRAMID_ATTRIBUTE_HEIGHT = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_PYRAMID) + 0x3,
    /*! \brief The <tt>\ref vx_df_image_e</tt> format of the image. Use a <tt>\ref vx_df_image</tt> parameter. */
    VX_PYRAMID_ATTRIBUTE_FORMAT = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_PYRAMID) + 0x4,
};

/*! \brief The remap object attributes.
 * \ingroup group_remap
 */
enum vx_remap_attribute_e {
    /*! \brief The source width. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_REMAP_ATTRIBUTE_SOURCE_WIDTH = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_REMAP) + 0x0,
    /*! \brief The source height. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_REMAP_ATTRIBUTE_SOURCE_HEIGHT = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_REMAP) + 0x1,
    /*! \brief The destination width. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_REMAP_ATTRIBUTE_DESTINATION_WIDTH = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_REMAP) + 0x2,
    /*! \brief The destination height. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_REMAP_ATTRIBUTE_DESTINATION_HEIGHT = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_REMAP) + 0x3,
};

/*! \brief The array object attributes.
 * \ingroup group_array
 */
enum vx_array_attribute_e {
    /*! \brief The type of the Array items. Use a <tt>\ref vx_enum</tt> parameter. */
    VX_ARRAY_ATTRIBUTE_ITEMTYPE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_ARRAY) + 0x0,
    /*! \brief The number of items in the Array. Use a <tt>\ref vx_size</tt> parameter. */
    VX_ARRAY_ATTRIBUTE_NUMITEMS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_ARRAY) + 0x1,
    /*! \brief The maximal number of items that the Array can hold. Use a <tt>\ref vx_size</tt> parameter. */
    VX_ARRAY_ATTRIBUTE_CAPACITY = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_ARRAY) + 0x2,
    /*! \brief Queries an array item size. Use a <tt>\ref vx_size</tt> parameter. */
    VX_ARRAY_ATTRIBUTE_ITEMSIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_ARRAY) + 0x3,
};

/*! \brief The meta format object attributes.
 * \ingroup group_user_kernels
 */
enum vx_meta_format_attribute_e {
    /*! \brief Configures a delta rectangle during kernel output parameter validation. Use a <tt>\ref vx_delta_rectangle_t</tt>. */
    VX_META_FORMAT_ATTRIBUTE_DELTA_RECTANGLE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_META_FORMAT) + 0x0,
};

/*! \brief The channel enumerations for channel extractions.
 * \see vxChannelExtractNode
 * \see vxuChannelExtract
 * \see VX_KERNEL_CHANNEL_EXTRACT
 * \ingroup group_basic_features
 */
enum vx_channel_e {
    /*! \brief Used by formats with unknown channel types. */
    VX_CHANNEL_0 = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CHANNEL) + 0x0,
    /*! \brief Used by formats with unknown channel types. */
    VX_CHANNEL_1 = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CHANNEL) + 0x1,
    /*! \brief Used by formats with unknown channel types. */
    VX_CHANNEL_2 = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CHANNEL) + 0x2,
    /*! \brief Used by formats with unknown channel types. */
    VX_CHANNEL_3 = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CHANNEL) + 0x3,

    /*! \brief Use to extract the RED channel, no matter the byte or packing order. */
    VX_CHANNEL_R = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CHANNEL) + 0x10,
    /*! \brief Use to extract the GREEN channel, no matter the byte or packing order. */
    VX_CHANNEL_G = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CHANNEL) + 0x11,
    /*! \brief Use to extract the BLUE channel, no matter the byte or packing order. */
    VX_CHANNEL_B = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CHANNEL) + 0x12,
    /*! \brief Use to extract the ALPHA channel, no matter the byte or packing order. */
    VX_CHANNEL_A = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CHANNEL) + 0x13,
    /*! \brief Use to extract the LUMA channel, no matter the byte or packing order. */
    VX_CHANNEL_Y = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CHANNEL) + 0x14,
    /*! \brief Use to extract the Cb/U channel, no matter the byte or packing order. */
    VX_CHANNEL_U = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CHANNEL) + 0x15,
    /*! \brief Use to extract the Cr/V/Value channel, no matter the byte or packing order. */
    VX_CHANNEL_V = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_CHANNEL) + 0x16,
};

/*! \brief An enumeration of memory import types.
 * \ingroup group_context
 */
enum vx_import_type_e {
    /*! \brief For memory allocated through OpenVX, this is the import type. */
    VX_IMPORT_TYPE_NONE = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMPORT_MEM) + 0x0,

    /*! \brief The default memory type to import from the Host. */
    VX_IMPORT_TYPE_HOST = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMPORT_MEM) + 0x1,
};

/*! \brief The image reconstruction filters supported by image resampling operations.
 *
 * The edge of a pixel is interpreted as being aligned to the edge of the image.
 * The value for an output pixel is evaluated at the center of that pixel.
 *
 * This means, for example, that an even enlargement of a factor of two in nearest-neighbor
 * interpolation will replicate every source pixel into a 2x2 quad in the destination, and that
 * an even shrink by a factor of two in bilinear interpolation will create each destination pixel
 * by average a 2x2 quad of source pixels.
 *
 * Samples that cross the boundary of the source image have values determined by the border
 * mode - see <tt>\ref vx_border_mode_e</tt> and <tt>\ref VX_NODE_ATTRIBUTE_BORDER_MODE</tt>.
 * \see vxuScaleImage
 * \see vxScaleImageNode
 * \see VX_KERNEL_SCALE_IMAGE
 * \see vxuWarpAffine
 * \see vxWarpAffineNode
 * \see VX_KERNEL_WARP_AFFINE
 * \see vxuWarpPerspective
 * \see vxWarpPerspectiveNode
 * \see VX_KERNEL_WARP_PERSPECTIVE
 * \ingroup group_basic_features
 */
enum vx_interpolation_type_e {
    /*! \brief Output values are defined to match the source pixel whose center is nearest to the sample position. */
    VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_INTERPOLATION) + 0x0,
    /*! \brief Output values are defined by bilinear interpolation between the pixels whose centers are closest
     * to the sample position, weighted linearly by the distance of the sample from the pixel centers. */
    VX_INTERPOLATION_TYPE_BILINEAR = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_INTERPOLATION) + 0x1,
    /*! \brief Output values are determined by averaging the source pixels whose areas fall under the
     * area of the destination pixel, projected onto the source image. */
    VX_INTERPOLATION_TYPE_AREA = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_INTERPOLATION) + 0x2,
};

/*! \brief The image color space list used by the <tt>\ref VX_IMAGE_ATTRIBUTE_SPACE</tt> attribute of a <tt>\ref vx_image</tt>.
 * \ingroup group_image
 */
enum vx_color_space_e {
    /*! \brief Use to indicate that no color space is used. */
    VX_COLOR_SPACE_NONE = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_COLOR_SPACE) + 0x0,
    /*! \brief Use to indicate that the BT.601 coefficients and SMPTE C primaries are used for conversions. */
    VX_COLOR_SPACE_BT601_525 = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_COLOR_SPACE) + 0x1,
    /*! \brief Use to indicate that the BT.601 coefficients and BTU primaries are used for conversions. */
    VX_COLOR_SPACE_BT601_625 = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_COLOR_SPACE) + 0x2,
    /*! \brief Use to indicate that the BT.709 coefficients are used for conversions. */
    VX_COLOR_SPACE_BT709 = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_COLOR_SPACE) + 0x3,

    /*! \brief All images in VX are by default BT.709 */
    VX_COLOR_SPACE_DEFAULT = VX_COLOR_SPACE_BT709,
};

/*! \brief The image channel range list used by the <tt>\ref VX_IMAGE_ATTRIBUTE_RANGE</tt> attribute of a <tt>\ref vx_image</tt>.
 *  \ingroup group_image
 */
enum vx_channel_range_e {
    /*! \brief Full range of the unit of the channel */
    VX_CHANNEL_RANGE_FULL = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_COLOR_RANGE) + 0x0,
    /*! \brief Restricted range of the unit of the channel based on the space given */
    VX_CHANNEL_RANGE_RESTRICTED = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_COLOR_RANGE) + 0x1,
};

/*! \brief The parameter state type.
 * \ingroup group_parameter
 */
enum vx_parameter_state_e {
    /*! \brief Default. The parameter must be supplied. If not set, during
     * Verify, an error is returned.
     */
    VX_PARAMETER_STATE_REQUIRED = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_PARAMETER_STATE) + 0x0,
    /*! \brief The parameter may be unspecified. The kernel takes care not
     * to deference optional parameters until it is certain they are valid.
     */
    VX_PARAMETER_STATE_OPTIONAL = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_PARAMETER_STATE) + 0x1,
};

/*! \brief The border mode list.
 * \ingroup group_borders
 */
enum vx_border_mode_e {
    /*! \brief No defined border mode behavior is given. */
    VX_BORDER_MODE_UNDEFINED = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_BORDER_MODE) + 0x0,
    /*! \brief For nodes that support this behavior, a constant value is
     * \e filled-in when accessing out-of-bounds pixels.
     */
    VX_BORDER_MODE_CONSTANT = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_BORDER_MODE) + 0x1,
    /*! \brief For nodes that support this behavior, a replication of the nearest
     * edge pixels value is given for out-of-bounds pixels.
     */
    VX_BORDER_MODE_REPLICATE = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_BORDER_MODE) + 0x2,
};

/*! \brief The termination criteria list.
 * \see group_vision_function_opticalflowpyrlk
 * \ingroup group_context
 */
enum vx_termination_criteria_e {
    /*! \brief Indicates a termination after a set number of iterations. */
    VX_TERM_CRITERIA_ITERATIONS = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_TERM_CRITERIA) + 0x0,
    /*! \brief Indicates a termination after matching against the value of eplison provided to the function. */
    VX_TERM_CRITERIA_EPSILON = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_TERM_CRITERIA) + 0x1,
    /*! \brief Indicates that both an iterations and eplison method are employed. Whichever one matches first
     * causes the termination.
     */
    VX_TERM_CRITERIA_BOTH = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_TERM_CRITERIA) + 0x2,
};

/*! \brief A normalization type.
 * \see group_vision_function_canny
 * \ingroup group_vision_function_canny
 */
enum vx_norm_type_e {
    /*! \brief The L1 normalization. */
    VX_NORM_L1 = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NORM_TYPE) + 0x0,
    /*! \brief The L2 normalization. */
    VX_NORM_L2 = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NORM_TYPE) + 0x1,
};

/*! \brief The delay attribute list.
 * \ingroup group_delay
 */
enum vx_delay_attribute_e {
    /*! \brief The type of reference contained in the delay. Use a <tt>\ref vx_enum</tt> parameter. */
    VX_DELAY_ATTRIBUTE_TYPE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_DELAY) + 0x0,
    /*! \brief The number of items in the delay. Use a <tt>\ref vx_size</tt> parameter.*/
    VX_DELAY_ATTRIBUTE_SLOTS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_DELAY) + 0x1,
};

/*! \brief The memory accessor hint flags.
 * These enumeration values are used to indicate desired \e system behavior,
 * not the \b User intent. For example: these can be interpretted as hints to the
 * system about cache operations or marshalling operations.
 * \ingroup group_context
 */
enum vx_accessor_e {
    /*! \brief The memory shall be treated by the system as if it were read-only.
     * If the User writes to this memory, the results are implementation defined.
     */
    VX_READ_ONLY = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_ACCESSOR) + 0x1,
    /*! \brief The memory shall be treated by the system as if it were write-only.
     * If the User reads from this memory, the results are implementation defined.
     */
    VX_WRITE_ONLY = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_ACCESSOR) + 0x2,
    /*! \brief The memory shall be treated by the system as if it were readable and writeable.
     */
    VX_READ_AND_WRITE = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_ACCESSOR) + 0x3,
};

/*! \brief The Round Policy Enumeration.
 * \ingroup group_context
 */
enum vx_round_policy_e {
    /*! \brief When scaling, this truncates the least significant values that are lost in operations. */
    VX_ROUND_POLICY_TO_ZERO = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_ROUND_POLICY) + 0x1,
    /*! \brief When scaling, this rounds to nearest even output value. */
    VX_ROUND_POLICY_TO_NEAREST_EVEN = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_ROUND_POLICY) + 0x2,
};

/*!
 * \brief The entry point into modules loaded by <tt>\ref vxLoadKernels</tt>.
 * \param [in] context The handle to the implementation context.
 * \note The symbol exported from the user module must be <tt>vxPublishKernels</tt> in extern C format.
 * \ingroup group_user_kernels
 */
typedef vx_status (VX_API_CALL *vx_publish_kernels_f)(vx_context context);

/*!
 * \brief The pointer to the Host side kernel.
 * \param [in] node The handle to the node that contains this kernel.
 * \param [in] parameters The array of parameter references.
 * \param [in] num The number of parameters.
 * \ingroup group_user_kernels
 */
typedef vx_status (VX_CALLBACK *vx_kernel_f)(vx_node node, const vx_reference *parameters, vx_uint32 num);

/*!
 * \brief The pointer to the kernel initializer. If the host code requires a call
 * to initialize data once all the parameters have been validated, this function is called
 * if not NULL.
 * \param [in] node The handle to the node that contains this kernel.
 * \param [in] parameters The array of parameter references.
 * \param [in] num The number of parameters.
 * \ingroup group_user_kernels
 */
typedef vx_status (VX_CALLBACK *vx_kernel_initialize_f)(vx_node node, const vx_reference *parameters, vx_uint32 num);

/*!
 * \brief The pointer to the kernel deinitializer. If the host code requires a call
 * to deinitialize data during a node garbage collection, this function is called
 * if not NULL.
 * \param [in] node The handle to the node that contains this kernel.
 * \param [in] parameters The array of parameter references.
 * \param [in] num The number of parameters.
 * \ingroup group_user_kernels
 */
typedef vx_status (VX_CALLBACK *vx_kernel_deinitialize_f)(vx_node node, const vx_reference *parameters, vx_uint32 num);

/*!
 * \brief The user-defined kernel node input parameter validation function.
 * \note This function is called once for each VX_INPUT or VX_BIDIRECTIONAL
 * parameter index.
 * \param [in] node The handle to the node that is being validated.
 * \param [in] index The index of the parameter being validated.
 * \return An error code describing the validation status on this
 * parameter.
 * \retval VX_ERROR_INVALID_FORMAT The parameter format was incorrect.
 * \retval VX_ERROR_INVALID_VALUE The value of the parameter was incorrect.
 * \retval VX_ERROR_INVALID_DIMENSION The dimensionality of the parameter was incorrect.
 * \retval VX_ERROR_INVALID_PARAMETERS The index was out of bounds.
 * \ingroup group_user_kernels
 */
typedef vx_status (VX_CALLBACK *vx_kernel_input_validate_f)(vx_node node, vx_uint32 index);

/*!
 * \brief The user-defined kernel node output parameter validation function. The function only
 * needs to fill in the meta data structure.
 * \note This function is called once for each VX_OUTPUT parameter index.
 * \param [in] node The handle to the node that is being validated.
 * \param [in] index The index of the parameter being validated.
 * \param [in] ptr A pointer to a pre-allocated structure that the system holds.
 * The validation function fills in the correct type, format, and dimensionality for
 * the system to use either to create memory or to check against existing memory.
 * \return An error code describing the validation status on this
 * parameter.
 * \retval VX_ERROR_INVALID_PARAMETERS The index is out of bounds.
 * \ingroup group_user_kernels
 */
typedef vx_status (VX_CALLBACK *vx_kernel_output_validate_f)(vx_node node, vx_uint32 index, vx_meta_format meta);

#if defined(_WIN32) || defined(UNDER_CE)
#if defined(_WIN64)
/*! Use to aid in debugging values in OpenVX.
 * \ingroup group_basic_features
 */
#define VX_FMT_REF  "%I64u"
/*! Use to aid in debugging values in OpenVX.
 * \ingroup group_basic_features
 */
#define VX_FMT_SIZE "%I64u"
#else
/*! Use to aid in debugging values in OpenVX.
 * \ingroup group_basic_features
 */
#define VX_FMT_REF  "%lu"
/*! Use to aid in debugging values in OpenVX.
 * \ingroup group_basic_features
 */
#define VX_FMT_SIZE "%lu"
#endif
#else
/*! Use to aid in debugging values in OpenVX.
 * \ingroup group_basic_features
 */
#define VX_FMT_REF  "%p"
/*! Use to aid in debugging values in OpenVX.
 * \ingroup group_basic_features
 */
#define VX_FMT_SIZE "%zu"
#endif
/*! Use to indicate the 1:1 ratio in Q22.10 format.
 * \ingroup group_basic_features
 */
#define VX_SCALE_UNITY (1024u)

/*!
 * \brief The addressing image patch structure is used by the Host only
 * to address pixels in an image patch. The fields of the structure are defined as:
 * \arg dim - The dimensions of the image in logical pixel units in the x & y direction.
 * \arg stride - The physical byte distance from a logical pixel to the next
 * logically adjacent pixel in the positive x or y direction.
 * \arg scale - The relationship of scaling from the primary plane (typically
 * the zero indexed plane) to this plane. An integer down-scaling factor of \f$ f \f$ shall be
 * set to a value equal to \f$ scale = \frac{unity}{f} \f$ and an integer up-scaling factor of \f$ f \f$
 * shall be set to a value of \f$ scale = unity * f \f$. \f$ unity \f$ is defined as <tt>\ref VX_SCALE_UNITY</tt>.
 * \arg step - The step is the number of logical pixel units to skip to
 * arrive at the next physically unique pixel. For example, on a plane that is
 * half-scaled in a dimension, the step in that dimension is 2 to indicate that
 * every other pixel in that dimension is an alias. This is useful in situations
 * where iteration over unique pixels is required, such as in serializing
 * or de-serializing the image patch information.
 * \see <tt>\ref vxAccessImagePatch</tt>
 * \ingroup group_image
 * \include vx_imagepatch.c
 */
typedef struct _vx_imagepatch_addressing_t {
    vx_uint32 dim_x;        /*!< \brief Width of patch in X dimension in pixels. */
    vx_uint32 dim_y;        /*!< \brief Height of patch in Y dimension in pixels. */
    vx_int32  stride_x;     /*!< \brief Stride in X dimension in bytes. */
    vx_int32  stride_y;     /*!< \brief Stride in Y dimension in bytes. */
    vx_uint32 scale_x;      /*!< \brief Scale of X dimension. For sub-sampled planes this is the scaling factor of the dimension of the plane in relation to the zero plane. Use <tt>\ref VX_SCALE_UNITY</tt> in the numerator. */
    vx_uint32 scale_y;      /*!< \brief Scale of Y dimension. For sub-sampled planes this is the scaling factor of the dimension of the plane in relation to the zero plane. Use <tt>\ref VX_SCALE_UNITY</tt> in the numerator.  */
    vx_uint32 step_x;       /*!< \brief Step of X dimension in pixels. */
    vx_uint32 step_y;       /*!< \brief Step of Y dimension in pixels. */
} vx_imagepatch_addressing_t;

/*! \brief Use to initialize a <tt>\ref vx_imagepatch_addressing_t</tt> structure on the stack.
 * \ingroup group_image
 */
#define VX_IMAGEPATCH_ADDR_INIT {0u, 0u, 0, 0, 0u, 0u, 0u, 0u}

/*! \brief The performance measurement structure.
 * \ingroup group_performance
 */
typedef struct _vx_perf_t {
    vx_uint64 tmp;          /*!< \brief Holds the last measurement. */
    vx_uint64 beg;          /*!< \brief Holds the first measurement in a set. */
    vx_uint64 end;          /*!< \brief Holds the last measurement in a set. */
    vx_uint64 sum;          /*!< \brief Holds the summation of durations. */
    vx_uint64 avg;          /*!< \brief Holds the average of the durations. */
    vx_uint64 min;          /*!< \brief Holds the minimum of the durations. */
    vx_uint64 num;          /*!< \brief Holds the number of measurements. */
    vx_uint64 max;          /*!< \brief Holds the maximum of the durations. */
} vx_perf_t;

/*! \brief Initializes a <tt>\ref vx_perf_t</tt> on the stack.
 * \ingroup group performance
 */
#define VX_PERF_INIT    {0ul, 0ul, 0ul, 0ul, 0ul, 0ul}

/*! \brief The Kernel Information Structure. This is returned by the Context
 * to indicate which kernels are available in the OpenVX implementation.
 * \ingroup group_kernel
 */
typedef struct _vx_kernel_info_t {
    /*! \brief The kernel enumeration value from <tt>\ref vx_kernel_e</tt> (or an
     * extension thereof).
     * \see vxGetKernelByEnum
     */
    vx_enum enumeration;

    /*! \brief The kernel name in dotted hierarchical format.
     * e.g. "org.khronos.openvx.sobel3x3"
     * \see vxGetKernelByName
     */
    vx_char name[VX_MAX_KERNEL_NAME];
} vx_kernel_info_t;

/*! \brief Use to indicate a half-scale pyramid.
 * \ingroup group_pyramid
 */
#define VX_SCALE_PYRAMID_HALF       (0.5f)

/*! \brief Use to indicate a ORB scaled pyramid whose scaling factor is \f$ \frac{1}{\root 4 \of {2}} \f$.
 * \ingroup group_pyramid
 */
#define VX_SCALE_PYRAMID_ORB        ((vx_float32)0.8408964f)

/*! \brief Use with the enumeration <tt>\ref VX_NODE_ATTRIBUTE_BORDER_MODE</tt> to set the
 * border mode behavior of a node that supports borders.
 * \ingroup group_borders
 */
typedef struct _vx_border_mode_t {
    /*! \brief See <tt>\ref vx_border_mode_e</tt>. */
    vx_enum mode;
    /*! \brief For the mode <tt>\ref VX_BORDER_MODE_CONSTANT</tt>, this value is
     * filled into each pixel. If there are sub-channels in the pixel then this
     * value is divided up accordingly.
     */
    vx_uint32 constant_value;
} vx_border_mode_t;

/*! \brief The keypoint data structure.
 * \ingroup group_basic_features
 */
typedef struct _vx_keypoint_t {
    vx_int32 x;                 /*!< \brief The x coordinate. */
    vx_int32 y;                 /*!< \brief The y coordinate. */
    vx_float32 strength;        /*!< \brief The strength of the keypoint. Its definition is specific to the corner detector. */
    vx_float32 scale;           /*!< \brief Initialized to 0 by corner detectors. */
    vx_float32 orientation;     /*!< \brief Initialized to 0 by corner detectors. */
    vx_int32 tracking_status;   /*!< \brief A zero indicates a lost point. Initialized to 1 by corner detectors. */
    vx_float32 error;           /*!< \brief A tracking method specific error. Initialized to 0 by corner detectors. */
} vx_keypoint_t;

/*! \brief The rectangle data structure that is shared with the users.
 * \ingroup group_basic_features
 */
typedef struct _vx_rectangle_t {
    vx_uint32 start_x;          /*!< \brief The Start X coordinate. */
    vx_uint32 start_y;          /*!< \brief The Start Y coordinate. */
    vx_uint32 end_x;            /*!< \brief The End X coordinate. */
    vx_uint32 end_y;            /*!< \brief The End Y coordinate. */
} vx_rectangle_t;

/*! \brief The changes in dimensions of the rectangle between input and output
 * images in an output parameter validator. Used in conjunction with
 * <tt>\ref VX_META_FORMAT_ATTRIBUTE_DELTA_RECTANGLE</tt> and
 * <tt>\ref vxSetMetaFormatAttribute</tt>.
 * \see vx_kernel_output_validate_f
 * \see vx_meta_format
 * \ingroup group_basic_features
 */
typedef struct _vx_delta_rectangle_t {
    vx_int32 delta_start_x; /*!< \brief The change in the start x. */
    vx_int32 delta_start_y; /*!< \brief The change in the start y. */
    vx_int32 delta_end_x;   /*!< \brief The change in the end x. */
    vx_int32 delta_end_y;   /*!< \brief The change in the end y. */
} vx_delta_rectangle_t;

/*! \brief The 2D Coordinates structure.
 * \ingroup group_basic_features
 */
typedef struct _vx_coordinates2d_t {
    vx_uint32 x;    /*!< \brief The X coordinate. */
    vx_uint32 y;    /*!< \brief The Y coordinate. */
} vx_coordinates2d_t;

/*! \brief The 3D Coordinates structure.
 * \ingroup group_basic_features
 */
typedef struct _vx_coordinates3d_t {
    vx_uint32 x;    /*!< \brief The X coordinate. */
    vx_uint32 y;    /*!< \brief The Y coordinate. */
    vx_uint32 z;    /*!< \brief The Z coordinate. */
} vx_coordinates3d_t;

/*! \brief The log callback function.
 * \ingroup group_log
 */
typedef void (VX_CALLBACK *vx_log_callback_f)(vx_context context,
                                  vx_reference ref,
                                  vx_status status,
                                  const vx_char string[]);

#endif
