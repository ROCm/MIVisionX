/* 
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#define _CRT_SECURE_NO_WARNINGS
#include "vxUtils.h"
#include "runvx_plugin_image.h"
#include "runvx_plugin_loader.h"
#include "runvx_plugin_hash.h"

///////////////////////////////////////////////////////////////////////////
// Minimal built-in MD5 fallback (public domain, RFC 1321 based)
///////////////////////////////////////////////////////////////////////////
#if !_WIN32
namespace {

struct RunVxMD5Ctx {
    uint32_t state[4];
    uint32_t count[2];
    uint8_t  buffer[64];
};

static inline uint32_t rol(uint32_t v, uint32_t n) { return (v << n) | (v >> (32 - n)); }

static void RunVxMD5Transform(uint32_t state[4], const uint8_t block[64]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t x[16];
    for (int i = 0; i < 16; i++) {
        x[i] = block[i*4] | (block[i*4+1] << 8) | (block[i*4+2] << 16) | (block[i*4+3] << 24);
    }
#define FF(a,b,c,d,k,s,t) { a += ((b & c) | (~b & d)) + x[k] + t; a = rol(a,s); a += b; }
#define GG(a,b,c,d,k,s,t) { a += ((b & d) | (c & ~d)) + x[k] + t; a = rol(a,s); a += b; }
#define HH(a,b,c,d,k,s,t) { a += (b ^ c ^ d) + x[k] + t; a = rol(a,s); a += b; }
#define II(a,b,c,d,k,s,t) { a += (c ^ (b | ~d)) + x[k] + t; a = rol(a,s); a += b; }
    // Round 1
    FF(a,b,c,d, 0, 7, 0xd76aa478); FF(d,a,b,c, 1,12, 0xe8c7b756); FF(c,d,a,b, 2,17, 0x242070db); FF(b,c,d,a, 3,22, 0xc1bdceee);
    FF(a,b,c,d, 4, 7, 0xf57c0faf); FF(d,a,b,c, 5,12, 0x4787c62a); FF(c,d,a,b, 6,17, 0xa8304613); FF(b,c,d,a, 7,22, 0xfd469501);
    FF(a,b,c,d, 8, 7, 0x698098d8); FF(d,a,b,c, 9,12, 0x8b44f7af); FF(c,d,a,b,10,17, 0xffff5bb1); FF(b,c,d,a,11,22, 0x895cd7be);
    FF(a,b,c,d,12, 7, 0x6b901122); FF(d,a,b,c,13,12, 0xfd987193); FF(c,d,a,b,14,17, 0xa679438e); FF(b,c,d,a,15,22, 0x49b40821);
    // Round 2
    GG(a,b,c,d, 1, 5, 0xf61e2562); GG(d,a,b,c, 6, 9, 0xc040b340); GG(c,d,a,b,11,14, 0x265e5a51); GG(b,c,d,a, 0,20, 0xe9b6c7aa);
    GG(a,b,c,d, 5, 5, 0xd62f105d); GG(d,a,b,c,10, 9, 0x02441453); GG(c,d,a,b,15,14, 0xd8a1e681); GG(b,c,d,a, 4,20, 0xe7d3fbc8);
    GG(a,b,c,d, 9, 5, 0x21e1cde6); GG(d,a,b,c,14, 9, 0xc33707d6); GG(c,d,a,b, 3,14, 0xf4d50d87); GG(b,c,d,a, 8,20, 0x455a14ed);
    GG(a,b,c,d,13, 5, 0xa9e3e905); GG(d,a,b,c, 2, 9, 0xfcefa3f8); GG(c,d,a,b, 7,14, 0x676f02d9); GG(b,c,d,a,12,20, 0x8d2a4c8a);
    // Round 3
    HH(a,b,c,d, 5, 4, 0xfffa3942); HH(d,a,b,c, 8,11, 0x8771f681); HH(c,d,a,b,11,16, 0x6d9d6122); HH(b,c,d,a,14,23, 0xfde5380c);
    HH(a,b,c,d, 1, 4, 0xa4beea44); HH(d,a,b,c, 4,11, 0x4bdecfa9); HH(c,d,a,b, 7,16, 0xf6bb4b60); HH(b,c,d,a,10,23, 0xbebfbc70);
    HH(a,b,c,d,13, 4, 0x289b7ec6); HH(d,a,b,c, 0,11, 0xeaa127fa); HH(c,d,a,b, 3,16, 0xd4ef3085); HH(b,c,d,a, 6,23, 0x04881d05);
    HH(a,b,c,d, 9, 4, 0xd9d4d039); HH(d,a,b,c,12,11, 0xe6db99e5); HH(c,d,a,b,15,16, 0x1fa27cf8); HH(b,c,d,a, 2,23, 0xc4ac5665);
    // Round 4
    II(a,b,c,d, 0, 6, 0xf4292244); II(d,a,b,c, 7,10, 0x432aff97); II(c,d,a,b,14,15, 0xab9423a7); II(b,c,d,a, 5,21, 0xfc93a039);
    II(a,b,c,d,12, 6, 0x655b59c3); II(d,a,b,c, 3,10, 0x8f0ccc92); II(c,d,a,b,10,15, 0xffeff47d); II(b,c,d,a, 1,21, 0x85845dd1);
    II(a,b,c,d, 8, 6, 0x6fa87e4f); II(d,a,b,c,15,10, 0xfe2ce6e0); II(c,d,a,b, 6,15, 0xa3014314); II(b,c,d,a,13,21, 0x4e0811a1);
    II(a,b,c,d, 4, 6, 0xf7537e82); II(d,a,b,c,11,10, 0xbd3af235); II(c,d,a,b, 2,15, 0x2ad7d2bb); II(b,c,d,a, 9,21, 0xeb86d391);
#undef FF
#undef GG
#undef HH
#undef II
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
}

static void RunVxMD5Init(RunVxMD5Ctx * ctx) {
    ctx->count[0] = ctx->count[1] = 0;
    ctx->state[0] = 0x67452301;
    ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE;
    ctx->state[3] = 0x10325476;
}

static void RunVxMD5Update(RunVxMD5Ctx * ctx, const uint8_t * data, size_t len) {
    size_t i, index, partLen;
    index = (size_t)((ctx->count[0] >> 3) & 0x3F);
    partLen = 64 - index;
    ctx->count[0] += (uint32_t)(len << 3);
    if (ctx->count[0] < (len << 3)) ctx->count[1]++;
    ctx->count[1] += (uint32_t)(len >> 29);
    if (len >= partLen) {
        memcpy(&ctx->buffer[index], data, partLen);
        RunVxMD5Transform(ctx->state, ctx->buffer);
        for (i = partLen; i + 64 <= len; i += 64) {
            RunVxMD5Transform(ctx->state, &data[i]);
        }
        index = 0;
    } else {
        i = 0;
    }
    memcpy(&ctx->buffer[index], &data[i], len - i);
}

static void RunVxMD5Final(uint8_t digest[16], RunVxMD5Ctx * ctx) {
    uint8_t bits[8];
    size_t index, padLen;
    for (int i = 0; i < 8; i++) {
        bits[i] = (uint8_t)((ctx->count[i/4] >> ((i%4)*8)) & 0xFF);
    }
    index = (size_t)((ctx->count[0] >> 3) & 0x3F);
    padLen = (index < 56) ? (56 - index) : (120 - index);
    static const uint8_t PADDING[64] = { 0x80 };
    RunVxMD5Update(ctx, PADDING, padLen);
    RunVxMD5Update(ctx, bits, 8);
    for (int i = 0; i < 4; i++) {
        digest[i*4]   = (uint8_t)(ctx->state[i] & 0xFF);
        digest[i*4+1] = (uint8_t)((ctx->state[i] >> 8) & 0xFF);
        digest[i*4+2] = (uint8_t)((ctx->state[i] >> 16) & 0xFF);
        digest[i*4+3] = (uint8_t)((ctx->state[i] >> 24) & 0xFF);
    }
}

} // anonymous namespace
#endif

#define IS_ALPHA(c) (((c) >= 'A' && (c) <= 'Z') || ((c) >= 'a' && (c) <= 'z'))
#define TO_UPPER(c) ((c) & 0xDF)

#if _WIN32 && ENABLE_OPENCL
#pragma comment(lib, "OpenCL.lib")
#endif

// enumeration constants
static struct { const char * name; vx_enum value; } s_table_constants[] = {
	{ "CHANNEL_0|VX_CHANNEL_0", VX_CHANNEL_0 },
	{ "CHANNEL_1|VX_CHANNEL_1", VX_CHANNEL_1 },
	{ "CHANNEL_2|VX_CHANNEL_2", VX_CHANNEL_2 },
	{ "CHANNEL_3|VX_CHANNEL_3", VX_CHANNEL_3 },
	{ "CHANNEL_R|VX_CHANNEL_R", VX_CHANNEL_R },
	{ "CHANNEL_G|VX_CHANNEL_G", VX_CHANNEL_G },
	{ "CHANNEL_B|VX_CHANNEL_B", VX_CHANNEL_B },
	{ "CHANNEL_A|VX_CHANNEL_A", VX_CHANNEL_A },
	{ "CHANNEL_Y|VX_CHANNEL_Y", VX_CHANNEL_Y },
	{ "CHANNEL_U|VX_CHANNEL_U", VX_CHANNEL_U },
	{ "CHANNEL_V|VX_CHANNEL_V", VX_CHANNEL_V },
	{ "WRAP|VX_CONVERT_POLICY_WRAP", VX_CONVERT_POLICY_WRAP },
	{ "SATURATE|VX_CONVERT_POLICY_SATURATE", VX_CONVERT_POLICY_SATURATE },
	{ "NEAREST_NEIGHBOR|VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR", VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR },
	{ "BILINEAR|VX_INTERPOLATION_TYPE_BILINEAR", VX_INTERPOLATION_TYPE_BILINEAR },
	{ "AREA|VX_INTERPOLATION_TYPE_AREA", VX_INTERPOLATION_TYPE_AREA },
	{ "BINARY|VX_THRESHOLD_TYPE_BINARY", VX_THRESHOLD_TYPE_BINARY },
	{ "RANGE|VX_THRESHOLD_TYPE_RANGE", VX_THRESHOLD_TYPE_RANGE },
	{ "NORM_L1|VX_NORM_L1", VX_NORM_L1 },
	{ "NORM_L2|VX_NORM_L2", VX_NORM_L2 },
	{ "ROUND_POLICY_TO_ZERO|VX_ROUND_POLICY_TO_ZERO", VX_ROUND_POLICY_TO_ZERO },
	{ "ROUND_POLICY_TO_NEAREST_EVEN|VX_ROUND_POLICY_TO_NEAREST_EVEN", VX_ROUND_POLICY_TO_NEAREST_EVEN },
	{ "CRITERIA_ITERATIONS|VX_TERM_CRITERIA_ITERATIONS", VX_TERM_CRITERIA_ITERATIONS },
	{ "CRITERIA_EPSILON|VX_TERM_CRITERIA_EPSILON", VX_TERM_CRITERIA_EPSILON },
	{ "CRITERIA_BOTH|VX_TERM_CRITERIA_BOTH", VX_TERM_CRITERIA_BOTH },
	{ "RECTANGLE|VX_TYPE_RECTANGLE", VX_TYPE_RECTANGLE },
	{ "KEYPOINT|VX_TYPE_KEYPOINT", VX_TYPE_KEYPOINT },
	{ "COORDINATES2D|VX_TYPE_COORDINATES2D", VX_TYPE_COORDINATES2D },
	{ "COORDINATES3D|VX_TYPE_COORDINATES3D", VX_TYPE_COORDINATES3D },
	{ "ENUM|VX_TYPE_ENUM", VX_TYPE_ENUM },
	{ "UINT64|VX_TYPE_UINT64", VX_TYPE_UINT64 },
	{ "INT64|VX_TYPE_INT64", VX_TYPE_INT64 },
	{ "UINT32|VX_TYPE_UINT32", VX_TYPE_UINT32 },
	{ "INT32|VX_TYPE_INT32", VX_TYPE_INT32 },
	{ "UINT16|VX_TYPE_UINT16", VX_TYPE_UINT16 },
	{ "INT16|VX_TYPE_INT16", VX_TYPE_INT16 },
	{ "UINT8|VX_TYPE_UINT8", VX_TYPE_UINT8 },
	{ "INT8|VX_TYPE_INT8", VX_TYPE_INT8 },
	{ "FLOAT16|VX_TYPE_FLOAT16", VX_TYPE_FLOAT16 },
	{ "FLOAT32|VX_TYPE_FLOAT32", VX_TYPE_FLOAT32 },
	{ "FLOAT64|VX_TYPE_FLOAT64", VX_TYPE_FLOAT64 },
	{ "SIZE|VX_TYPE_SIZE", VX_TYPE_SIZE },
	{ "BOOL|VX_TYPE_BOOL", VX_TYPE_BOOL },
	{ "CHAR|VX_TYPE_CHAR", VX_TYPE_CHAR },
	{ "STRING|VX_TYPE_STRING_AMD", VX_TYPE_STRING_AMD },
	{ "BORDER_MODE_UNDEFINED|VX_BORDER_MODE_UNDEFINED", VX_BORDER_MODE_UNDEFINED },
	{ "BORDER_MODE_REPLICATE|VX_BORDER_MODE_REPLICATE", VX_BORDER_MODE_REPLICATE },
	{ "BORDER_MODE_CONSTANT|VX_BORDER_MODE_CONSTANT", VX_BORDER_MODE_CONSTANT },
	{ "VX_DIRECTIVE_DISABLE_LOGGING", VX_DIRECTIVE_DISABLE_LOGGING },
	{ "VX_DIRECTIVE_ENABLE_LOGGING", VX_DIRECTIVE_ENABLE_LOGGING },
	{ "VX_DIRECTIVE_ENABLE_PERFORMANCE", VX_DIRECTIVE_ENABLE_PERFORMANCE },
	{ "VX_DIRECTIVE_READ_ONLY|VX_DIRECTIVE_AMD_READ_ONLY", VX_DIRECTIVE_AMD_READ_ONLY },
	{ "VX_DIRECTIVE_AMD_ENABLE_PROFILE_CAPTURE", VX_DIRECTIVE_AMD_ENABLE_PROFILE_CAPTURE },
	{ "VX_DIRECTIVE_AMD_DISABLE_PROFILE_CAPTURE", VX_DIRECTIVE_AMD_DISABLE_PROFILE_CAPTURE },
	{ "VX_DIRECTIVE_AMD_DISABLE_GPU_FLUSH", VX_DIRECTIVE_AMD_DISABLE_GPU_FLUSH },
	{ "VX_MEMORY_TYPE_NONE", VX_MEMORY_TYPE_NONE },
	{ "VX_MEMORY_TYPE_HOST", VX_MEMORY_TYPE_HOST },
	{ "VX_MEMORY_TYPE_OPENCL", VX_MEMORY_TYPE_OPENCL },
	{ "FULL|VX_CHANNEL_RANGE_FULL", VX_CHANNEL_RANGE_FULL },
	{ "RESTRICTED|VX_CHANNEL_RANGE_RESTRICTED", VX_CHANNEL_RANGE_RESTRICTED },
	{ "BT601_525|VX_COLOR_SPACE_BT601_525", VX_COLOR_SPACE_BT601_525 },
	{ "BT601_625|VX_COLOR_SPACE_BT601_625", VX_COLOR_SPACE_BT601_625 },
	{ "BT709|VX_COLOR_SPACE_BT709", VX_COLOR_SPACE_BT709 },
	{ "VX_NN_POOLING_MAX", VX_NN_POOLING_MAX },
	{ "VX_NN_POOLING_AVG", VX_NN_POOLING_AVG },
	{ "VX_NN_DS_SIZE_ROUNDING_FLOOR", VX_NN_DS_SIZE_ROUNDING_FLOOR },
	{ "VX_NN_DS_SIZE_ROUNDING_CEILING", VX_NN_DS_SIZE_ROUNDING_CEILING },
	{ "VX_NN_ACTIVATION_LOGISTIC", VX_NN_ACTIVATION_LOGISTIC },
	{ "VX_NN_ACTIVATION_HYPERBOLIC_TAN", VX_NN_ACTIVATION_HYPERBOLIC_TAN },
	{ "VX_NN_ACTIVATION_RELU", VX_NN_ACTIVATION_RELU },
	{ "VX_NN_ACTIVATION_BRELU", VX_NN_ACTIVATION_BRELU },
	{ "VX_NN_ACTIVATION_SOFTRELU", VX_NN_ACTIVATION_SOFTRELU },
	{ "VX_NN_ACTIVATION_ABS", VX_NN_ACTIVATION_ABS },
	{ "VX_NN_ACTIVATION_SQUARE", VX_NN_ACTIVATION_SQUARE },
	{ "VX_NN_ACTIVATION_SQRT", VX_NN_ACTIVATION_SQRT },
	{ "VX_NN_ACTIVATION_LINEAR", VX_NN_ACTIVATION_LINEAR },
	{ "VX_NN_ACTIVATION_LEAKY_RELU", VX_NN_ACTIVATION_LEAKY_RELU },
	{ "VX_NN_NORMALIZATION_SAME_MAP", VX_NN_NORMALIZATION_SAME_MAP },
	{ "VX_NN_NORMALIZATION_ACROSS_MAPS", VX_NN_NORMALIZATION_ACROSS_MAPS },
	{ "VX_TYPE_HOG_PARAMS", VX_TYPE_HOG_PARAMS },
	{ "VX_TYPE_HOUGH_LINES_PARAMS", VX_TYPE_HOUGH_LINES_PARAMS },
	{ "VX_TYPE_LINE_2D", VX_TYPE_LINE_2D },
	{ "VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS", VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS },
	{ "VX_TYPE_NN_CONVOLUTION_PARAMS", VX_TYPE_NN_CONVOLUTION_PARAMS },
	{ "VX_TYPE_NN_DECONVOLUTION_PARAMS", VX_TYPE_NN_DECONVOLUTION_PARAMS },
	{ "VX_TYPE_NN_ROI_POOL_PARAMS", VX_TYPE_NN_ROI_POOL_PARAMS },
	// error codes
	{ "VX_FAILURE", VX_FAILURE },
	{ "VX_ERROR_REFERENCE_NONZERO", VX_ERROR_REFERENCE_NONZERO },
	{ "VX_ERROR_MULTIPLE_WRITERS", VX_ERROR_MULTIPLE_WRITERS },
	{ "VX_ERROR_GRAPH_ABANDONED", VX_ERROR_GRAPH_ABANDONED },
	{ "VX_ERROR_GRAPH_SCHEDULED", VX_ERROR_GRAPH_SCHEDULED },
	{ "VX_ERROR_INVALID_SCOPE", VX_ERROR_INVALID_SCOPE },
	{ "VX_ERROR_INVALID_NODE", VX_ERROR_INVALID_NODE },
	{ "VX_ERROR_INVALID_GRAPH", VX_ERROR_INVALID_GRAPH },
	{ "VX_ERROR_INVALID_TYPE", VX_ERROR_INVALID_TYPE },
	{ "VX_ERROR_INVALID_VALUE", VX_ERROR_INVALID_VALUE },
	{ "VX_ERROR_INVALID_DIMENSION", VX_ERROR_INVALID_DIMENSION },
	{ "VX_ERROR_INVALID_FORMAT", VX_ERROR_INVALID_FORMAT },
	{ "VX_ERROR_INVALID_LINK", VX_ERROR_INVALID_LINK },
	{ "VX_ERROR_INVALID_REFERENCE", VX_ERROR_INVALID_REFERENCE },
	{ "VX_ERROR_INVALID_MODULE", VX_ERROR_INVALID_MODULE },
	{ "VX_ERROR_INVALID_PARAMETERS", VX_ERROR_INVALID_PARAMETERS },
	{ "VX_ERROR_OPTIMIZED_AWAY", VX_ERROR_OPTIMIZED_AWAY },
	{ "VX_ERROR_NO_MEMORY", VX_ERROR_NO_MEMORY },
	{ "VX_ERROR_NO_RESOURCES", VX_ERROR_NO_RESOURCES },
	{ "VX_ERROR_NOT_COMPATIBLE", VX_ERROR_NOT_COMPATIBLE },
	{ "VX_ERROR_NOT_ALLOCATED", VX_ERROR_NOT_ALLOCATED },
	{ "VX_ERROR_NOT_SUFFICIENT", VX_ERROR_NOT_SUFFICIENT },
	{ "VX_ERROR_NOT_SUPPORTED", VX_ERROR_NOT_SUPPORTED },
	{ "VX_ERROR_NOT_IMPLEMENTED", VX_ERROR_NOT_IMPLEMENTED },
	// for debug purposes only
	{ "KEYPOINT_XYS", AGO_TYPE_KEYPOINT_XYS },
	{ "VX_TYPE_LUT", VX_TYPE_LUT },
	{ "VX_TYPE_DISTRIBUTION", VX_TYPE_DISTRIBUTION },
	{ "VX_TYPE_PYRAMID", VX_TYPE_PYRAMID },
	{ "VX_TYPE_THRESHOLD", VX_TYPE_THRESHOLD },
	{ "VX_TYPE_MATRIX", VX_TYPE_MATRIX },
	{ "VX_TYPE_CONVOLUTION", VX_TYPE_CONVOLUTION },
	{ "VX_TYPE_SCALAR", VX_TYPE_SCALAR },
	{ "VX_TYPE_ARRAY", VX_TYPE_ARRAY },
	{ "VX_TYPE_IMAGE", VX_TYPE_IMAGE },
	{ "VX_TYPE_REMAP", VX_TYPE_REMAP },
	{ "VX_TYPE_TENSOR", VX_TYPE_TENSOR },
	{ "VX_TYPE_INVALID", VX_TYPE_INVALID },
	{ "VX_TYPE_STRING", VX_TYPE_STRING_AMD },
	{ "AGO_TYPE_MEANSTDDEV_DATA", AGO_TYPE_MEANSTDDEV_DATA },
	{ "AGO_TYPE_MINMAXLOC_DATA", AGO_TYPE_MINMAXLOC_DATA },
	{ "AGO_TYPE_CANNY_STACK", AGO_TYPE_CANNY_STACK },
	{ "AGO_TYPE_SCALE_MATRIX", AGO_TYPE_SCALE_MATRIX },
	{ NULL, 0 }
};

//  ovxEnum2Name -- the returns a global pointer, so returned string has to be saved by caller immediately
const char * ovxEnum2Name(vx_enum e)
{
	for (vx_uint32 i = 0; s_table_constants[i].name; i++) {
		if (s_table_constants[i].value == e) {
			static char name[128]; strcpy(name, s_table_constants[i].name);
			for (int j = 0; name[j]; j++) {
				if (name[j] == '|') {
					name[j] = '\0';
					break;
				}
			}
			return s_table_constants[i].name;
		}
	}
	return NULL;
}

//  ovxEnum2String -- return enum name or hex value as a string
void ovxEnum2String(vx_enum e, char str[])
{
	const char * name = ovxEnum2Name(e);
	if (e) strcpy(str, name);
	else snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "0x%x", e);
}

//  ovxName2Enum -- returns enum corresponding to name or hex value in the input string
vx_enum ovxName2Enum(const char * name)
{
	for (vx_uint32 i = 0; s_table_constants[i].name; i++) {
		char nameList[128]; strcpy(nameList, s_table_constants[i].name);
		// search for name in '|' separated nameList:
		//   s - points to beginning of current name in nameList
		//   t - running pointer in nameList
		for (char * s = nameList, *t = nameList;; t++) {
			if (*t == '|' || *t == '\0') {
				char tc = *t; *t = '\0';
				if (!_stricmp(s, name)) {
					// found name, so return corresponding enum value
					return s_table_constants[i].value;
				}
				if (tc == '\0')
					break; // reached end of nameList, so abort searching
				else
					s = t + 1; // make s point to beginning of next name in nameList
			}
		}
	}
	// if none found, try reading as an integer (may be user wanted to specify a hex value for enum)
	vx_enum value = 0;
	(void)sscanf(name, "%i", &value);
	return value;
}

const char * stristr(const char * str1, const char * str2)
{
	if (!*str2) return str1;
	for (const char * cp = str1; *cp; cp++) {
		const char * s2 = str2;
		for (const char * s1 = cp; *s1 && *s2 && (IS_ALPHA(*s1) && IS_ALPHA(*s2)) ? !(TO_UPPER(*s1) - TO_UPPER(*s2)) : !(*s1 - *s2); s1++)
			++s2;
		if (!*s2)
			return cp;
	}
	return nullptr;
}

///////////////////////////////////////////
// For supporting ~ in R/W file names
static char s_rootDir[512] = ".";
void SetRootDir(const char * rootDir)
{
	strcpy(s_rootDir, rootDir);
}
const char * RootDirUpdated(const char * filePath)
{
	static char updatedFilePath[8192];
	int j = 0;
	for (int i = 0; filePath[i]; i++) {
		if (filePath[i] != '~')
			updatedFilePath[j++] = filePath[i];
		else {
			for (int k = 0; s_rootDir[k]; k++)
				updatedFilePath[j++] = s_rootDir[k];
		}
	}
	updatedFilePath[j] = 0;
	return updatedFilePath;
}

vector<string> &split(const string &s, char delim, vector<string> &elems){
	if (delim == ' ') {
		const char * p = s.c_str();
		while (*p) {
			while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
				p++;
			const char * q = p;
			while (*p && !(*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
				p++;
			if (*q){
				char item[1024];
				strncpy(item, q, p - q); item[p - q] = 0;
				elems.push_back(item);
			}
		}
	}
	else {
		stringstream ss(s);
		string item;
		while (getline(ss, item, delim)){
			elems.push_back(item);
		}
	}
	return elems;
}

int convert_image_format(string format){
	if (format.size() == 4){
		return ((format[0]) | (format[1] << 8) | (format[2] << 16) | (format[3] << 24));
	}
	else{
		printf("ERROR: %s is not a proper image format\n", format.c_str());
		throw - 1;
	}
}

CHasher::CHasher(){
	memset(m_hash, 0, sizeof(m_hash));
	for (int i = 0; i < 32; i++)
		m_checkSum[i] = '0';
	m_checkSum[32] = '\0';
	m_handle = NULL;
	m_useBuiltInMD5 = false;
}

CHasher::~CHasher(){
	Shutdown();
}

void CHasher::Initialize(){
#if _WIN32
	DWORD dwStatus = 0;
	if (!CryptAcquireContext(&m_cryptProv, NULL, MS_DEF_PROV, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT))
	{
		dwStatus = GetLastError();
		printf("CryptAcquireContext failed: %d\n", dwStatus);
		throw - 1;
	}

	if (!CryptCreateHash(m_cryptProv, CALG_MD5, 0, 0, &m_cryptHash))
	{
		dwStatus = GetLastError();
		printf("CryptCreateHash failed: %d\n", dwStatus);
		CryptReleaseContext(m_cryptProv, 0);
		throw - 1;
	}
#else
	IRunVxHashPlugin * plugin = RunVxGetHashPlugin();
	if (plugin) {
		m_handle = plugin->Create();
		if (!m_handle) {
			printf("WARNING: hash plugin Create() failed; falling back to built-in MD5\n");
		}
	}
	if (!m_handle) {
		m_handle = new RunVxMD5Ctx;
		RunVxMD5Init((RunVxMD5Ctx *)m_handle);
		m_useBuiltInMD5 = true;
	}
#endif
}

void CHasher::Process(vx_uint8 * data_ptr, vx_size count){
#if _WIN32
	DWORD dwStatus = 0;
	if (!CryptHashData(m_cryptHash, (BYTE*)data_ptr, (DWORD)count, 0))
	{
		dwStatus = GetLastError();
		printf("CryptHashData failed: %d\n", dwStatus);
		CryptReleaseContext(m_cryptProv, 0);
		CryptDestroyHash(m_cryptHash);
		throw - 1;
	}
#else
	if (m_useBuiltInMD5) {
		RunVxMD5Update((RunVxMD5Ctx *)m_handle, data_ptr, count);
	}
	else {
		IRunVxHashPlugin * plugin = RunVxGetHashPlugin();
		if (plugin) {
			if (!plugin->Update(m_handle, data_ptr, count)) {
				printf("ERROR: hash plugin Update(*,*,%zu) failed\n", count);
			}
		}
	}
#endif
}

const char * CHasher::GetCheckSum(){

#if _WIN32
	DWORD cbHash = 16;
	DWORD dwStatus = 0;
	if (!CryptGetHashParam(m_cryptHash, HP_HASHVAL, m_hash, &cbHash, 0)){
		dwStatus = GetLastError();
		printf("CryptGetHashParam failed: %d\n", dwStatus);
		CryptReleaseContext(m_cryptProv, 0);
		CryptDestroyHash(m_cryptHash);
		throw - 1;
	}
#else
	if (m_useBuiltInMD5) {
		RunVxMD5Final(m_hash, (RunVxMD5Ctx *)m_handle);
	}
	else {
		IRunVxHashPlugin * plugin = RunVxGetHashPlugin();
		if (plugin) {
			if (!plugin->Final(m_handle, m_hash)) {
				printf("ERROR: hash plugin Final() failed\n");
			}
		}
	}
#endif
	char hex[] = "0123456789abcdef";
	for (int i = 0; i < 16; i++){
		m_checkSum[i * 2] = hex[m_hash[i] >> 4];
		m_checkSum[(i * 2) + 1] = hex[m_hash[i] & 0xF];
	}
	return m_checkSum;
}

void CHasher::Shutdown(){
#if _WIN32
	CryptReleaseContext(m_cryptProv, 0);
	CryptDestroyHash(m_cryptHash);
#else
	if (m_handle) {
		if (m_useBuiltInMD5) {
			delete (RunVxMD5Ctx *)m_handle;
		} else {
			IRunVxHashPlugin * plugin = RunVxGetHashPlugin();
			if (plugin) plugin->Destroy(m_handle);
		}
		m_handle = NULL;
	}
#endif
}

// Compute checksum of rectangular region specified within an image
void ComputeChecksum(char checkSumString[64], vx_image image, vx_rectangle_t * rectRegion)
{
	// get number of planes
	vx_df_image format = VX_DF_IMAGE_VIRT;
	vx_size num_planes = 0;
	ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
	ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_PLANES, &num_planes, sizeof(num_planes)));
	// compute checksum
	CHasher checksum; checksum.Initialize();
	for (vx_uint32 plane = 0; plane < (vx_uint32)num_planes; plane++) {
		vx_imagepatch_addressing_t addr;
		vx_uint8 * base_ptr = nullptr;
		ERROR_CHECK(vxAccessImagePatch(image, rectRegion, plane, &addr, (void **)&base_ptr, VX_READ_ONLY));
		vx_uint32 width = ((addr.dim_x * addr.scale_x) / VX_SCALE_UNITY);
		vx_uint32 height = ((addr.dim_y * addr.scale_y) / VX_SCALE_UNITY);
		vx_uint32 width_in_bytes = (format == VX_DF_IMAGE_U1_AMD) ? ((width + 7) >> 3) : (width * addr.stride_x);
		for (vx_uint32 y = 0; y < height; y++) {
			checksum.Process(base_ptr + y * addr.stride_y, width_in_bytes);
		}
		ERROR_CHECK(vxCommitImagePatch(image, rectRegion, plane, &addr, base_ptr));
	}
	// copy the checksum string
	strcpy(checkSumString, checksum.GetCheckSum());
}

// <template>ComparePixels -- compares an image with a reference image
template<typename PixelType, typename CompareType>
size_t ComparePixels(PixelType * pImg_, size_t img_stride_y, PixelType * pRef_, size_t ref_stride_y, vx_uint32 width, vx_uint32 height, CompareType errLimitMin, CompareType errLimitMax)
{
	const vx_uint8 * pImg = (const vx_uint8 *)pImg_;
	const vx_uint8 * pRef = (const vx_uint8 *)pRef_;
	size_t errorPixelCount = 0;
	for (vx_uint32 y = 0; y < height; y++) {
		const PixelType * p = (const PixelType *)pImg;
		const PixelType * q = (const PixelType *)pRef;
		for (size_t x = 0; x < width; x++) {
			CompareType err = (CompareType)p[x] - (CompareType)q[x];
			if (err < errLimitMin || err > errLimitMax)
				errorPixelCount++;
		}
		pImg += img_stride_y;
		pRef += ref_stride_y;
	}
	return errorPixelCount;
}

size_t ComparePixelsU001(vx_uint8 * pImg, size_t img_stride_y, vx_uint8 * pRef, size_t ref_stride_y, vx_uint32 width, vx_uint32 height)
{
	size_t errorPixelCount = 0;
	for (vx_uint32 y = 0; y < height; y++) {
		const vx_uint8 * p = (const vx_uint8 *)pImg;
		const vx_uint8 * q = (const vx_uint8 *)pRef;
		for (size_t x = 0; x < width; x++) {
			size_t bytepos = x >> 3, bitpos = x & 7;
			if ((p[bytepos] ^ q[bytepos]) & (1 << bitpos)) {
				errorPixelCount++;
			}
		}
		pImg += img_stride_y;
		pRef += ref_stride_y;
	}
	return errorPixelCount;
}

// Compare rectangular region specified within an image and return number of pixels mismatching
size_t CompareImage(vx_image image, vx_rectangle_t * rectRegion, vx_uint8 * refImage, float errLimitMin, float errLimitMax, int frameNumber, const char * fileNameRef)
{
	// get number of planes, image format, and pixel type
	vx_df_image format = VX_DF_IMAGE_VIRT;
	vx_size num_planes = 0; vx_uint32 image_width = 0, image_height = 0;
	ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &image_width, sizeof(image_width)));
	ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &image_height, sizeof(image_height)));
	ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
	ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_PLANES, &num_planes, sizeof(num_planes)));
	// set pixel type and compute frame size in bytes
	vx_enum pixelType = VX_TYPE_UINT8; // default
	if (format == VX_DF_IMAGE_S16) pixelType = VX_TYPE_INT16;
	else if (format == VX_DF_IMAGE_U16) pixelType = VX_TYPE_UINT16;
	else if (format == VX_DF_IMAGE_S32) pixelType = VX_TYPE_INT32;
	else if (format == VX_DF_IMAGE_U32) pixelType = VX_TYPE_UINT32;
	else if (format == VX_DF_IMAGE_F32_AMD || format == VX_DF_IMAGE_F32x3_AMD) pixelType = VX_TYPE_FLOAT32;
	// compare plane by plane
	vx_size errorPixelCountTotal = 0;
	vx_uint8 * pRefPlane = refImage;
	for (vx_uint32 plane = 0; plane < (vx_uint32)num_planes; plane++) {
		vx_imagepatch_addressing_t addr = { 0 };
		vx_uint8 * base_ptr = nullptr;
		ERROR_CHECK(vxAccessImagePatch(image, rectRegion, plane, &addr, (void **)&base_ptr, VX_READ_ONLY));
		vx_uint32 region_width = ((addr.dim_x * addr.scale_x) / VX_SCALE_UNITY);
		vx_uint32 region_height = (addr.dim_y * addr.scale_y) / VX_SCALE_UNITY;
		vx_uint32 plane_width = ((image_width * addr.scale_x) / VX_SCALE_UNITY);
		vx_uint32 plane_height = ((image_height * addr.scale_y) / VX_SCALE_UNITY);
		vx_uint32 plane_width_in_bytes = (format == VX_DF_IMAGE_U1_AMD) ? ((plane_width + 7) >> 3) : (plane_width * addr.stride_x);
		vx_uint32 start_x = ((rectRegion->start_x * addr.scale_x) / VX_SCALE_UNITY);
		vx_uint32 start_y = ((rectRegion->start_y * addr.scale_y) / VX_SCALE_UNITY);
		vx_uint8 * pRef = pRefPlane + start_y * plane_width_in_bytes + start_x * addr.stride_x;
		vx_size errorPixelCount = 0;
		if (pixelType == VX_TYPE_INT16) {
			errorPixelCount = ComparePixels((vx_int16 *)base_ptr, addr.stride_y, (vx_int16 *)pRef, plane_width_in_bytes, region_width, region_height, (vx_int32)errLimitMin, (vx_int32)errLimitMax);
		}
		else if (pixelType == VX_TYPE_UINT16) {
			errorPixelCount = ComparePixels((vx_uint16 *)base_ptr, addr.stride_y, (vx_uint16 *)pRef, plane_width_in_bytes, region_width, region_height, (vx_int32)errLimitMin, (vx_int32)errLimitMax);
		}
		else if (pixelType == VX_TYPE_INT32) {
			errorPixelCount = ComparePixels((vx_int32 *)base_ptr, addr.stride_y, (vx_int32 *)pRef, plane_width_in_bytes, region_width, region_height, (vx_int64)errLimitMin, (vx_int64)errLimitMax);
		}
		else if (pixelType == VX_TYPE_UINT32) {
			errorPixelCount = ComparePixels((vx_uint32 *)base_ptr, addr.stride_y, (vx_uint32 *)pRef, plane_width_in_bytes, region_width, region_height, (vx_int64)errLimitMin, (vx_int64)errLimitMax);
		}
		else if (pixelType == VX_TYPE_FLOAT32) {
			errorPixelCount = ComparePixels((vx_float32 *)base_ptr, addr.stride_y, (vx_float32 *)pRef, plane_width_in_bytes, region_width, region_height, (vx_float32)errLimitMin, (vx_float32)errLimitMax);
		}
		else if (format == VX_DF_IMAGE_U1_AMD) {
			errorPixelCount = ComparePixelsU001((vx_uint8 *)base_ptr, addr.stride_y, (vx_uint8 *)pRef, plane_width_in_bytes, region_width, region_height);
		}
		else {
			errorPixelCount = ComparePixels((vx_uint8 *)base_ptr, addr.stride_y, (vx_uint8 *)pRef, plane_width_in_bytes, region_width, region_height, (vx_int32)errLimitMin, (vx_int32)errLimitMax);
		}
		ERROR_CHECK(vxCommitImagePatch(image, rectRegion, plane, &addr, base_ptr));
		// report results
		errorPixelCountTotal += errorPixelCount;
		if (errorPixelCount > 0) {
			char name[64]; vxGetReferenceName((vx_reference)image, name, sizeof(name));
			printf("ERROR: Image COMPARE MISMATCHED %s plane#%d " VX_FMT_SIZE "-pixel(s) with frame#%d of %s\n", name, plane, errorPixelCount, frameNumber, fileNameRef ? fileNameRef : "???");
		}
		// skip to begnning of next plane
		pRefPlane += plane_height * plane_width_in_bytes;
	}
	return errorPixelCountTotal;
}

// get image width in bytes from image
vx_size CalculateImageWidthInBytes(vx_image image)
{
	AgoImageFormatDescription format_description;
	vx_context context = vxGetContext((vx_reference)image);
	vx_df_image format = VX_DF_IMAGE_VIRT;
	vx_uint32 width;
	ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
	ERROR_CHECK(vxQueryImage(image, VX_IMAGE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK(vxGetContextImageFormatDescription(context, format, &format_description));

	return ((width * format_description.pixelSizeInBitsNum + format_description.pixelSizeInBitsDenom - 1) / format_description.pixelSizeInBitsDenom + 7) >> 3;
}

// read image
int ReadImage(vx_image image, vx_rectangle_t * rectFull, FILE * fp)
{
	// get number of planes, image width in bytes for single plane 
	vx_size num_planes = 0;
	ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_PLANES, &num_planes, sizeof(num_planes)));
	vx_size width_in_bytes = (num_planes == 1) ? CalculateImageWidthInBytes(image) : 0;
	// read all image planes into vx_image and check if EOF has occured while reading
	bool eofDetected = false;
	for (vx_uint32 plane = 0; plane < (vx_uint32)num_planes; plane++) {
		vx_imagepatch_addressing_t addr;
		vx_uint8 * src = NULL;
		ERROR_CHECK(vxAccessImagePatch(image, rectFull, plane, &addr, (void **)&src, VX_WRITE_ONLY));
		vx_size width = (addr.dim_x * addr.scale_x) / VX_SCALE_UNITY;
		if (addr.stride_x != 0)
			width_in_bytes = (width * addr.stride_x);
		for (vx_uint32 y = 0; y < addr.dim_y; y += addr.step_y){
			vx_uint8 *srcp = (vx_uint8 *)vxFormatImagePatchAddress2d(src, 0, y, &addr);
			if (fread(srcp, 1, width_in_bytes, fp) != width_in_bytes) {
				eofDetected = true;
				break;
			}
		}
		ERROR_CHECK(vxCommitImagePatch(image, rectFull, plane, &addr, src));
	}
	// return 1 if EOF detected, other 0
	return eofDetected ? 1 : 0;
}

// write image
int WriteImage(vx_image image, vx_rectangle_t * rectFull, FILE * fp)
{
	// get number of planes, image width in bytes for single plane 
	vx_size num_planes = 0;
	ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_PLANES, &num_planes, sizeof(num_planes)));
	vx_size width_in_bytes = (num_planes == 1) ? CalculateImageWidthInBytes(image) : 0;
	// write all image planes from vx_image
	bool eofDetected = false;
	for (vx_uint32 plane = 0; plane < (vx_uint32)num_planes; plane++) {
		vx_imagepatch_addressing_t addr;
		vx_uint8 * src = NULL;
		ERROR_CHECK(vxAccessImagePatch(image, rectFull, plane, &addr, (void **)&src, VX_READ_ONLY));
		vx_size width = (addr.dim_x * addr.scale_x) / VX_SCALE_UNITY;
		if (addr.stride_x != 0)
			width_in_bytes = (width * addr.stride_x);
		for (vx_uint32 y = 0; y < addr.dim_y; y += addr.step_y){
			vx_uint8 *srcp = (vx_uint8 *)vxFormatImagePatchAddress2d(src, 0, y, &addr);
			fwrite(srcp, 1, width_in_bytes, fp);
		}
		ERROR_CHECK(vxCommitImagePatch(image, rectFull, plane, &addr, src));
	}
	return 0;
}

// write image compressed
int WriteImageCompressed(vx_image image, vx_rectangle_t * rectFull, const char * fileName) 
{
    vx_size num_planes = 0;
    int im_width = 0, im_height = 0, im_format = 0;
    vx_df_image df_image;
    ERROR_CHECK(vxQueryImage(image, VX_IMAGE_WIDTH, &im_width, sizeof(im_width)));
    ERROR_CHECK(vxQueryImage(image, VX_IMAGE_HEIGHT, &im_height, sizeof(im_height)));
    ERROR_CHECK(vxQueryImage(image, VX_IMAGE_FORMAT, &df_image, sizeof(df_image)));
    ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_PLANES, &num_planes, sizeof(num_planes)));

    im_format = (int)df_image;

    // Compressed image formats (JPEG, PNG, BMP, …) represent the full image as a
    // single unit. Write plane 0 only — it carries the interleaved/packed pixel data
    // that the image plugin's encoder expects. Writing additional planes would
    // overwrite the output file with raw sub-plane data and corrupt the result.
    IRunVxImagePlugin * plugin = RunVxGetImagePlugin();
    if (!plugin) {
        printf("ERROR: this feature requires the runvx image plugin\n");
        throw - 1;
    }
    for (vx_uint32 plane = 0; plane < (vx_uint32)num_planes; plane++) {
        vx_imagepatch_addressing_t addr;
        vx_uint8 * src = NULL;
        ERROR_CHECK(vxAccessImagePatch(image, rectFull, plane, &addr, (void **)&src, VX_READ_ONLY));

        if (plane == 0) {
            if (!plugin->WriteCompressed(fileName, im_width, im_height, im_format, src, addr.stride_y)) {
                printf("ERROR: WriteCompressed(%s) failed\n", fileName);
                throw - 1;
            }
        }

        ERROR_CHECK(vxCommitImagePatch(image, rectFull, plane, &addr, src));
    }
    return 0;
}

// read scalar value into a string
int ReadScalarToString(vx_scalar scalar, char str[])
{
	vx_enum type; ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
	if (type == VX_TYPE_FLOAT32) {
		float v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%g", v);
	}
	else if (type == VX_TYPE_FLOAT64) {
		double v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%lg", v);
	}
	else if (type == VX_TYPE_SIZE) {
		vx_size v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, VX_FMT_SIZE, v);
	}
	else if (type == VX_TYPE_INT8 || type == VX_TYPE_CHAR) {
		vx_int8 v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%d", v);
	}
	else if (type == VX_TYPE_INT16) {
		vx_int16 v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%d", v);
	}
	else if (type == VX_TYPE_INT32 || type == VX_TYPE_BOOL) {
		vx_int32 v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%d", v);
	}
	else if (type == VX_TYPE_INT64) {
		vx_int64 v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%" PRId64, v);
	}
	else if (type == VX_TYPE_UINT8) {
		vx_uint8 v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%d", v);
	}
	else if (type == VX_TYPE_UINT16) {
		vx_uint16 v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%d", v);
	}
	else if (type == VX_TYPE_UINT32) {
		vx_uint32 v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%d", v);
	}
	else if (type == VX_TYPE_UINT64) {
		vx_uint64 v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%" PRIu64, v);
	}
	else if (type == VX_TYPE_ENUM) {
		vx_enum v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		const char * name = ovxEnum2Name(v);
		if (name) strcpy(str, name);
		else snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "0x%x", v);
	}
	else if (type == VX_TYPE_DF_IMAGE || type == VX_TYPE_STRING_AMD) {
		str[4] = 0; // needed for VX_TYPE_DF_IMAGE
		ERROR_CHECK(vxReadScalarValue(scalar, str));
	}
	else {
		// unknown types will be printed in hex
		vx_uint64 v = 0; ERROR_CHECK(vxReadScalarValue(scalar, &v));
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "0x%" PRIx64, v);
	}
	return 0;
}

// get scalar value from struct types.
int GetScalarValueForStructTypes(vx_enum type, const char str[], void * value)
{
	auto getNextToken = [](const char *& s, char * token, size_t size) -> const char * {
		size_t i = 0;
		for (size--; *s && *s != ',' && *s != '}'; s++) {
			if(i < size)
				token[i++] = *s;
		}
		if(*s == ',' || *s == '}')
			s++;
		token[i] = '\0';
		return token;
	};

	char token[1024];
	const char * s = &str[1];
	if(str[0] != '{') {
		printf("ERROR: GetScalarValueForStructTypes: string must start with '{'\n");
		return -1;
	}
	else if (type == VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS) {
		vx_tensor_matrix_multiply_params_t v;
		v.transpose_input1 = ovxName2Enum(getNextToken(s, token, sizeof(token))) ? vx_true_e : vx_false_e;
		v.transpose_input2 = ovxName2Enum(getNextToken(s, token, sizeof(token))) ? vx_true_e : vx_false_e;
		v.transpose_input3 = ovxName2Enum(getNextToken(s, token, sizeof(token))) ? vx_true_e : vx_false_e;
		*(vx_tensor_matrix_multiply_params_t *)value = v;
	}
	else if (type == VX_TYPE_NN_CONVOLUTION_PARAMS) {
		vx_nn_convolution_params_t v;
		v.padding_x = atoi(getNextToken(s, token, sizeof(token)));
		v.padding_y = atoi(getNextToken(s, token, sizeof(token)));
		v.overflow_policy = ovxName2Enum(getNextToken(s, token, sizeof(token)));
		v.rounding_policy = ovxName2Enum(getNextToken(s, token, sizeof(token)));
		v.down_scale_size_rounding = ovxName2Enum(getNextToken(s, token, sizeof(token)));
		v.dilation_x = atoi(getNextToken(s, token, sizeof(token)));
		v.dilation_y = atoi(getNextToken(s, token, sizeof(token)));
		*(vx_nn_convolution_params_t *)value = v;
	}
	else if (type == VX_TYPE_NN_DECONVOLUTION_PARAMS) {
		vx_nn_deconvolution_params_t v;
		v.padding_x = atoi(getNextToken(s, token, sizeof(token)));
		v.padding_y = atoi(getNextToken(s, token, sizeof(token)));
		v.overflow_policy = ovxName2Enum(getNextToken(s, token, sizeof(token)));
		v.rounding_policy = ovxName2Enum(getNextToken(s, token, sizeof(token)));
		v.a_x = atoi(getNextToken(s, token, sizeof(token)));
		v.a_y = atoi(getNextToken(s, token, sizeof(token)));
		*(vx_nn_deconvolution_params_t *)value = v;
	}
	else if (type == VX_TYPE_NN_ROI_POOL_PARAMS) {
		vx_nn_roi_pool_params_t v;
		v.pool_type = ovxName2Enum(getNextToken(s, token, sizeof(token)));
		*(vx_nn_roi_pool_params_t *)value = v;
	}
	else {
		printf("ERROR: GetScalarValueForStructTypes: unsupported type 0x%08x\n", type);
		return -1;
	}
	return 0;
}

// get scalar value from string
int GetScalarValueFromString(vx_enum type, const char str[], vx_uint64 * value)
{
	if (type == VX_TYPE_FLOAT32) {
		float v = 0; (void)sscanf(str, "%g", &v);
		*(float *)value = v;
	}
	else if (type == VX_TYPE_FLOAT64) {
		double v = 0; (void)sscanf(str, "%lg", &v);
		*(double *)value = v;
	}
	else if (type == VX_TYPE_FLOAT16) {
		float v = 0; (void)sscanf(str, "%g", &v);
		vx_uint32 f = *(vx_uint32 *)&v;
		*(vx_uint16 *)value = ((f >> 16) & 0x8000) | ((((f & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((f >> 13) & 0x03ff);
	}
	else if (type == VX_TYPE_SIZE) {
		vx_size v = 0; (void)sscanf(str, VX_FMT_SIZE, &v);
		*(vx_size *)value = v;
	}
	else if (type == VX_TYPE_INT8 || type == VX_TYPE_INT16 || type == VX_TYPE_INT32 ||
		type == VX_TYPE_UINT8 || type == VX_TYPE_UINT16 || type == VX_TYPE_UINT32 ||
		type == VX_TYPE_CHAR || type == VX_TYPE_BOOL || type == VX_TYPE_DF_IMAGE)
	{
		vx_int32 v = 0; (void)sscanf(str, "%i", &v);
		*(vx_int32 *)value = v;
	}
	else if (type == VX_TYPE_INT64 || type == VX_TYPE_UINT64) {
		vx_int64 v = 0; (void)sscanf(str, "%" PRIi64, &v);
		*(vx_int64 *)value = v;
	}
	else if (type == VX_TYPE_ENUM) {
		vx_enum v = ovxName2Enum(str);
		*(vx_enum *)value = v;
	}
	else {
		return -1;
	}
	return 0;
}

// write scalar value from a string
int WriteScalarFromString(vx_scalar scalar, const char str[])
{
	vx_enum type; ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
	if (type == VX_TYPE_FLOAT32) {
		float v = 0; (void)sscanf(str, "%g", &v);
		ERROR_CHECK(vxWriteScalarValue(scalar, &v));
	}
	else if (type == VX_TYPE_FLOAT64) {
		double v = 0; (void)sscanf(str, "%lg", &v);
		ERROR_CHECK(vxWriteScalarValue(scalar, &v));
	}
	else if (type == VX_TYPE_SIZE) {
		vx_size v = 0; (void)sscanf(str, VX_FMT_SIZE, &v);
		ERROR_CHECK(vxWriteScalarValue(scalar, &v));
	}
	else if (type == VX_TYPE_INT8 || type == VX_TYPE_INT16 || type == VX_TYPE_INT32 || 
		     type == VX_TYPE_UINT8 || type == VX_TYPE_UINT16 || type == VX_TYPE_UINT32 ||
			 type == VX_TYPE_CHAR || type == VX_TYPE_BOOL)
	{
		vx_int32 v = 0; (void)sscanf(str, "%i", &v);
		ERROR_CHECK(vxWriteScalarValue(scalar, &v));
	}
	else if (type == VX_TYPE_INT64 || type == VX_TYPE_UINT64) {
		vx_int64 v = 0; (void)sscanf(str, "%" PRIi64, &v);
		ERROR_CHECK(vxWriteScalarValue(scalar, &v));
	}
	else if (type == VX_TYPE_ENUM) {
		vx_enum v = ovxName2Enum(str);
		ERROR_CHECK(vxWriteScalarValue(scalar, &v));
	}
	else if (type == VX_TYPE_DF_IMAGE || type == VX_TYPE_STRING_AMD) {
		ERROR_CHECK(vxWriteScalarValue(scalar, str));
	}
	else {
		// unknown types will be assumed to be in hex format
		vx_int64 v = 0; (void)sscanf(str, "%" PRIi64, &v);
		ERROR_CHECK(vxWriteScalarValue(scalar, &v));
	}
	return 0;
}

// put scalar value to string
int PutScalarValueToString(vx_enum type, const void * value, char str[])
{
	if (type == VX_TYPE_FLOAT32) {
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%g", *(float *)value);
	}
	else if (type == VX_TYPE_FLOAT64) {
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%lg", *(double *)value);
	}
	else if (type == VX_TYPE_SIZE) {
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, VX_FMT_SIZE, *(vx_size *)value);
	}
	else if (type == VX_TYPE_INT8 || type == VX_TYPE_CHAR) {
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%d", *(vx_int8 *)value);
	}
	else if (type == VX_TYPE_INT16) {
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%d", *(vx_int16 *)value);
	}
	else if (type == VX_TYPE_INT32 || type == VX_TYPE_BOOL) {
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%d", *(vx_int32 *)value);
	}
	else if (type == VX_TYPE_INT64) {
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%" PRId64, *(vx_int64 *)value);
	}
	else if (type == VX_TYPE_UINT8) {
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%u", *(vx_uint8 *)value);
	}
	else if (type == VX_TYPE_UINT16) {
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%u", *(vx_uint16 *)value);
	}
	else if (type == VX_TYPE_UINT32) {
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%u", *(vx_uint32 *)value);
	}
	else if (type == VX_TYPE_UINT64) {
		snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "%" PRIu64, *(vx_uint64 *)value);
	}
	else if (type == VX_TYPE_ENUM) {
		vx_enum v = *(vx_enum *)value;
		const char * name = ovxEnum2Name(v);
		if (name) strcpy(str, name);
		else snprintf(str, MAX_SCALAR_TO_STRING_SIZE, "0x%x", v);
	}
	else if (type == VX_TYPE_DF_IMAGE || type == VX_TYPE_STRING_AMD) {
		if (type == VX_TYPE_DF_IMAGE) {
			str[4] = 0; strncpy(str, (const char *)value, 4);
		}
		else strcpy(str, (const char *)value);
	}
	else {
		return -1;
	}
	return 0;
}

