/*
Copyright (c) 2017 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include "flat/flat_parser.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <set>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

////
// MAGIC numbers
//
#define VARIABLES_FILE_MAGIC 0xF00DD1E0
#define VARIABLES_DATA_MAGIC 0xF00DD1E1
#define VARIABLES_EOFF_MAGIC 0xF00DD1E2

////
// NNEF to OpenVX Translator
//
class NNEF2OpenVX_Translator : public nnef::Parser::Callback
{
public:
    NNEF2OpenVX_Translator(std::string nnefFolder_, std::string openvxFolder_, bool useVirtual_, int verbose_)
      : nnefFolder(nnefFolder_), openvxFolder(openvxFolder_), useVirtual(useVirtual_), verbose(verbose_)
    {
    }

protected:
    ////
    // class variables
    //
    int verbose;
    bool useVirtual;
    std::string nnefFolder;
    std::string openvxFolder;
    std::string openvxFilenameC;
    std::ofstream ovxC;
    std::vector<std::string> inputList;
    std::vector<std::string> outputList;
    std::vector<std::string> virtualList;
    std::vector<std::string> variableList;
    std::map<std::string,std::tuple<size_t, char *>> variableBinary;
    std::map<std::string,nnef::Shape> inputShape;
    std::map<std::string,nnef::Shape> outputShape;
    std::map<std::string,nnef::Shape> virtualShape;
    std::map<std::string,nnef::Shape> variableShape;
    std::map<std::string,std::string> variableLabel;
    std::map<std::string,size_t> variableRequiredDims;
    std::vector<nnef::Prototype> opsProto;
    std::vector<nnef::Dictionary<nnef::Value>> opsValues;
    std::vector<nnef::Dictionary<nnef::Shape>> opsShapes;
    std::vector<bool> operationRemoved;
    std::map<std::string,bool> variableMerged;
    std::map<std::string,std::string> virtualRename;
    std::map<std::string,std::string> convNewBiasName;

private:
    // utility functions
    static void getTensorDims(const nnef::Shape& shape, std::vector<size_t>& dims, size_t num_dims)
    {
        size_t rank = shape.rank();
        if(num_dims == 0)
            num_dims = rank;
        dims.clear();
        size_t count = 0;
        if(rank > 1) {
            for(; count < (num_dims - rank); count++) {
                dims.push_back(1);
            }
        }
        for(size_t i = 0; i < rank; i++, count++) {
            dims.push_back(shape[rank-1-i]);
        }
        for(; count < num_dims; count++) {
            dims.push_back(1);
        }
    }
    static std::string codeGenTensorCreate (const std::string& name, const nnef::Shape& shape, bool useVirtual, size_t num_dims)
    {
        std::stringstream ss;
        std::vector<size_t> dims;
        getTensorDims(shape, dims, num_dims);
        ss << "    vx_size " << name << "_dims[" << dims.size() << "] = {";
        for(size_t i = 0; i < dims.size(); i++) {
            ss << (i == 0 ? " " : ", ") << dims[i];
        }
        ss << " };" << std::endl;
        ss << "    vx_tensor " << name << " = "
           << (useVirtual ? "vxCreateVirtualTensor(graph, " : "vxCreateTensor(context, ")
           << dims.size() << ", " << name << "_dims, VX_TYPE_FLOAT32, 0);" << std::endl;
        ss << "    ERROR_CHECK_OBJECT(" << name << ");" << std::endl;
        return ss.str();
    }
    static unsigned int loadTensorFile(const std::string& nnefFolder, const std::string& label, const nnef::Shape& shape, char *& data)
    {
        std::string fileName = nnefFolder + "/" + label + ".dat";
        FILE * fp = fopen(fileName.c_str(), "rb");
        if(!fp) {
            printf("ERROR: unable to open: %s\n", fileName.c_str());
            exit(1);
        }
        enum TensorDataType : unsigned char {
            TensorDataType_Float,
            TensorDataType_Quantized,
            TensorDataType_Signed,
            TensorDataType_Unsigned
        };
        struct TensorFileHeader {
            unsigned char  magic[2];
            unsigned char  major;
            unsigned char  minor;
            unsigned int   offset;
            unsigned int   rank;
            unsigned int   dim[8];
            unsigned char  data_type;
            unsigned char  bit_width;
            unsigned short quant_alg_len;
            char           quant_alg[1024];
        } h = { 0 };
        unsigned int offset = 0;
        offset += fread(&h.magic, 1, sizeof(h.magic), fp);
        offset += fread(&h.major, 1, sizeof(h.major), fp);
        offset += fread(&h.minor, 1, sizeof(h.minor), fp);
        offset += fread(&h.offset, 1, sizeof(h.offset), fp);
        offset += fread(&h.rank, 1, sizeof(h.rank), fp);
        if(h.rank > 0) {
            offset += fread(h.dim, 1, h.rank * sizeof(h.dim[0]), fp);
        }
        offset += fread(&h.data_type, 1, sizeof(h.data_type), fp);
        offset += fread(&h.bit_width, 1, sizeof(h.bit_width), fp);
        offset += fread(&h.quant_alg_len, 1, sizeof(h.quant_alg_len), fp);
        if(h.quant_alg_len > 0) {
            offset += fread(h.quant_alg, 1, h.quant_alg_len, fp);
        }
        if(h.magic[0] != 0x4e || h.magic[1] != 0xef || h.major != 1 || h.minor != 0
                              || h.bit_width == 0 || h.rank > 8 || h.quant_alg_len >= 1024
                              || (12 + h.rank * 4 + 4 + h.quant_alg_len) != offset || h.offset < offset)
        {
            printf("ERROR: invalid or unsupported tensor file: %s\n", fileName.c_str());
            printf(" [ 0x%02x, 0x%02x, %d, %d, %d, %d, {", h.magic[0], h.magic[1], h.major, h.minor, h.offset, h.rank);
            for(unsigned int i = 0; i < h.rank; i++) printf(" %d", h.dim[i]);
            printf(" }, %d, %d, %d, '%s' ] offset = %d\n", h.data_type, h.bit_width, h.quant_alg_len, h.quant_alg, offset);
            exit(1);
        }
        if(h.offset > offset) {
            fseek(fp, h.offset, SEEK_SET);
        }
        unsigned int size = h.bit_width;
        for(unsigned int i = 0; i < h.rank; i++) {
            size *= h.dim[i];
            if(h.dim[i] != shape[i]) {
                printf("ERROR: dimension[%d] mismatch: %d in %s (must be %d)\n", i, h.dim[i], fileName.c_str(), shape[i]);
                exit(1);
            }
        }
        size = (size + 7) >> 3;
        data = nullptr;
        if(h.data_type == TensorDataType_Float && h.bit_width == 32) {
            data = new char [size];
            if(!data) {
                printf("ERROR: memory allocation for %d bytes failed for %s\n", size, fileName.c_str());
                exit(1);
            }
            unsigned int n = fread(data, 1, size, fp);
            if(n != size) {
                printf("ERROR: unable to read %d bytes of data from %s\n", size, fileName.c_str());
                exit(1);
            }
        }
        else {
            printf("ERROR: import of Tensor DataType=%d BitWidth=%d is not yet supported\n", h.data_type, h.bit_width);
            exit(1);
        }
        fclose(fp);
        return size;
    }

    std::string virtualName(const std::string name)
    {
        auto it = virtualRename.find(name);
        return (it != virtualRename.end()) ? it->second : name;
    }

    void codeGenOperation(size_t pos, bool getVariables, bool genCode, int verbose)
    {
        ////
        // make sure that operation is not disabled
        //
        if(operationRemoved[pos]) {
            return;
        }

        ////
        // get operation details
        //
        const nnef::Prototype& proto = opsProto[pos];
        const nnef::Dictionary<nnef::Value>& args = opsValues[pos];
        const nnef::Dictionary<nnef::Shape>& shapes = opsShapes[pos];
        if(verbose & 1) {
            std::cout << '\t';
            for ( size_t i = 0; i < proto.resultCount(); ++i ) {
                auto& result = proto.result(i);
                if ( i ) std::cout << ", ";
                std::cout << args[result.name()];
            }
            std::cout << " = " << proto.name() << "(";
            for ( size_t i = 0; i < proto.paramCount(); ++i ) {
                auto& param = proto.param(i);
                if ( i ) std::cout << ", ";
                if ( !param.type()->isTensor() )
                    std::cout << param.name() << " = ";
                std::cout << args[param.name()];
            }
            std::cout << ")" << std::endl;
        }

        ////
        // utility functions
        //
        auto getTensorOrScalar = [] (const nnef::Value& v) -> std::string {
            std::string value = "0";
            if(v) {
                if(v.kind() == nnef::Value::Tensor) {
                    value = v.tensor().id;
                }
                else if(v.kind() == nnef::Value::Scalar) {
                    value = std::to_string(v.scalar());
                }
            }
            return value;
        };
        auto getExtentArray = [] (const nnef::Value& v) -> std::vector<size_t> {
            std::vector<size_t> value;
            if(v && v.kind() == nnef::Value::Array) {
                auto&& a = v.array();
                for(auto& i : a) {
                    value.push_back(i.integer());
                }
            }
            return value;
        };
        auto getPaddingInfo = [] (const nnef::Value& v, size_t pad[4]) {
            std::vector<size_t> value;
            if(v && v.kind() == nnef::Value::Array) {
                auto&& a = v.array();
                if(a.size() == 2) {
                    pad[0] = a[0][0].integer();
                    pad[1] = a[0][1].integer();
                    pad[2] = a[1][0].integer();
                    pad[3] = a[1][1].integer();
                    // TODO: protection against -ve values
                    if(pad[0] > 16384) pad[0] = 0;
                    if(pad[1] > 16384) pad[1] = 0;
                    if(pad[2] > 16384) pad[2] = 0;
                    if(pad[3] > 16384) pad[3] = 0;
                }
            }
        };

        ////
        // process operations
        //
        std::string opname = proto.name();
        if(opname == "external") {
            const std::string& output = args["output"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << std::endl;
            }
            if(getVariables) {
                inputShape[output] = shape;
            }
        }
        else if(opname == "variable") {
            const std::string& output = args["output"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            const std::string& label = args["label"].string();
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " label=" << label << std::endl;
            }
            if(getVariables) {
                variableList.push_back(output);
                variableMerged[output] = false;
                variableShape[output] = shape;
                variableLabel[output] = label;
            }
        }
        else if(opname == "conv") {
            const std::string& output = args["output"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            const std::string& input = args["input"].tensor().id;
            const std::string& filter = args["filter"].tensor().id;
                  std::string  bias = getTensorOrScalar(args["bias"]);
            const std::string& border = args["border"].string();
            const auto& padding = args["padding"];
            const auto& stride = args["stride"];
            const auto& dilation = args["dilation"];
            const auto& groups = args["groups"] ? args["groups"].integer() : 1;
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " " << input << " " << filter << " " << bias
                          << " border=" << border << " " << padding << " " << stride << " " << dilation << " " << groups << std::endl;
            }
            if(getVariables) {
                if(std::find(outputList.begin(), outputList.end(), output) == outputList.end()) {
                    virtualList.push_back(output);
                    virtualShape[output] = shape;
                }
                else {
                    outputShape[output] = shape;
                }
                variableRequiredDims[filter] = 4;
                if(bias[0] != '0') {
                    variableRequiredDims[bias] = 2;
                }
            }
            if(genCode) {
                if(std::find(virtualList.begin(), virtualList.end(), output) != virtualList.end()) {
                    ovxC << codeGenTensorCreate(output, shape, useVirtual, 4);
                }
                if(bias[0] == '0') {
                    if(convNewBiasName.find(output) != convNewBiasName.end()) {
                        bias = convNewBiasName.find(output)->second;
                    }
                }
                if(shape[2] == 1 && shape[3] == 1) {
                    ovxC << "    { vx_node node = vxFullyConnectedLayer(graph, " << virtualName(input) << ", " << filter << ", "
                         << ((bias[0] == '0') ? "NULL" : bias) << ", VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN, " << output << ");" << std::endl;
                    ovxC << "      ERROR_CHECK_STATUS(vxReleaseNode(&node));" << std::endl;
                    ovxC << "    }" << std::endl;
                }
                else {
                    std::vector<size_t>&& vDilation = getExtentArray(dilation);
                    size_t pad[4] = { 0, 0, 0, 0 };
                    getPaddingInfo(padding, pad);
                    ovxC << "    { vx_nn_convolution_params_t conv_params = { 0 };" << std::endl;
                    ovxC << "      conv_params.padding_x = " << pad[1] << ";" << std::endl;
                    ovxC << "      conv_params.padding_y = " << pad[0] << ";" << std::endl;
                    ovxC << "      conv_params.dilation_x = " << (vDilation.size() > 1 ? vDilation[1] - 1 : 0) << ";" << std::endl;
                    ovxC << "      conv_params.dilation_y = " << (vDilation.size() > 0 ? vDilation[0] - 1 : 0) << ";" << std::endl;
                    ovxC << "      conv_params.overflow_policy = " << "VX_CONVERT_POLICY_SATURATE" << ";" << std::endl;
                    ovxC << "      conv_params.rounding_policy = " << "VX_ROUND_POLICY_TO_NEAREST_EVEN" << ";" << std::endl;
                    ovxC << "      conv_params.down_scale_size_rounding = " << "VX_NN_DS_SIZE_ROUNDING_FLOOR" << ";" << std::endl;
                    ovxC << "      vx_node node = vxConvolutionLayer(graph, " << virtualName(input) << ", " << filter << ", "
                         << ((bias[0] == '0') ? "NULL" : bias) << ", &conv_params, sizeof(conv_params), " << output << ");" << std::endl;
                    ovxC << "      ERROR_CHECK_STATUS(vxReleaseNode(&node));" << std::endl;
                    ovxC << "    }" << std::endl;
                }
            }
        }
        else if(opname == "relu") {
            const std::string& output = args["y"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            const std::string& input = args["x"].tensor().id;
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " " << input << std::endl;
            }
            if(getVariables) {
                if(std::find(outputList.begin(), outputList.end(), output) == outputList.end()) {
                    virtualList.push_back(output);
                    virtualShape[output] = shape;
                }
                else {
                    outputShape[output] = shape;
                }
            }
            if(genCode) {
                if(std::find(virtualList.begin(), virtualList.end(), output) != virtualList.end()) {
                    ovxC << codeGenTensorCreate(output, shape, useVirtual, 4);
                }
                ovxC << "    { vx_node node = vxActivationLayer(graph, " << virtualName(input) << ", VX_NN_ACTIVATION_RELU, 0.0f, 0.0f, " << output << ");" << std::endl;
                ovxC << "      ERROR_CHECK_STATUS(vxReleaseNode(&node));" << std::endl;
                ovxC << "    }" << std::endl;
            }
        }
        else if(opname == "max_pool") {
            const std::string& output = args["output"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            const std::string& input = args["input"].tensor().id;
            const auto& size = args["size"];
            const std::string& border = args["border"].string();
            const auto& padding = args["padding"];
            const auto& stride = args["stride"];
            const auto& dilation = args["dilation"];
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " " << input
                          << " size=" << size << " border=" << border << " " << padding << " " << stride << " " << dilation << std::endl;
            }
            if(getVariables) {
                if(std::find(outputList.begin(), outputList.end(), output) == outputList.end()) {
                    virtualList.push_back(output);
                    virtualShape[output] = shape;
                }
                else {
                    outputShape[output] = shape;
                }
            }
            if(genCode) {
                if(std::find(virtualList.begin(), virtualList.end(), output) != virtualList.end()) {
                    ovxC << codeGenTensorCreate(output, shape, useVirtual, 4);
                }
                std::vector<size_t>&& vSize = getExtentArray(size);
                size_t pad[4] = { 0, 0, 0, 0 };
                getPaddingInfo(padding, pad);
                ovxC << "    { vx_node node = vxPoolingLayer(graph, " << virtualName(input) << ", VX_NN_POOLING_MAX, "
                     << size[3] << ", " << size[2] << ", " << pad[1] << ", " << pad[0] << ", "
                     << "VX_ROUND_POLICY_TO_NEAREST_EVEN, " << output << ");" << std::endl;
                ovxC << "      ERROR_CHECK_STATUS(vxReleaseNode(&node));" << std::endl;
                ovxC << "    }" << std::endl;
            }
        }
        else if(opname == "avg_pool") {
            const std::string& output = args["output"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            const std::string& input = args["input"].tensor().id;
            const auto& size = args["size"];
            const std::string& border = args["border"].string();
            const auto& padding = args["padding"];
            const auto& stride = args["stride"];
            const auto& dilation = args["dilation"];
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " " << input
                          << " size=" << size << " border=" << border << " " << padding << " " << stride << " " << dilation << std::endl;
            }
            if(getVariables) {
                if(std::find(outputList.begin(), outputList.end(), output) == outputList.end()) {
                    virtualList.push_back(output);
                    virtualShape[output] = shape;
                }
                else {
                    outputShape[output] = shape;
                }
            }
            if(genCode) {
                if(std::find(virtualList.begin(), virtualList.end(), output) != virtualList.end()) {
                    ovxC << codeGenTensorCreate(output, shape, useVirtual, 4);
                }
                std::vector<size_t>&& vSize = getExtentArray(size);
                size_t pad[4] = { 0, 0, 0, 0 };
                getPaddingInfo(padding, pad);
                ovxC << "    { vx_node node = vxPoolingLayer(graph, " << virtualName(input) << ", VX_NN_POOLING_AVG, "
                     << size[3] << ", " << size[2] << ", " << pad[1] << ", " << pad[0] << ", "
                     << "VX_ROUND_POLICY_TO_NEAREST_EVEN, " << output << ");" << std::endl;
                ovxC << "      ERROR_CHECK_STATUS(vxReleaseNode(&node));" << std::endl;
                ovxC << "    }" << std::endl;
            }
        }
        else if(opname == "concat") {
            const std::string& output = args["value"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            std::vector<std::string> itemList;
            const auto& inputpar = args["values"];
            for(size_t i = 0; i < inputpar.size(); i++) {
                std::string name = inputpar[i].tensor().id;
                itemList.push_back(name);
            }
            const int axis = args["axis"].integer();
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " [";
                for(auto& v : itemList) std::cout << " " << v;
                std::cout << " ] axis=" << axis << std::endl;
            }
            if(getVariables) {
                if(std::find(outputList.begin(), outputList.end(), output) == outputList.end()) {
                    virtualList.push_back(output);
                    virtualShape[output] = shape;
                }
                else {
                    outputShape[output] = shape;
                }
            }
            if(genCode) {
                if(std::find(virtualList.begin(), virtualList.end(), output) != virtualList.end()) {
                    ovxC << codeGenTensorCreate(output, shape, useVirtual, 4);
                }
                ovxC << "    { vx_node node = vxConcatLayer(graph, " << output;
                for(auto& v : itemList) {
                    ovxC << ", " << virtualName(v);
                }
                for(size_t i = itemList.size(); i < 8; i++) {
                    ovxC << ", NULL";
                }
                ovxC << ");" << std::endl;
                ovxC << "      ERROR_CHECK_STATUS(vxReleaseNode(&node));" << std::endl;
                ovxC << "    }" << std::endl;
            }
        }
        else if(opname == "batch_normalization") {
            const std::string& output = args["output"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            const std::string& input = args["input"].tensor().id;
            const std::string& mean = args["mean"].tensor().id;
            const std::string& variance = args["variance"].tensor().id;
            std::string scale = getTensorOrScalar(args["scale"]);
            std::string offset = getTensorOrScalar(args["offset"]);
            const float epsilon = args["epsilon"].scalar();
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " " << input
                          << " " << mean << " " << variance << " " << offset << " " << scale << " " << epsilon << std::endl;
            }
            if(getVariables) {
                if(std::find(outputList.begin(), outputList.end(), output) == outputList.end()) {
                    virtualList.push_back(output);
                    virtualShape[output] = shape;
                }
                else {
                    outputShape[output] = shape;
                }
            }
            if(genCode) {
                if(std::find(virtualList.begin(), virtualList.end(), output) != virtualList.end()) {
                    ovxC << codeGenTensorCreate(output, shape, useVirtual, 4);
                }
                ovxC << "    { vx_node node = vxBatchNormalizationLayer(graph, " << virtualName(input) << ", " << mean << ", " << variance
                     << ", " << (scale[0] == '1' ? "NULL" : scale) << ", " << (offset[0] == '0' ? "NULL" : offset)
                     << ", " << epsilon << ", " << output << ");" << std::endl;
                ovxC << "      ERROR_CHECK_STATUS(vxReleaseNode(&node));" << std::endl;
                ovxC << "    }" << std::endl;
            }
        }
        else if(opname == "mul") {
            const std::string& output = args["z"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            const std::string& input1 = args["x"].tensor().id;
            const std::string& input2 = args["y"].tensor().id;
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " " << input1 << " " << input2 << std::endl;
            }
            if(getVariables) {
                if(std::find(outputList.begin(), outputList.end(), output) == outputList.end()) {
                    virtualList.push_back(output);
                    virtualShape[output] = shape;
                }
                else {
                    outputShape[output] = shape;
                }
            }
            if(genCode) {
                if(std::find(virtualList.begin(), virtualList.end(), output) != virtualList.end()) {
                    ovxC << codeGenTensorCreate(output, shape, useVirtual, 4);
                }
                ovxC << "    { float one = 1.0f;" << std::endl;
                ovxC << "      vx_scalar scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &one);" << std::endl;
                ovxC << "      vx_node node = vxTensorMultiplyNode(graph, " << virtualName(input1) << ", " << virtualName(input2) << ", scale, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN, " << output << ");" << std::endl;
                ovxC << "      ERROR_CHECK_STATUS(vxReleaseScalar(&scale));" << std::endl;
                ovxC << "      ERROR_CHECK_STATUS(vxReleaseNode(&node));" << std::endl;
                ovxC << "    }" << std::endl;
            }
        }
        else if(opname == "add") {
            const std::string& output = args["z"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            const std::string& input1 = args["x"].tensor().id;
            const std::string& input2 = args["y"].tensor().id;
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " " << input1 << " " << input2 << std::endl;
            }
            if(getVariables) {
                if(std::find(outputList.begin(), outputList.end(), output) == outputList.end()) {
                    virtualList.push_back(output);
                    virtualShape[output] = shape;
                }
                else {
                    outputShape[output] = shape;
                }
            }
            if(genCode) {
                if(std::find(virtualList.begin(), virtualList.end(), output) != virtualList.end()) {
                    ovxC << codeGenTensorCreate(output, shape, useVirtual, 4);
                }
                ovxC << "    { vx_node node = vxTensorAddNode(graph, " << virtualName(input1) << ", " << virtualName(input2) << ", VX_CONVERT_POLICY_SATURATE, " << output << ");" << std::endl;
                ovxC << "      ERROR_CHECK_STATUS(vxReleaseNode(&node));" << std::endl;
                ovxC << "    }" << std::endl;
            }
        }
        else if(opname == "softmax") {
            const std::string& output = args["y"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            const std::string& input = args["x"].tensor().id;
            std::vector<size_t>&& axes = getExtentArray(args["axes"]);
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " " << input << " " << args["axes"] << std::endl;
            }
            if(axes.size() != 1 || axes[0] != 1) {
                std::cout << "ERROR: " << opname << " with " << args["axes"] << " is *** not yet supported ***" << std::endl;
                exit(1);
            }
            if(getVariables) {
                if(std::find(outputList.begin(), outputList.end(), output) == outputList.end()) {
                    virtualList.push_back(output);
                    virtualShape[output] = shape;
                }
                else {
                    outputShape[output] = shape;
                }
            }
            if(genCode) {
                if(std::find(virtualList.begin(), virtualList.end(), output) != virtualList.end()) {
                    ovxC << codeGenTensorCreate(output, shape, useVirtual, 4);
                }
                ovxC << "    { vx_node node = vxSoftmaxLayer(graph, " << virtualName(input) << ", " << output << ");" << std::endl;
                ovxC << "      ERROR_CHECK_STATUS(vxReleaseNode(&node));" << std::endl;
                ovxC << "    }" << std::endl;
            }
        }
        else if(opname == "sum_reduce") {
            const std::string& output = args["output"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            const std::string& input = args["input"].tensor().id;
            const auto& axes = args["axes"];
            const bool normalize = args["normalize"].logical();
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " " << input << " " << axes << " " << normalize << std::endl;
            }
            if(getVariables) {
                if(std::find(outputList.begin(), outputList.end(), output) == outputList.end()) {
                    virtualList.push_back(output);
                    virtualShape[output] = shape;
                }
                else {
                    outputShape[output] = shape;
                }
            }
            if(genCode) {
                if(std::find(virtualList.begin(), virtualList.end(), output) != virtualList.end()) {
                    ovxC << codeGenTensorCreate(output, shape, useVirtual, 4);
                }
                std::cout << opname << " *** not yet supported ***" << std::endl;
                exit(1);
            }
        }
        else if(opname == "mean_reduce") {
            const std::string& output = args["output"].tensor().id;
            const nnef::Shape& shape = shapes[output];
            const std::string& input = args["input"].tensor().id;
            const auto& axes = args["axes"];
            if(verbose & 2) {
                std::cout << opname << " " << output << " " << shape << " " << input << " " << axes << std::endl;
            }
            if(getVariables) {
                if(std::find(outputList.begin(), outputList.end(), output) == outputList.end()) {
                    virtualList.push_back(output);
                    virtualShape[output] = shape;
                }
                else {
                    outputShape[output] = shape;
                }
            }
            if(genCode) {
                if(std::find(virtualList.begin(), virtualList.end(), output) != virtualList.end()) {
                    ovxC << codeGenTensorCreate(output, shape, useVirtual, 4);
                }
                std::cout << opname << " *** not yet supported ***" << std::endl;
                exit(1);
            }
        }
        else {
            std::cout << opname << " *** not yet supported ***" << std::endl;
            exit(1);
        }
    }

    void codeGenMergeVariables()
    {
        auto getTensorOrScalar = [] (const nnef::Value& v) -> std::string {
            std::string value = "0";
            if(v) {
                if(v.kind() == nnef::Value::Tensor) {
                    value = v.tensor().id;
                }
                else if(v.kind() == nnef::Value::Scalar) {
                    value = std::to_string(v.scalar());
                }
            }
            return value;
        };

        size_t prevPos = 0;
        std::string prevOpName = "", prevOutput = "";
        for(size_t pos = 0; pos < opsProto.size(); pos++) {
            std::string opname = opsProto[pos].name();
            if(prevOpName == "batch_normalization" && opname == "conv") {
                // get "batch_normalization" variables
                const nnef::Dictionary<nnef::Value>& argsBN = opsValues[prevPos];
                const nnef::Dictionary<nnef::Shape>& shapesBN = opsShapes[prevPos];
                const std::string& inputBN = argsBN["input"].tensor().id;
                const std::string& mean = argsBN["mean"].tensor().id;
                const std::string& variance = argsBN["variance"].tensor().id;
                std::string scale = getTensorOrScalar(argsBN["scale"]);
                std::string offset = getTensorOrScalar(argsBN["offset"]);
                const float epsilon = argsBN["epsilon"].scalar();
                const nnef::Shape& shapeMean = shapesBN[mean];
                // get "conv" variables
                const nnef::Dictionary<nnef::Value>& argsConv = opsValues[pos];
                const nnef::Dictionary<nnef::Shape>& shapesConv = opsShapes[pos];
                const std::string& outputConv = argsConv["output"].tensor().id;
                const std::string& filter = argsConv["filter"].tensor().id;
                const std::string& bias = getTensorOrScalar(argsConv["bias"]);
                const nnef::Shape& shapeFilter = shapesConv[filter];
                // get filter and mean dimensions
                size_t filterDimsCount = shapeFilter.rank(), meanDimsCount = shapeMean.rank();
                std::vector<size_t> filterDims, meanDims;
                getTensorDims(shapeFilter, filterDims, filterDimsCount);
                getTensorDims(shapeMean, meanDims, meanDimsCount);
                // check validity of dimensions
                size_t K = (filterDimsCount == 4) ? filterDims[3] : filterDims[1];
                size_t N = (filterDimsCount == 4) ? (filterDims[0] * filterDims[1] * filterDims[2]) : filterDims[0];
                if((filterDimsCount == 4 || filterDimsCount == 2) && meanDimsCount == 2 && K == meanDims[0]) {
                    // fuse batch_normalization variables into conv variables
                    std::tuple<unsigned int, char *> filterBinary = variableBinary[filter];
                    std::tuple<unsigned int, char *> meanBinary = variableBinary[mean];
                    std::tuple<unsigned int, char *> varianceBinary = variableBinary[variance];
                    float * filterBuf = (float *)std::get<1>(filterBinary);
                    float * biasBuf = nullptr;
                    float * meanBuf = (float *)std::get<1>(meanBinary);
                    float * varianceBuf = (float *)std::get<1>(varianceBinary);
                    float * scaleBuf = nullptr;
                    float * offsetBuf = nullptr;
                    if(bias[0] != '0') {
                        std::tuple<unsigned int, char *> biasBinary = variableBinary[bias];
                        biasBuf = (float *)std::get<1>(biasBinary);
                    }
                    else if(convNewBiasName.find(outputConv) != convNewBiasName.end()) {
                        std::tuple<unsigned int, char *> biasBinary = variableBinary[convNewBiasName[outputConv]];
                        biasBuf = (float *)std::get<1>(biasBinary);
                    }
                    else {
                        size_t size = K * sizeof(float);
                        char * data = new char [size];
                        biasBuf = (float *)data;
                        for(size_t i = 0; i < K; i++) {
                            biasBuf[i] = 0;
                        }
                        std::string name = filter + "__new_bias";
                        std::tuple<unsigned int, char *> binary(size, data);
                        variableBinary[name] = binary;
                        convNewBiasName[outputConv] = name;
                        variableList.push_back(name);
                        variableMerged[name] = false;
                        nnef::Shape shape(1);
                        shape[0] = K;
                        shape[1] = 1;
                        variableShape[name] = shape;
                        variableRequiredDims[name] = 2;
                    }
                    if(scale[0] != '1') {
                        scaleBuf = (float *)std::get<1>(variableBinary[scale]);
                    }
                    if(offset[0] != '0') {
                        offsetBuf = (float *)std::get<1>(variableBinary[offset]);
                    }
                    for(size_t k = 0; k < K; k++) {
                        double mk = 1.0 / sqrt((double)varianceBuf[k] + epsilon);
                        double ck = -meanBuf[k] * mk;
                        if(scaleBuf) {
                            mk *= scaleBuf[k];
                            ck *= scaleBuf[k];
                        }
                        if(offsetBuf) {
                            ck += offsetBuf[k];
                        }
                        float * W = &filterBuf[k*N];
                        double Wsum = 0;
                        for(size_t j = 0; j < N; j++) {
                            Wsum += W[j];
                            W[j] = (float)(W[j] * mk);
                        }
                        if(biasBuf) {
                            biasBuf[k] = (float)(Wsum * ck + biasBuf[k]);
                        }
                    }
                    // mark that batch_normalization is disabled and rename output as input
                    operationRemoved[prevPos] = true;
                    virtualRename[argsConv["input"].tensor().id] = inputBN;
                    // mark the merged variables
                    variableMerged[mean] = true;
                    variableMerged[variance] = true;
                    if(scaleBuf) variableMerged[scale] = true;
                    if(offsetBuf) variableMerged[offset] = true;
                }
                // use conv as previous layer
                prevPos = pos;
                prevOpName = opname;
                prevOutput = argsConv["output"].tensor().id;
            }
            else if(prevOpName == "conv" && opname == "batch_normalization") {
                // get "conv" variables
                const nnef::Dictionary<nnef::Value>& argsConv = opsValues[prevPos];
                const nnef::Dictionary<nnef::Shape>& shapesConv = opsShapes[prevPos];
                const std::string& outputConv = argsConv["output"].tensor().id;
                const std::string& filter = argsConv["filter"].tensor().id;
                const std::string& bias = getTensorOrScalar(argsConv["bias"]);
                const nnef::Shape& shapeFilter = shapesConv[filter];
                // get "batch_normalization" variables
                const nnef::Dictionary<nnef::Value>& argsBN = opsValues[pos];
                const nnef::Dictionary<nnef::Shape>& shapesBN = opsShapes[pos];
                const std::string& mean = argsBN["mean"].tensor().id;
                const std::string& variance = argsBN["variance"].tensor().id;
                std::string scale = getTensorOrScalar(argsBN["scale"]);
                std::string offset = getTensorOrScalar(argsBN["offset"]);
                const float epsilon = argsBN["epsilon"].scalar();
                const nnef::Shape& shapeMean = shapesBN[mean];
                // get filter and mean dimensions
                size_t filterDimsCount = shapeFilter.rank(), meanDimsCount = shapeMean.rank();
                std::vector<size_t> filterDims, meanDims;
                getTensorDims(shapeFilter, filterDims, filterDimsCount);
                getTensorDims(shapeMean, meanDims, meanDimsCount);
                // check validity of dimensions
                size_t K = (filterDimsCount == 4) ? filterDims[3] : filterDims[1];
                size_t N = (filterDimsCount == 4) ? (filterDims[0] * filterDims[1] * filterDims[2]) : filterDims[0];
                if((filterDimsCount == 4 || filterDimsCount == 2) && meanDimsCount == 2 && K == meanDims[0]) {
                    // fuse batch_normalization variables into conv variables
                    std::tuple<unsigned int, char *> filterBinary = variableBinary[filter];
                    std::tuple<unsigned int, char *> meanBinary = variableBinary[mean];
                    std::tuple<unsigned int, char *> varianceBinary = variableBinary[variance];
                    float * filterBuf = (float *)std::get<1>(filterBinary);
                    float * biasBuf = nullptr;
                    float * meanBuf = (float *)std::get<1>(meanBinary);
                    float * varianceBuf = (float *)std::get<1>(varianceBinary);
                    float * scaleBuf = nullptr;
                    float * offsetBuf = nullptr;
                    if(bias[0] != '0') {
                        std::tuple<unsigned int, char *> biasBinary = variableBinary[bias];
                        biasBuf = (float *)std::get<1>(biasBinary);
                    }
                    else if(convNewBiasName.find(outputConv) != convNewBiasName.end()) {
                        std::tuple<unsigned int, char *> biasBinary = variableBinary[convNewBiasName[outputConv]];
                        biasBuf = (float *)std::get<1>(biasBinary);
                    }
                    else {
                        size_t size = K * sizeof(float);
                        char * data = new char [size];
                        biasBuf = (float *)data;
                        for(size_t i = 0; i < K; i++) {
                            biasBuf[i] = 0;
                        }
                        std::string name = filter + "__new_bias";
                        std::tuple<unsigned int, char *> binary(size, data);
                        variableBinary[name] = binary;
                        convNewBiasName[outputConv] = name;
                        variableList.push_back(name);
                        variableMerged[name] = false;
                        nnef::Shape shape(1);
                        shape[0] = K;
                        shape[1] = 1;
                        variableShape[name] = shape;
                        variableRequiredDims[name] = 2;
                    }
                    if(scale[0] != '1') {
                        scaleBuf = (float *)std::get<1>(variableBinary[scale]);
                    }
                    if(offset[0] != '0') {
                        offsetBuf = (float *)std::get<1>(variableBinary[offset]);
                    }
                    for(size_t k = 0; k < K; k++) {
                        double mk = 1.0 / sqrt((double)varianceBuf[k] + epsilon);
                        double ck = -meanBuf[k] * mk;
                        if(scaleBuf) {
                            mk *= scaleBuf[k];
                            ck *= scaleBuf[k];
                        }
                        if(offsetBuf) {
                            ck += offsetBuf[k];
                        }
                        float * W = &filterBuf[k*N];
                        for(size_t j = 0; j < N; j++) {
                            W[j] = (float)(W[j] * mk);
                        }
                        if(biasBuf) {
                            biasBuf[k] = (float)(mk * biasBuf[k] + ck);
                        }
                    }
                    // mark that batch_normalization is disabled, rename output as input, and use conv as previous layer
                    operationRemoved[pos] = true;
                    virtualRename[argsBN["output"].tensor().id] = outputConv;
                    prevOutput = argsBN["output"].tensor().id;
                    // mark the merged variables
                    variableMerged[mean] = true;
                    variableMerged[variance] = true;
                    if(scaleBuf) variableMerged[scale] = true;
                    if(offsetBuf) variableMerged[offset] = true;
                }
                else {
                    // use batch_normalization as previous layer
                    prevPos = pos;
                    prevOpName = opname;
                    prevOutput = argsBN["output"].tensor().id;
                }
            }
            else if((prevOpName == "mul" || prevOpName == "add") && opname == "conv") {
                // get "mul" or "add" variables
                const nnef::Dictionary<nnef::Value>& argsOP = opsValues[prevPos];
                const nnef::Dictionary<nnef::Shape>& shapesOP = opsShapes[prevPos];
                const std::string& x = argsOP["x"].tensor().id;
                const std::string& y = argsOP["y"].tensor().id;
                std::string var, inputBN;
                nnef::Shape shapeVar;
                if(std::find(variableList.begin(), variableList.end(), x) != variableList.end()) {
                    inputBN = y;
                    var = x;
                    shapeVar = shapesOP[x];
                }
                else if(std::find(variableList.begin(), variableList.end(), y) != variableList.end()) {
                    inputBN = x;
                    var = y;
                    shapeVar = shapesOP[y];
                }
                // get "conv" variables
                const nnef::Dictionary<nnef::Value>& argsConv = opsValues[pos];
                const nnef::Dictionary<nnef::Shape>& shapesConv = opsShapes[pos];
                const std::string& outputConv = argsConv["output"].tensor().id;
                const std::string& filter = argsConv["filter"].tensor().id;
                const std::string& bias = getTensorOrScalar(argsConv["bias"]);
                const nnef::Shape& shapeFilter = shapesConv[filter];
                // get var dimensions
                size_t filterDimsCount = shapeFilter.rank(), varDimsCount = 0;
                std::vector<size_t> filterDims, varDims;
                getTensorDims(shapeFilter, filterDims, filterDimsCount);
                if(var.length() > 0) {
                    varDimsCount = shapeVar.rank();
                    getTensorDims(shapeVar, varDims, varDimsCount);
                }
                // check validity of dimensions
                size_t K = (filterDimsCount == 4) ? filterDims[3] : filterDims[1];
                size_t N = (filterDimsCount == 4) ? (filterDims[0] * filterDims[1] * filterDims[2]) : filterDims[0];
                if((filterDimsCount == 4 || filterDimsCount == 2) && varDimsCount == 2 && K == varDims[0]) {
                    // fuse var into conv variables
                    std::tuple<unsigned int, char *> filterBinary = variableBinary[filter];
                    std::tuple<unsigned int, char *> biasBinary = variableBinary[bias];
                    std::tuple<unsigned int, char *> varBinary = variableBinary[var];
                    float * filterBuf = (float *)std::get<1>(filterBinary);
                    float * biasBuf = nullptr;
                    float * varBuf = (float *)std::get<1>(varBinary);
                    if(bias[0] != '0') {
                        std::tuple<unsigned int, char *> biasBinary = variableBinary[bias];
                        biasBuf = (float *)std::get<1>(biasBinary);
                    }
                    else if(convNewBiasName.find(outputConv) != convNewBiasName.end()) {
                        std::tuple<unsigned int, char *> biasBinary = variableBinary[convNewBiasName[outputConv]];
                        biasBuf = (float *)std::get<1>(biasBinary);
                    }
                    else {
                        size_t size = K * sizeof(float);
                        char * data = new char [size];
                        biasBuf = (float *)data;
                        for(size_t i = 0; i < K; i++) {
                            biasBuf[i] = 0;
                        }
                        std::string name = filter + "__new_bias";
                        std::tuple<unsigned int, char *> binary(size, data);
                        variableBinary[name] = binary;
                        convNewBiasName[outputConv] = name;
                        variableList.push_back(name);
                        variableMerged[name] = false;
                        nnef::Shape shape(1);
                        shape[0] = K;
                        shape[1] = 1;
                        variableShape[name] = shape;
                        variableRequiredDims[name] = 2;
                    }
                    if(prevOpName == "mul") {
                        for(size_t k = 0; k < K; k++) {
                            double mk = varBuf[k];
                            size_t N = filterDims[0] * filterDims[1] * filterDims[2];
                            float * W = &filterBuf[k*N];
                            for(size_t j = 0; j < N; j++) {
                                W[j] = (float)(W[j] * mk);
                            }
                        }
                    }
                    else {
                        for(size_t k = 0; k < K; k++) {
                            double ck = varBuf[k];
                            size_t N = filterDims[0] * filterDims[1] * filterDims[2];
                            float * W = &filterBuf[k*N];
                            double Wsum = 0;
                            for(size_t j = 0; j < N; j++) {
                                Wsum += W[j];
                            }
                            biasBuf[k] = (float)(ck * Wsum + biasBuf[k]);
                        }
                    }
                    // mark that OP is disabled, rename output as input, and use conv as previous layer
                    operationRemoved[prevPos] = true;
                    virtualRename[argsConv["input"].tensor().id] = inputBN;
                    prevOutput = argsConv["output"].tensor().id;
                    // mark the merged variables
                    variableMerged[var] = true;
                }
                else {
                    // use conv as previous layer
                    prevPos = pos;
                    prevOpName = opname;
                    prevOutput = argsConv["output"].tensor().id;
                }
            }
            else if(prevOpName == "conv" && (opname == "mul" || opname == "add")) {
                // get "conv" variables
                const nnef::Dictionary<nnef::Value>& argsConv = opsValues[prevPos];
                const nnef::Dictionary<nnef::Shape>& shapesConv = opsShapes[prevPos];
                const std::string& outputConv = argsConv["output"].tensor().id;
                const std::string& filter = argsConv["filter"].tensor().id;
                const std::string& bias = getTensorOrScalar(argsConv["bias"]);
                const nnef::Shape& shapeFilter = shapesConv[filter];
                // get "mul" or "add" variables
                const nnef::Dictionary<nnef::Value>& argsOP = opsValues[pos];
                const nnef::Dictionary<nnef::Shape>& shapesOP = opsShapes[pos];
                const std::string& x = argsOP["x"].tensor().id;
                const std::string& y = argsOP["y"].tensor().id;
                std::string var;
                nnef::Shape shapeVar;
                if(std::find(variableList.begin(), variableList.end(), x) != variableList.end()) {
                    var = x;
                    shapeVar = shapesOP[x];
                }
                else if(std::find(variableList.begin(), variableList.end(), y) != variableList.end()) {
                    var = y;
                    shapeVar = shapesOP[y];
                }
                // get var dimensions
                size_t filterDimsCount = shapeFilter.rank(), varDimsCount = 0;
                std::vector<size_t> filterDims, varDims;
                getTensorDims(shapeFilter, filterDims, filterDimsCount);
                if(var.length() > 0) {
                    varDimsCount = shapeVar.rank();
                    getTensorDims(shapeVar, varDims, varDimsCount);
                }
                // check validity of dimensions
                size_t K = (filterDimsCount == 4) ? filterDims[3] : filterDims[1];
                size_t N = (filterDimsCount == 4) ? (filterDims[0] * filterDims[1] * filterDims[2]) : filterDims[0];
                if((filterDimsCount == 4 || filterDimsCount == 2) && varDimsCount == 2 && K == varDims[0]) {
                    // fuse var into conv variables
                    std::tuple<unsigned int, char *> filterBinary = variableBinary[filter];
                    std::tuple<unsigned int, char *> biasBinary = variableBinary[bias];
                    std::tuple<unsigned int, char *> varBinary = variableBinary[var];
                    float * filterBuf = (float *)std::get<1>(filterBinary);
                    float * biasBuf = nullptr;
                    float * varBuf = (float *)std::get<1>(varBinary);
                    if(bias[0] != '0') {
                        std::tuple<unsigned int, char *> biasBinary = variableBinary[bias];
                        biasBuf = (float *)std::get<1>(biasBinary);
                    }
                    else if(convNewBiasName.find(outputConv) != convNewBiasName.end()) {
                        std::tuple<unsigned int, char *> biasBinary = variableBinary[convNewBiasName[outputConv]];
                        biasBuf = (float *)std::get<1>(biasBinary);
                    }
                    else {
                        size_t size = K * sizeof(float);
                        char * data = new char [size];
                        biasBuf = (float *)data;
                        for(size_t i = 0; i < K; i++) {
                            biasBuf[i] = 0;
                        }
                        std::string name = filter + "__new_bias";
                        std::tuple<unsigned int, char *> binary(size, data);
                        variableBinary[name] = binary;
                        convNewBiasName[outputConv] = name;
                        variableList.push_back(name);
                        variableMerged[name] = false;
                        nnef::Shape shape(1);
                        shape[0] = K;
                        shape[1] = 1;
                        variableShape[name] = shape;
                        variableRequiredDims[name] = 2;
                    }
                    if(opname == "mul") {
                        for(size_t k = 0; k < K; k++) {
                            double mk = varBuf[k];
                            float * W = &filterBuf[k*N];
                            for(size_t j = 0; j < N; j++) {
                                W[j] = (float)(W[j] * mk);
                            }
                            if(biasBuf) {
                                biasBuf[k] = (float)(mk * biasBuf[k]);
                            }
                        }
                    }
                    else {
                        for(size_t k = 0; k < K; k++) {
                            float ck = varBuf[k];
                            biasBuf[k] = biasBuf[k] + ck;
                        }
                    }
                    // mark that OP is disabled, rename output as input, and use conv as previous layer
                    operationRemoved[pos] = true;
                    virtualRename[argsOP["z"].tensor().id] = outputConv;
                    prevOutput = argsOP["z"].tensor().id;
                    // mark the merged variables
                    variableMerged[var] = true;
                }
                else {
                    // use OP as previous layer
                    prevPos = pos;
                    prevOpName = opname;
                    prevOutput = argsOP["z"].tensor().id;
                }
            }
            else if(opname == "max_pool" || opname == "avg_pool") {
                const nnef::Dictionary<nnef::Value>& args = opsValues[pos];
                const std::string& input = args["input"].tensor().id;
                if(input != prevOutput || prevOpName != "conv") {
                    prevPos = pos;
                    prevOpName = opname;
                }
                prevOutput = args["output"].tensor().id;
            }
            else if(opname == "conv" || opname == "batch_normalization") {
                const nnef::Dictionary<nnef::Value>& args = opsValues[pos];
                const std::string& input = args["input"].tensor().id;
                prevPos = pos;
                prevOpName = opname;
                prevOutput = args["output"].tensor().id;
            }
            else if(opname == "add" || opname == "mul") {
                const nnef::Dictionary<nnef::Value>& args = opsValues[pos];
                const std::string& input1 = args["x"].tensor().id;
                const std::string& input2 = args["y"].tensor().id;
                prevPos = pos;
                prevOpName = opname;
                prevOutput = args["z"].tensor().id;
            }
            else {
                prevPos = 0;
                prevOpName = "";
                prevOutput = "";
            }
        }
    }

protected:
    ////
    // translator callback implementations
    //
    virtual void beginGraph( const nnef::Prototype& proto )
    {
        // show NNEF syntax
        if(verbose & 1) {
            std::cout << "graph " << proto.name() << "( ";
            for ( size_t i = 0; i < proto.paramCount(); ++i ) {
                auto& param = proto.param(i);
                if ( i ) std::cout << ", ";
                std::cout << param.name();
            }
            std::cout << " ) -> ( ";
            for ( size_t i = 0; i < proto.resultCount(); ++i ) {
                auto& result = proto.result(i);
                if ( i ) std::cout << ", ";
                std::cout << result.name();
            }
            std::cout << " )" << std::endl << '{' << std::endl;
        }

        ////
        // get input and output parameter list
        //
        for (size_t i = 0; i < proto.paramCount(); ++i) {
            inputList.push_back(proto.param(i).name());
        }
        for (size_t i = 0; i < proto.resultCount(); ++i) {
            outputList.push_back(proto.result(i).name());
        }

        ////
        // generate OpenVX C code preamble
        //
        openvxFilenameC = openvxFolder + "/annmodule.cpp";
        ovxC.open(openvxFilenameC);
        if(!ovxC) {
            printf("ERROR: unable to create: %s\n", openvxFilenameC.c_str());
            exit(1);
        }
    }

    virtual void endGraph( const nnef::Prototype& proto )
    {
        // show NNEF syntax
        if(verbose & 1) {
            std::cout << '}' << std::endl;
        }

        ////
        // generate OpenVX C code preamble
        //
        ovxC << "#include \"annmodule.h\"" << std::endl
             << "#include <VX/vx_khr_nn.h>" << std::endl
             << "#include <vx_amd_nn.h>" << std::endl
             << "#include <vx_ext_amd.h>" << std::endl
             << "#include <stdio.h>" << std::endl
             << std::endl
             << "#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status     , \"ERROR: failed with status = (%d) at \" __FILE__ \"#%d\\n\", status, __LINE__); return status; } }" << std::endl
             << "#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status, \"ERROR: failed with status = (%d) at \" __FILE__ \"#%d\\n\", status, __LINE__); return status; } }" << std::endl
             << std::endl
             << "static vx_status initializeTensor(vx_context context, vx_tensor tensor, FILE * fp, const char * binaryFilename)" << std::endl
             << "{" << std::endl
             << "    vx_enum data_type = VX_TYPE_FLOAT32;" << std::endl
             << "    vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];" << std::endl
             << "    ERROR_CHECK_STATUS(vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_type, sizeof(vx_enum)));" << std::endl
             << "    ERROR_CHECK_STATUS(vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(vx_size)));" << std::endl
             << "    ERROR_CHECK_STATUS(vxQueryTensor(tensor, VX_TENSOR_DIMS, &dims, num_of_dims * sizeof(vx_size)));" << std::endl
             << "    vx_size itemsize = sizeof(float);" << std::endl
             << "    if(data_type == VX_TYPE_UINT8 || data_type == VX_TYPE_INT8) {" << std::endl
             << "        itemsize = sizeof(vx_uint8);" << std::endl
             << "    }" << std::endl
             << "    else if(data_type == VX_TYPE_UINT16 || data_type == VX_TYPE_INT16 || data_type == VX_TYPE_FLOAT16) {" << std::endl
             << "        itemsize = sizeof(vx_uint16);" << std::endl
             << "    }" << std::endl
             << "    vx_size count = dims[0] * dims[1] * dims[2] * dims[3];" << std::endl
             << std::endl
             << "    vx_uint32 h[2] = { 0 };" << std::endl
             << "    fread(h, 1, sizeof(h), fp);" << std::endl
             << "    if(h[0] != 0x" << std::hex << VARIABLES_DATA_MAGIC << std::dec << " || (vx_size)h[1] != (count*itemsize)) {" << std::endl
             << "      vxAddLogEntry((vx_reference)tensor, VX_FAILURE, \"ERROR: invalid data (magic,size)=(0x%x,%d) in %s at byte position %d -- expected size is %ld\\n\", h[0], h[1], binaryFilename, ftell(fp)-sizeof(h), count*itemsize);" << std::endl
             << "      return VX_FAILURE;" << std::endl
             << "    }" << std::endl
             << std::endl
             << "    vx_map_id map_id;" << std::endl
             << "    float * ptr;" << std::endl
             << "    ERROR_CHECK_STATUS(vxMapTensorPatch(tensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));" << std::endl
             << "    vx_size n = fread(ptr, itemsize, count, fp);" << std::endl
             << "    if(n != count) {" << std::endl
             << "        vxAddLogEntry((vx_reference)tensor, VX_FAILURE, \"ERROR: expected char[%ld], but got char[%ld] in %s\\n\", count*itemsize, n*itemsize, binaryFilename);" << std::endl
             << "        return VX_FAILURE;" << std::endl
             << "    }" << std::endl
             << "    ERROR_CHECK_STATUS(vxUnmapTensorPatch(tensor, map_id));" << std::endl
             << std::endl
             << "    return VX_SUCCESS;" << std::endl
             << "}" << std::endl
             << std::endl
             << "vx_status annAddToGraph(vx_graph graph";
        for(auto& name : inputList) {
            ovxC << ", vx_tensor " << name;
        }
        for(auto& name : outputList) {
            ovxC << ", vx_tensor " << name;
        }
        ovxC << ", const char * binaryFilename)" << std::endl
             << "{" << std::endl
             << "    vx_context context = vxGetContext((vx_reference)graph);" << std::endl
             << "    ERROR_CHECK_OBJECT(context);" << std::endl
             << "    ERROR_CHECK_STATUS(vxLoadKernels(context, \"vx_nn\"));" << std::endl;

        ////
        // get variables
        //
        for(size_t i = 0; i < opsProto.size(); i++) {
            codeGenOperation(i, true, false, verbose);
        }

        ////
        // get data
        //
        for(auto& name : variableList) {
            unsigned int size = 0;
            char * data = nullptr;
            if(variableShape.find(name) != variableShape.end() && variableLabel.find(name) != variableLabel.end()) {
                auto& shape = variableShape[name];
                auto& label = variableLabel[name];
                size = loadTensorFile(nnefFolder, label, shape, data);
            }
            if(size > 0 && data) {
                std::tuple<unsigned int, char *> binary(size, data);
                variableBinary[name] = binary;
            }
            else {
                printf("ERROR: unable to load binary data for variable '%s'\n", name.c_str());
                exit(1);
            }
        }

        ////
        // merge variables
        //
        codeGenMergeVariables();

        ////
        // create and initialize variables file
        //
        ovxC << std::endl;
        ovxC << "    // create variables" << std::endl;
        for(auto& name : variableList) {
            if(!variableMerged[name]) {
                if(variableShape.find(name) != variableShape.end()) {
                    auto& shape = variableShape[name];
                    int num_dims = 0;
                    auto it = variableRequiredDims.find(name);
                    if(it != variableRequiredDims.end()) {
                        num_dims = it->second;
                    }
                    ovxC << codeGenTensorCreate(name, shape, false, num_dims);
                }
                else {
                    printf("ERROR: something wrong with variable '%s': variableShape is missing\n", name.c_str());
                    exit(1);
                }
            }
        }
        ovxC << std::endl
             << "    // initialize variables" << std::endl
             << "    FILE * fp__variables = fopen(binaryFilename, \"rb\");" << std::endl
             << "    if(!fp__variables) {" << std::endl
             << "        vxAddLogEntry((vx_reference)context, VX_FAILURE, \"ERROR: unable to open: %s\\n\", binaryFilename);" << std::endl
             << "        return VX_FAILURE;" << std::endl
             << "    }" << std::endl
             << "    { vx_uint32 magic = 0;" << std::endl
             << "      fread(&magic, 1, sizeof(magic), fp__variables);" << std::endl
             << "      if(magic != 0x" << std::hex << VARIABLES_FILE_MAGIC << std::dec << ") {" << std::endl
             << "        vxAddLogEntry((vx_reference)context, VX_FAILURE, \"ERROR: invalid file magic in %s\\n\", binaryFilename);" << std::endl
             << "        return VX_FAILURE;" << std::endl
             << "      }" << std::endl
             << "    }" << std::endl;
        std::string variablesFilename = openvxFolder + "/weights.bin";
        FILE * fpVariables = fopen(variablesFilename.c_str(), "wb");
        if(!fpVariables) {
            printf("ERROR: unable to create: %s\n", variablesFilename.c_str());
            exit(1);
        }
        unsigned int magic_file = VARIABLES_FILE_MAGIC;
        unsigned int magic_data = VARIABLES_DATA_MAGIC;
        fwrite(&magic_file, 1, sizeof(magic_file), fpVariables);
        for(auto& name : variableList) {
            if(!variableMerged[name]) {
                if(variableShape.find(name) != variableShape.end()) {
                    auto& shape = variableShape[name];
                    std::tuple<unsigned int, char *> binary = variableBinary[name];
                    unsigned int size = std::get<0>(binary);
                    char * data = std::get<1>(binary);
                    if(size > 0 && data) {
                        fwrite(&magic_data, 1, sizeof(magic_data), fpVariables);
                        fwrite(&size, 1, sizeof(size), fpVariables);
                        fwrite(data, 1, size, fpVariables);
                        delete[] data;
                        std::tuple<unsigned int, char *> empty(0, nullptr);
                        variableBinary[name] = empty;
                        ovxC << "    ERROR_CHECK_STATUS(initializeTensor(context, " << name << ", fp__variables, binaryFilename));" << std::endl;
                    }
                    else {
                        printf("ERROR: something wrong with variable '%s': variableBinary is not valid\n", name.c_str());
                        exit(1);
                    }
                }
                else {
                    printf("ERROR: something wrong with variable '%s': variableShape is missing\n", name.c_str());
                    exit(1);
                }
            }
        }
        unsigned int magic_eoff = VARIABLES_EOFF_MAGIC;
        fwrite(&magic_eoff, 1, sizeof(magic_eoff), fpVariables);
        fclose(fpVariables);
        ovxC << "    { vx_uint32 magic = 0;" << std::endl
             << "      fread(&magic, 1, sizeof(magic), fp__variables);" << std::endl
             << "      if(magic != 0x" << std::hex << VARIABLES_EOFF_MAGIC << std::dec << ") {" << std::endl
             << "        vxAddLogEntry((vx_reference)context, VX_FAILURE, \"ERROR: invalid eoff magic in %s\\n\", binaryFilename);" << std::endl
             << "        return VX_FAILURE;" << std::endl
             << "      }" << std::endl
             << "      fclose(fp__variables);" << std::endl
             << "    }" << std::endl;
        std::cout << "OK: created '" << variablesFilename << "'" << std::endl;

        ////
        // instantiate nodes in graph
        //
        ovxC << std::endl;
        ovxC << "    // create nodes in graph" << std::endl;
        for(auto i = 0; i < opsProto.size(); i++) {
            codeGenOperation(i, false, true, 0);
        }

        ////
        // generate clean-up code
        //
        ovxC << std::endl;
        ovxC << "    // release internal tensors" << std::endl;
        for(auto& name : virtualList) {
            if(virtualRename.find(name) == virtualRename.end()) {
                ovxC << "    ERROR_CHECK_STATUS(vxReleaseTensor(&" << name << "));" << std::endl;
            }
        }
        for(auto& name : variableList) {
            if(!variableMerged[name]) {
                ovxC << "    ERROR_CHECK_STATUS(vxReleaseTensor(&" << name << "));" << std::endl;
            }
        }
        ovxC << std::endl;
        ovxC << "    return VX_SUCCESS;" << std::endl;
        ovxC << "}" << std::endl;
        ovxC.close();
        std::cout << "OK: created '" << openvxFilenameC << "'" << std::endl;

        ////
        // generate OpenVX header file
        //
        openvxFilenameC = openvxFolder + "/annmodule.h";
        ovxC.open(openvxFilenameC);
        if(!ovxC) {
            printf("ERROR: unable to create: %s\n", openvxFilenameC.c_str());
            exit(1);
        }
        ovxC << "#ifndef included_file_annmodule_h" << std::endl
             << "#define included_file_annmodule_h" << std::endl
             << std::endl
             << "#include <VX/vx.h>" << std::endl
             << std::endl;
        ovxC << "////" << std::endl
             << "// initialize graph neural network for inference" << std::endl;
        for(auto& name : inputList) {
            if(inputShape.find(name) != inputShape.end()) {
                std::vector<size_t> dims;
                getTensorDims(inputShape[name], dims, 4);
                ovxC << "//   " << name << " -- dims[] = {";
                for(size_t i = 0; i < dims.size(); i++) {
                    ovxC << (i == 0 ? " " : ", ") << dims[i];
                }
                ovxC << " } (input)" << std::endl;
            }
        }
        for(auto& name : outputList) {
            if(outputShape.find(name) != outputShape.end()) {
                std::vector<size_t> dims;
                getTensorDims(outputShape[name], dims, 4);
                ovxC << "//   " << name << " -- dims[] = {";
                for(size_t i = 0; i < dims.size(); i++) {
                    ovxC << (i == 0 ? " " : ", ") << dims[i];
                }
                ovxC << " } (output)" << std::endl;
            }
        }
        ovxC << "//" << std::endl
             << "vx_status annAddToGraph(vx_graph graph";
        for(auto& name : inputList) {
            ovxC << ", vx_tensor " << name;
        }
        for(auto& name : outputList) {
            ovxC << ", vx_tensor " << name;
        }
        ovxC << ", const char * binaryFilename);" << std::endl
             << std::endl
             << "#endif" << std::endl;
        ovxC.close();
        std::cout << "OK: created '" << openvxFilenameC << "'" << std::endl;

        ////
        // generate a simple test program
        //
        openvxFilenameC = openvxFolder + "/anntest.cpp";
        ovxC.open(openvxFilenameC);
        if(!ovxC) {
            printf("ERROR: unable to create: %s\n", openvxFilenameC.c_str());
            exit(1);
        }
        ovxC << "#include \"annmodule.h\"" << std::endl
             << "#include <vx_ext_amd.h>" << std::endl
             << "#include <iostream>" << std::endl
             << "#include <stdio.h>" << std::endl
             << "#include <string.h>" << std::endl
             << "#include <string>" << std::endl
             << "#include <inttypes.h>" << std::endl
             << "#include <chrono>" << std::endl
             << "#include <unistd.h>" << std::endl
             << "" << std::endl
             << "#if ENABLE_OPENCV" << std::endl
             << "#include <opencv2/opencv.hpp>" << std::endl
             << "#include <opencv/cv.h>" << std::endl
             << "#include <opencv/highgui.h>" << std::endl
             << "using namespace cv; " << std::endl
             << "#endif" << std::endl
             << "" << std::endl
             << "#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { printf(\"ERROR: failed with status = (%d) at \" __FILE__ \"#%d\", status, __LINE__); return -1; } }" << std::endl
             << "" << std::endl
             << "static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])" << std::endl
             << "{" << std::endl
             << "    size_t len = strlen(string);" << std::endl
             << "    if (len > 0) {" << std::endl
             << "        printf(\"%s\", string);" << std::endl
             << "        if (string[len - 1] != '\\n')" << std::endl
             << "            printf(\"\\n\");" << std::endl
             << "        fflush(stdout);" << std::endl
             << "    }" << std::endl
             << "}" << std::endl
             << "" << std::endl
             << "inline int64_t clockCounter()" << std::endl
             << "{" << std::endl
             << "    return std::chrono::high_resolution_clock::now().time_since_epoch().count();" << std::endl
             << "}" << std::endl
             << "" << std::endl
             << "inline int64_t clockFrequency()" << std::endl
             << "{" << std::endl
             << "    return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;" << std::endl
             << "}" << std::endl
             << "" << std::endl
             << "static vx_status copyTensor(vx_tensor tensor, std::string fileName, vx_enum usage = VX_WRITE_ONLY)" << std::endl
             << "{" << std::endl
             << "    vx_enum data_type = VX_TYPE_FLOAT32;" << std::endl
             << "    vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];" << std::endl
             << "    vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));" << std::endl
             << "    vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));" << std::endl
             << "    vxQueryTensor(tensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);" << std::endl
             << "    vx_size itemsize = sizeof(float);" << std::endl
             << "    if(data_type == VX_TYPE_UINT8 || data_type == VX_TYPE_INT8) {" << std::endl
             << "        itemsize = sizeof(vx_uint8);" << std::endl
             << "    }" << std::endl
             << "    else if(data_type == VX_TYPE_UINT16 || data_type == VX_TYPE_INT16 || data_type == VX_TYPE_FLOAT16) {" << std::endl
             << "        itemsize = sizeof(vx_uint16);" << std::endl
             << "    }" << std::endl
             << "    vx_size count = dims[0] * dims[1] * dims[2] * dims[3];" << std::endl
             << "    vx_map_id map_id;" << std::endl
             << "    float * ptr;" << std::endl
             << "    vx_status status = vxMapTensorPatch(tensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);" << std::endl
             << "    if(status) {" << std::endl
             << "        std::cerr << \"ERROR: vxMapTensorPatch() failed for \" << fileName << std::endl;" << std::endl
             << "        return -1;" << std::endl
             << "    }" << std::endl
             << "    if(usage == VX_WRITE_ONLY) {" << std::endl
             << "#if ENABLE_OPENCV" << std::endl
             << "        if(dims[3] == 1 && dims[2] == 3 && fileName.size() > 4 && (fileName.substr(fileName.size()-4, 4) == \".png\" || fileName.substr(fileName.size()-4, 4) == \".jpg\"))" << std::endl
             << "        {" << std::endl
             << "            Mat img = imread(fileName.c_str(), CV_LOAD_IMAGE_COLOR);" << std::endl
             << "            if(!img.data || img.rows != dims[1] || img.cols != dims[0]) {" << std::endl
             << "                std::cerr << \"ERROR: invalid image or dimensions in \" << fileName << std::endl;" << std::endl
             << "                return -1;" << std::endl
             << "            }" << std::endl
             << "            unsigned char * src = img.data;" << std::endl
             << "            for(vx_size c = 0; c < 3; c++) {" << std::endl
             << "                for(vx_size y = 0; y < dims[1]; y++) {" << std::endl
             << "                    for(vx_size x = 0; x < dims[0]; x++) {" << std::endl
             << "                        ptr[(c*stride[2]+y*stride[1]+x*stride[0])>>2] = src[y*dims[0]*3+x*3+c];" << std::endl
             << "                    }" << std::endl
             << "                }" << std::endl
             << "            }" << std::endl
             << "        }" << std::endl
             << "        else" << std::endl
             << "#endif" << std::endl
             << "        {" << std::endl
             << "            FILE * fp = fopen(fileName.c_str(), \"rb\");" << std::endl
             << "            if(!fp) {" << std::endl
             << "                std::cerr << \"ERROR: unable to open: \" << fileName << std::endl;" << std::endl
             << "                return -1;" << std::endl
             << "            }" << std::endl
             << "            vx_size n = fread(ptr, itemsize, count, fp);" << std::endl
             << "            fclose(fp);" << std::endl
             << "            if(n != count) {" << std::endl
             << "                std::cerr << \"ERROR: expected char[\" << count*itemsize << \"], but got char[\" << n*itemsize << \"] in \" << fileName << std::endl;" << std::endl
             << "                return -1;" << std::endl
             << "            }" << std::endl
             << "        }" << std::endl
             << "    }" << std::endl
             << "    else {" << std::endl
             << "        FILE * fp = fopen(fileName.c_str(), \"wb\");" << std::endl
             << "        if(!fp) {" << std::endl
             << "            std::cerr << \"ERROR: unable to open: \" << fileName << std::endl;" << std::endl
             << "            return -1;" << std::endl
             << "        }" << std::endl
             << "        fwrite(ptr, itemsize, count, fp);" << std::endl
             << "        fclose(fp);" << std::endl
             << "    }" << std::endl
             << "    status = vxUnmapTensorPatch(tensor, map_id);" << std::endl
             << "    if(status) {" << std::endl
             << "        std::cerr << \"ERROR: vxUnmapTensorPatch() failed for \" << fileName << std::endl;" << std::endl
             << "        return -1;" << std::endl
             << "    }" << std::endl
             << "    return 0;" << std::endl
             << "}" << std::endl
             << "" << std::endl
             << "int main(int argc, const char ** argv)" << std::endl
             << "{" << std::endl
             << "    // check command-line usage" << std::endl
             << "    if(argc < 2) {" << std::endl
             << "        printf(\"Usage: anntest <weights.bin> [<input/output-filename(s)>...]\\n\");" << std::endl
             << "        return -1;" << std::endl
             << "    }" << std::endl
             << "    const char * binaryFilename = argv[1];" << std::endl
             << "    argc -= 2;" << std::endl
             << "    argv += 2;" << std::endl
             << "" << std::endl
             << "    // create context, input, output, and graph" << std::endl
             << "    vxRegisterLogCallback(NULL, log_callback, vx_false_e);" << std::endl
             << "    vx_context context = vxCreateContext();" << std::endl
             << "    if(vxGetStatus((vx_reference)context)) {" << std::endl
             << "        printf(\"ERROR: vxCreateContext() failed\\n\");" << std::endl
             << "        return -1;" << std::endl
             << "    }" << std::endl
             << "    vxRegisterLogCallback(context, log_callback, vx_false_e);" << std::endl
             << "" << std::endl
             << "    // create input tensors and initialize" << std::endl
            ;
        for(auto& name : inputList) {
            std::vector<size_t> dims;
            getTensorDims(inputShape[name], dims, 4);
            ovxC << "    vx_size " << name << "_dims[" << dims.size() << "] = {";
            for(size_t i = 0; i < dims.size(); i++) {
                ovxC << (i == 0 ? " " : ", ") << dims[i];
            }
            ovxC << " };" << std::endl
                 << "    vx_tensor " << name << " = vxCreateTensor(context, " << dims.size() << ", " << name << "_dims, VX_TYPE_FLOAT32, 0);" << std::endl
                 << "    if(vxGetStatus((vx_reference)" << name << ")) {" << std::endl
                 << "        printf(\"ERROR: vxCreateTensor() failed for " << name << "\\n\");" << std::endl
                 << "        return -1;" << std::endl
                 << "    }" << std::endl
                 << "    if(*argv) {" << std::endl
                 << "        if(strcmp(*argv, \"-\") != 0) {" << std::endl
                 << "            if(copyTensor(" << name << ", *argv, VX_WRITE_ONLY) < 0) {" << std::endl
                 << "                return -1;" << std::endl
                 << "            }" << std::endl
                 << "            printf(\"OK: read tensor '" << name << "' from %s\\n\", *argv);" << std::endl
                 << "        }" << std::endl
                 << "        argv++;" << std::endl
                 << "    }" << std::endl
                ;
        }
        ovxC << "    // create output tensors" << std::endl;
        for(auto& name : outputList) {
            std::vector<size_t> dims;
            getTensorDims(outputShape[name], dims, 4);
            ovxC << "    vx_size " << name << "_dims[" << dims.size() << "] = {";
            for(size_t i = 0; i < dims.size(); i++) {
                ovxC << (i == 0 ? " " : ", ") << dims[i];
            }
            ovxC << " };" << std::endl
                 << "    vx_tensor " << name << " = vxCreateTensor(context, " << dims.size() << ", " << name << "_dims, VX_TYPE_FLOAT32, 0);" << std::endl
                 << "    if(vxGetStatus((vx_reference)" << name << ")) {" << std::endl
                 << "        printf(\"ERROR: vxCreateTensor() failed for " << name << "\\n\");" << std::endl
                 << "        return -1;" << std::endl
                 << "    }" << std::endl;
        }
        ovxC << "" << std::endl
             << "    // build graph using annmodule" << std::endl
             << "    vx_status status;" << std::endl
             << "    int64_t freq = clockFrequency(), t0, t1;" << std::endl
             << "    t0 = clockCounter();" << std::endl
             << "    vx_graph graph = vxCreateGraph(context);" << std::endl
             << "    status = vxGetStatus((vx_reference)graph);" << std::endl
             << "    if(status) {" << std::endl
             << "        printf(\"ERROR: vxCreateGraph(...) failed (%d)\\n\", status);" << std::endl
             << "        return -1;" << std::endl
             << "    }" << std::endl
             << "    status = annAddToGraph(graph, "
            ;
        for(auto& name : inputList) {
            ovxC << name << ", ";
        }
        for(auto& name : outputList) {
            ovxC << name << ", ";
        }
        ovxC << "binaryFilename);" << std::endl
             << "    if(status) {" << std::endl
             << "        printf(\"ERROR: annAddToGraph() failed (%d)\\n\", status);" << std::endl
             << "        return -1;" << std::endl
             << "    }" << std::endl
             << "    status = vxVerifyGraph(graph);" << std::endl
             << "    if(status) {" << std::endl
             << "        printf(\"ERROR: vxVerifyGraph(...) failed (%d)\\n\", status);" << std::endl
             << "        return -1;" << std::endl
             << "    }" << std::endl
             << "    t1 = clockCounter();" << std::endl
             << "    printf(\"OK: graph initialization with annAddToGraph() took %.3f msec\\n\", (float)(t1-t0)*1000.0f/(float)freq);" << std::endl
             << "" << std::endl
             << "    t0 = clockCounter();" << std::endl
             << "    status = vxProcessGraph(graph);" << std::endl
             << "    t1 = clockCounter();" << std::endl
             << "    if(status != VX_SUCCESS) {" << std::endl
             << "        printf(\"ERROR: vxProcessGraph() failed (%d)\\n\", status);" << std::endl
             << "        return -1;" << std::endl
             << "    }" << std::endl
             << "    printf(\"OK: vxProcessGraph() took %.3f msec (1st iteration)\\n\", (float)(t1-t0)*1000.0f/(float)freq);" << std::endl
             << "" << std::endl
             << "    // write outputs" << std::endl
            ;
        for(auto& name : outputList) {
            ovxC << "    if(*argv) {" << std::endl
                 << "        if(strcmp(*argv, \"-\") != 0) {" << std::endl
                 << "            if(copyTensor(" << name << ", *argv, VX_READ_ONLY) < 0) {" << std::endl
                 << "                return -1;" << std::endl
                 << "            }" << std::endl
                 << "            printf(\"OK: wrote tensor '" << name << "' into %s\\n\", *argv);" << std::endl
                 << "        }" << std::endl
                 << "        argv++;" << std::endl
                 << "    }" << std::endl
                ;
        }
        ovxC << "" << std::endl
             << "    t0 = clockCounter();" << std::endl
             << "    int N = 100;" << std::endl
             << "    for(int i = 0; i < N; i++) {" << std::endl
             << "        status = vxProcessGraph(graph);" << std::endl
             << "        if(status != VX_SUCCESS)" << std::endl
             << "            break;" << std::endl
             << "    }" << std::endl
             << "    t1 = clockCounter();" << std::endl
             << "    printf(\"OK: vxProcessGraph() took %.3f msec (average over %d iterations)\\n\", (float)(t1-t0)*1000.0f/(float)freq/(float)N, N);" << std::endl
             << "" << std::endl
             << "    // release resources" << std::endl
             << "    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));" << std::endl
            ;
        for(auto& name : inputList) {
            ovxC << "    ERROR_CHECK_STATUS(vxReleaseTensor(&" << name << "));" << std::endl;
        }
        for(auto& name : outputList) {
            ovxC << "    ERROR_CHECK_STATUS(vxReleaseTensor(&" << name << "));" << std::endl;
        }
        ovxC << "    ERROR_CHECK_STATUS(vxReleaseContext(&context));" << std::endl
             << "    printf(\"OK: successful\\n\");" << std::endl
             << "" << std::endl
             << "    return 0;" << std::endl
             << "}" << std::endl
             ;
        ovxC.close();
        std::cout << "OK: created '" << openvxFilenameC << "'" << std::endl;

        ////
        // generate CMakeLists.txt
        //
        openvxFilenameC = openvxFolder + "/CMakeLists.txt";
        ovxC.open(openvxFilenameC);
        if(!ovxC) {
            printf("ERROR: unable to create: %s\n", openvxFilenameC.c_str());
            exit(1);
        }
        ovxC << "cmake_minimum_required (VERSION 2.8)" << std::endl
             << "project (annmodule)" << std::endl
             << "set (CMAKE_CXX_STANDARD 11) " << std::endl
             << "list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)" << std::endl
             << "find_package(OpenCL REQUIRED)" << std::endl
             << "find_package(OpenCV QUIET)" << std::endl
             << "include_directories (${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers )" << std::endl
             << "include_directories (/opt/rocm/mivisionx/include)" << std::endl
             << "link_directories    (/opt/rocm/mivisionx/lib)" << std::endl
             << "list(APPEND SOURCES annmodule.cpp)" << std::endl
             << "add_library(${PROJECT_NAME} SHARED ${SOURCES})" << std::endl
             << "set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -msse4.2 -std=c++11\")" << std::endl
             << "target_link_libraries(${PROJECT_NAME} openvx vx_nn pthread)" << std::endl
             << "add_executable(anntest anntest.cpp)" << std::endl
             << "if (OpenCV_FOUND)" << std::endl
             << "  target_compile_definitions(anntest PUBLIC ENABLE_OPENCV=1)" << std::endl
             << "  include_directories(${OpenCV_INCLUDE_DIRS})" << std::endl
             << "  target_link_libraries(anntest ${OpenCV_LIBRARIES})" << std::endl
             << "else(OpenCV_FOUND)" << std::endl
             << "  target_compile_definitions(anntest PUBLIC ENABLE_OPENCV=0)" << std::endl
             << "endif(OpenCV_FOUND)" << std::endl
             << "target_link_libraries(anntest openvx vx_nn pthread ${PROJECT_NAME})" << std::endl
            ;
        ovxC.close();
        std::cout << "OK: created '" << openvxFilenameC << "'" << std::endl;
    }

    virtual void operation(const nnef::Prototype& proto,
                           const nnef::Dictionary<nnef::Value>& args,
                           const nnef::Dictionary<nnef::Shape>& shapes)
    {
        // save the operation details
        opsProto.push_back(proto);
        opsValues.push_back(args);
        opsShapes.push_back(shapes);
        operationRemoved.push_back(false);
    }

    virtual bool isAtomic( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args )
    {
        static std::set<std::string> atomics =
        {
            "sqr", "sqrt", "min", "max",
            "softmax", "relu", "tanh", "sigmoid",
            "batch_normalization", "max_pool", "avg_pool",
            "quantize_linear", "quantize_logarithmic"
        };
        return atomics.find(proto.name()) != atomics.end();
    }
};

int main(int argc, const char * argv[])
{
    ////
    // get command-line parameters
    //
    int verbose = 0;
    bool useVirtual = true;
    while(argc > 1 && argv[1][0] == '-') {
        if(!strcmp(argv[1], "--no-virtual")) {
            useVirtual = false;
            argc -= 1;
            argv += 1;
        }
        else if(argc > 2 && !strcmp(argv[1], "-v")) {
            verbose = atoi(argv[2]);
            argc -= 2;
            argv += 2;
        }
        else {
            printf("ERROR: invalid option: %s\n", argv[1]);
            return -1;
        }
    }
    if(argc < 3) {
        printf("Usage: nnef2openvx [-v <verbose>] [--no-virtual] <nnefContainerFolder> <openvxOutputFolder>\n");
        return -1;
    }
    std::string nnefContainedFolder = argv[1];
    std::string openvxOutputFolder = argv[2];
    std::string nnefFilename = nnefContainedFolder + "/graph.nnef";

    ////
    // parse NNEF structure and translate to OpenVX code
    //
    std::ifstream ifs(nnefFilename.c_str());
    if(!ifs) {
        printf("ERROR: unable to open: %s\n", nnefFilename.c_str());
        return -1;
    }
    mkdir(openvxOutputFolder.c_str(), 0777);
    printf("OK: parsing %s ...\n", nnefFilename.c_str());
    std::unique_ptr<nnef::Parser> parser((nnef::Parser*)new nnef::FlatParser());
    try {
        NNEF2OpenVX_Translator callback(nnefContainedFolder, openvxOutputFolder, useVirtual, verbose);
        parser->parse(ifs, callback);
    }
    catch(nnef::Error e) {
        printf("Parse error: [%u:%u] %s\n", e.position().line, e.position().column, e.what());
        auto origin = e.position().origin;
        while(origin) {
            printf("... evaluated from [%u:%u]\n", origin->line, origin->column);
            origin = origin->origin;
        }
    }
    ifs.close();

    return 0;
}
