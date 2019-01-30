# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

import sys, os, os.path
import numpy as np

class IrTensor:
    def __init__(self):
        self.name = 'Unknown'
        self.type = 'F032'
        self.shape = [0]
        self.format = 'NCHW'

    def setName(self,name):
        self.name = name

    def setInfo(self,type,shape):
        self.type = type
        self.shape = shape

    def setFormat(self,format):
        self.format = format

    def toString(self):
        return self.name + ';' + self.type + ';' + ','.join([str(v) for v in self.shape]) + ';' + self.format

    def fromString(self,s):
        lT = s.split(';')
        self.name = lT[0]
        self.type = lT[1]
        self.shape = [int(v) for v in lT[2].split(',')]
        self.format = lT[3]

class IrAttr:
    def __init__(self):
        self.dict_values = {
              'axis' : 0                # axis to compute
            , 'axes' : []               # list of axis
            , 'broadcast' : 0           # enable broadcasting
            , 'keepdims' : 1            # 1: keep reduced dimension
            , 'kernel_shape' : [1, 1]   # [x,y]
            , 'pads' : [0, 0, 0, 0]     # [left,top,right,bottom]
            , 'strides' : [1, 1]        # [x,y]
            , 'dilations' : [1, 1]      # [x,y]
            , 'group' : 1               # number of groups
            , 'epsilon' : 1e-5          # epsilon
            , 'alpha' : 1.0             # alpha
            , 'beta' : 1.0              # beta
            , 'transA' : 0              # transA
            , 'transB' : 0              # transB
            , 'bias' : 1.0              # bias
            , 'size' : 1                # size - number of channels to sum over
            , 'pooled_shape' : [1, 1]   # [x,y] ROI pool
            , 'spatial_scale' : 1.0     # spatial_scale - ROI pool
            , 'split' : []              # length of each output for split
            , 'border_mode' : 'fill_0'  # border mode: fill_0, discard
            , 'dim_round_mode' : 'floor' # rounding mode for output dim calculation: floor, ceil
            , 'mode' : 0                 # attribute to differentiate layer modes.
            , 'shape' : []               # shape attribute
            , 'zoom_factor' : 2          # zoom_factor
        }
        self.dict_set = []

    def set(self,name,value):
        if not name in self.dict_values:
            raise ValueError("Unsupported IR attribute: {}".format(name))
        if type(value) != type(self.dict_values[name]):
            raise ValueError("Invalid IR attribute value type: {} for {}".format(type(value).__name__, name))
        self.dict_values[name] = value
        if not name in self.dict_set:
            self.dict_set.append(name)

    def is_set(self,name):
        return True if name in self.dict_set else False

    def get(self,name):
        return self.dict_values[name]

    def toString(self):
        s = ''
        for name in self.dict_set:
            value = self.dict_values[name]
            if type(value).__name__ == 'list':
                skv = name + '=' + ','.join([str(v) for v in value])
            else:
                skv = name + '=' + str(value)
            s = skv if s == '' else s + ';' + skv
        return s

    def fromString(self,s):
        for sa in s.split(';'):
            saL = sa.split('=')
            name = saL[0]
            value = saL[1]
            value_type = type(self.dict_values[name]).__name__
            if value_type == 'list':
                self.set(name, [int(x) for x in value.split(',')])
            elif value_type == 'float':
                self.set(name, float(value))
            elif value_type == 'str':
                self.set(name, str(value))
            else:
                self.set(name, int(value))

class IrNode:
    def __init__(self):
        self.type = 'Unknown'
        self.inputs = []
        self.outputs = []
        self.attr = IrAttr()
        self.dict_types = {
            'conv' : 1,
            'conv_transpose' : 1,
            'batch_norm' : 1,
            'avg_pool' : 1,
            'max_pool' : 1,
            'relu' : 1,
            'sum' : 1,
            'add' : 1,
            'mul' : 1,
            'muladd' : 1,
            'sub' : 1,
            'gemm' : 1,
            'softmax' : 1,
            'lrn' : 1,
            'slice' : 1,
            'concat' : 1,
            'global_avg_pool' : 1,
            'leaky_relu' : 1,
            'reshape' : 1,
            'transpose' : 1,
            'copy' : 1,
            'upsample' : 1,
        }

    def set(self,type,inputs,outputs,attr):
        if not type in self.dict_types or self.dict_types[type] == 0:
            print('ERROR: IrNode.set: operation "%s" not supported' % (type))
            sys.exit(1)
        self.type = type
        self.inputs = inputs
        self.outputs = outputs
        self.attr = attr

    def toString(self):
        return 'node|' + self.type + \
                   '|' + ','.join([tensor for tensor in self.inputs]) + \
                   '|' + ','.join([tensor for tensor in self.outputs]) + \
                   '|' + self.attr.toString()

    def fromString(self,s):
        sL = s.split('|')
        self.type = sL[1]
        self.inputs = sL[2].split(',')
        self.outputs = sL[3].split(',')
        self.attr = IrAttr()
        if sL[4] != '':
            self.attr.fromString(sL[4])

class IrGraph:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.output_names = []
        self.initializers = []
        self.locals = []
        self.nodes = []
        self.binaries = {}
        self.tensor_dict = {}
        self.tensor_types = {}
        self.tensor_shapes = {}
        self.all_F032 = True
        self.all_F016 = False

    def addInput(self,tensor):
        self.inputs.append(tensor)
        self.tensor_dict[tensor.name] = tensor
        self.tensor_types[tensor.name] = tensor.type
        self.tensor_shapes[tensor.name] = tensor.shape
        if tensor.type != 'F032':
            self.all_F032 = False
            if tensor.type == 'F016':
                self.all_F016 = True
                self.all_F032 = False
    def addOutput(self,tensor):
        self.outputs.append(tensor)
        self.tensor_dict[tensor.name] = tensor
        self.tensor_types[tensor.name] = tensor.type
        self.tensor_shapes[tensor.name] = tensor.shape
        if self.all_F032 == True and tensor.type != 'F032':
            self.all_F032 = False
        if self.all_F016 == True and tensor.type != 'F016':
            self.all_F016 = False
        self.output_names.append(tensor.name)

    def addVariable(self,tensor):
        if len(tensor.shape) == 1:
            tensor.shape = [1, tensor.shape[0]]
        self.initializers.append(tensor)
        self.tensor_dict[tensor.name] = tensor
        self.tensor_types[tensor.name] = tensor.type
        self.tensor_shapes[tensor.name] = tensor.shape
        if self.all_F032 == True and tensor.type != 'F032':
            self.all_F032 = False
        if self.all_F016 == True and tensor.type != 'F016':
            self.all_F016 = False

    def addLocal(self,tensor):
        self.tensor_dict[tensor.name] = tensor
        self.tensor_types[tensor.name] = tensor.type
        self.tensor_shapes[tensor.name] = tensor.shape
        if self.all_F032 == True and tensor.type != 'F032':
            self.all_F032 = False
        if self.all_F016 == True and tensor.type != 'F016':
            self.all_F016 = False
        if not tensor.name in self.output_names:
            self.locals.append(tensor)

    def addNode(self,node):
        self.nodes.append(node)
        # sanity checks
        if node.type == 'mul' and len(node.inputs) != 2:
            raise ValueError("Unsupported 'mul': expects inputs to be 2")

    def addBinary(self,tensorName,binary):
        self.binaries[tensorName] = binary

    def removeTensor(self,name):
        tensor = self.tensor_dict[name]
        if tensor in self.initializers:
            self.initializers.remove(tensor)
            del self.binaries[tensor.name]
        elif tensor in self.locals:
            self.locals.remove(tensor)
        else:
            raise ValueError("nnir.removeTensor couldn't find : {}".format(tensor.name))
        del self.tensor_dict[tensor.name]

    def updateLocals(self):
        self.locals = []
        for node in self.nodes:
            for output in node.outputs:
                if node.type in ['sum', 'add', 'sub', 'mul', 'muladd', 'batch_norm', 'relu', 'leaky_relu', 'softmax']:
                    input = self.tensor_dict[node.inputs[0]]
                    local = IrTensor()
                    local.setName(output)
                    local.setInfo(input.type, input.shape)
                    local.setFormat(input.format)
                    self.addLocal(local)
                elif node.type in ['global_avg_pool']:
                    input = self.tensor_dict[node.inputs[0]]
                    local = IrTensor()
                    local.setName(output)
                    local.setInfo(input.type, [input.shape[0], input.shape[1], 1, 1])
                    local.setFormat(input.format)
                    self.addLocal(local)
                elif node.type in ['conv', 'avg_pool', 'max_pool', 'lrn']:
                    input = self.tensor_dict[node.inputs[0]]
                    pads = node.attr.get('pads')
                    strides = node.attr.get('strides')
                    dilations = node.attr.get('dilations')
                    kernel_shape = node.attr.get('kernel_shape')
                    dim_round_mode = node.attr.get('dim_round_mode')
                    input_shape = input.shape
                    k = input_shape[1]
                    if node.type == 'conv':
                        weight = self.tensor_dict[node.inputs[1]]
                        k = weight.shape[0]
                    round0 = 0
                    round1 = 0
                    if(dim_round_mode == 'ceil'):
                        round0 = strides[0] - 1
                        round1 = strides[1] - 1
                    output_shape = [input_shape[0], k, \
                        (pads[0] + input_shape[2] + pads[2] - ((kernel_shape[0] - 1) * dilations[0] + 1) + round0) // strides[0] + 1, \
                        (pads[1] + input_shape[3] + pads[3] - ((kernel_shape[1] - 1) * dilations[1] + 1) + round1) // strides[1] + 1]
                    local = IrTensor()
                    local.setName(output)
                    local.setInfo(input.type, output_shape)
                    local.setFormat(input.format)
                    self.addLocal(local)
                elif node.type in ['conv_transpose']:
                    input = self.tensor_dict[node.inputs[0]]
                    pads = node.attr.get('pads')
                    strides = node.attr.get('strides')
                    dilations = node.attr.get('dilations')
                    kernel_shape = node.attr.get('kernel_shape')
                    input_shape = input.shape
                    k = self.tensor_shapes[node.inputs[1]][0]
                    output_shape = [input_shape[0], k, \
                        (input_shape[2]-1)*strides[0] + (kernel_shape[0]-1)*dilations[0] + 1 - pads[0] - pads[2], \
                        (input_shape[3]-1)*strides[1] + (kernel_shape[1]-1)*dilations[1] + 1 - pads[1] - pads[3]]
                    local = IrTensor()
                    local.setName(output)
                    local.setInfo(input.type, output_shape)
                    local.setFormat(input.format)
                    self.addLocal(local)
                elif node.type in ['gemm']:
                    A = self.tensor_dict[node.inputs[0]]
                    B = self.tensor_dict[node.inputs[1]]
                    transA = node.attr.get('transA')
                    transB = node.attr.get('transB')
                    shapeA = A.shape
                    shapeB = B.shape
                    if transA == 0 and transB == 0:
                        output_shape = [shapeA[0], shapeB[1], 1, 1]
                    elif transA == 0 and transB != 0:
                        output_shape = [shapeA[0], shapeB[0], 1, 1]
                    elif transA != 0 and transB == 0:
                        output_shape = [shapeA[1], shapeB[1], 1, 1]
                    else:
                        output_shape = [shapeA[1], shapeB[0], 1, 1]
                    local = IrTensor()
                    local.setName(output)
                    local.setInfo(input.type, output_shape)
                    local.setFormat(input.format)
                    self.addLocal(local)
                elif node.type in ['concat']:
                    input = self.tensor_dict[node.inputs[0]]
                    shape = [input.shape[0], 0, input.shape[2], input.shape[3]]
                    for name in node.inputs:
                        lshape = self.tensor_shapes[name]
                        if shape[0:1] + shape[2:] != lshape[0:1] + lshape[2:]:
                            raise ValueError("concat: mismatch detected: " + node.inputs[0] + ":" + str(shape) + " " + name + ":" + str(lshape))
                        shape[1] = shape[1] + lshape[1]
                    local = IrTensor()
                    local.setName(output)
                    local.setInfo(input.type, shape)
                    local.setFormat(input.format)
                    self.addLocal(local)
                elif node.type in ['slice']:
                    input = self.tensor_dict[node.inputs[0]]
                    shape = [input.shape[0], input.shape[1] // len(node.outputs), input.shape[2], input.shape[3]]
                    for name in node.outputs:
                        local = IrTensor()
                        local.setName(name)
                        local.setInfo(input.type, shape)
                        local.setFormat(input.format)
                        self.addLocal(local)
                elif node.type in ['reshape']:
                    input = self.tensor_dict[node.inputs[0]]
                    param = node.attr.get('shape')
                    icount = 1
                    ocount = 1
                    shape = [input.shape[0]]
                    for d in input.shape[1:]:
                        icount = icount * d
                    for d in param:
                        if d > 0:
                            ocount = ocount * d
                    for d in param:
                        if d < 1:
                            d = icount // ocount
                            ocount = ocount * d
                        shape.append(d)
                    if icount != ocount:
                        raise ValueError("reshape: mismatch detected: " + node.inputs[0] + ":" + str(input.shape) + " " + node.outputs[0] + ":" + str(shape))
                    local = IrTensor()
                    local.setName(output)
                    local.setInfo(input.type, shape)
                    local.setFormat(input.format)
                    self.addLocal(local)
                elif node.type in ['transpose']:
                    input = self.tensor_dict[node.inputs[0]]
                    axes = node.attr.get('axes')
                    if input.format == 'NCHW' and axes == [0, 2, 3, 1]:
                        format = 'NHWC'
                    elif input.format == 'NHWC' and axes == [0, 3, 1, 2]:
                        format = 'NCHW'
                    else:
                        raise ValueError("transpose: unsupported transpose: " + input.toString() + " " + str(axes))
                    local = IrTensor()
                    local.setName(output)
                    local.setInfo(input.type, input.shape)
                    local.setFormat(format)
                    self.addLocal(local)
                elif node.type in ['copy']:
                    input = self.tensor_dict[node.inputs[0]]
                    local = IrTensor()
                    local.setName(output)
                    local.setInfo(input.type, input.shape)
                    local.setFormat(input.format)
                    self.addLocal(local)
                elif node.type in ['upsample']:
                    input = self.tensor_dict[node.inputs[0]]
                    zoom_factor = node.attr.get('zoom_factor')
                    if zoom_factor != 2:
                        raise ValueError("upsample: unsupported zoom_factor: " + str(zoom_factor))
                    shape = [input.shape[0], input.shape[1], input.shape[2]*zoom_factor, input.shape[3]*zoom_factor]
                    local = IrTensor()
                    local.setName(output)
                    local.setInfo(input.type, shape)
                    local.setFormat(input.format)
                    self.addLocal(local)
                else:
                    raise ValueError("Unsupported IR node type: {}".format(node.type))

    def removeUnusedTensors(self):
        usedTensorList = []
        for node in self.nodes:
            for tensor in node.inputs:
                usedTensorList.append(tensor)
            for tensor in node.outputs:
                usedTensorList.append(tensor)
        fullTensorList = []
        for name in self.tensor_dict:
            fullTensorList.append(name)
        for name in fullTensorList:
            if not name in usedTensorList:
                self.removeTensor(name)

    def updateBatchSize(self,batchSize):
        for tensor in self.inputs:
            tensor.shape[0] = batchSize
            self.tensor_shapes[tensor.name] = tensor.shape
            self.tensor_dict[tensor.name] = tensor
        for tensor in self.outputs:
            tensor.shape[0] = batchSize
            self.tensor_shapes[tensor.name] = tensor.shape
            self.tensor_dict[tensor.name] = tensor
        for tensor in self.locals:
            tensor.shape[0] = batchSize
            self.tensor_shapes[tensor.name] = tensor.shape
            self.tensor_dict[tensor.name] = tensor

    def convertFp16(self):
        if self.all_F032:
            for tensor in self.inputs:
                tensor.type = 'F016'
                self.tensor_types[tensor.name] = tensor.type
                self.tensor_dict[tensor.name] = tensor
            for tensor in self.outputs:
                tensor.type = 'F016'
                self.tensor_types[tensor.name] = tensor.type
                self.tensor_dict[tensor.name] = tensor
            for tensor in self.locals:
                tensor.type = 'F016'
                self.tensor_types[tensor.name] = tensor.type
                self.tensor_dict[tensor.name] = tensor
            for tensor in self.initializers:
                tensor.type = 'F016'
                self.tensor_types[tensor.name] = tensor.type
                self.tensor_dict[tensor.name] = tensor
            for idx, binary in enumerate(self.binaries):
                weight = np.frombuffer(self.binaries[binary], dtype=np.float32)
                self.addBinary(binary, np.getbuffer(weight.astype(np.float16)))
                #print("Add binary %s of size %d at Idx: %d len: %d" %(binary, len(self.binaries[binary]), idx, len(self.binaries)))
            self.all_F032 = False
            self.all_F016 = True    
        else:
            raise ValueError("The type is alreary Fp16")

    def fuseOps(self):
        tensorReadCount = {}
        for node in self.nodes:
            for name in node.inputs:
                if name in tensorReadCount:
                    tensorReadCount[name] = tensorReadCount[name] + 1
                else:
                    tensorReadCount[name] = 1
        fusedAnOp = True
        while (self.all_F032 or self.all_F016) and fusedAnOp:
            if self.all_F032:
                npType = np.float32
            else:
                npType = np.float16               
            fusedAnOp = False
            prevSkipNode = None
            prevNode = None
            prevOutput = ''
            nodesToRemove = []
            for node in self.nodes:
                # first change batch_norm into muladd
                if node.type == 'batch_norm':
                    scale = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                    offset = np.frombuffer(self.binaries[node.inputs[2]], dtype=npType)
                    #print('scale and offset read: ' + node.inputs[1] + ' ' + node.inputs[2])
                    mean = np.frombuffer(self.binaries[node.inputs[3]], dtype=npType)
                    #print('after mean binary file: ' + node.inputs[3] + '.raw' + 'len: ' + str(len(self.binaries[node.inputs[3]])))
                    variance = np.frombuffer(self.binaries[node.inputs[4]], dtype=npType)
                    epsilon = node.attr.get('epsilon')
                    scale = scale / np.sqrt(variance + epsilon)
                    offset = offset - mean * scale
                    node.type = 'muladd'
                    self.addBinary(node.inputs[1], np.getbuffer(scale))
                    self.addBinary(node.inputs[2], np.getbuffer(offset))
                    node.inputs = node.inputs[:3]
                    node.attr = IrAttr()
                # run through fuse rules
                if prevNode == None:
                    prevSkipNode = None
                    prevNode = node
                    prevOutput = node.outputs[0]
                elif node.type == 'copy':
                    if prevSkipNode != None:
                        prevSkipNode.outputs[0] = node.outputs[0]
                    else:
                        prevNode.outputs[0] = node.outputs[0]
                    prevOutput = node.outputs[0]
                    nodesToRemove.append(node)
                    fusedAnOp = True
                elif node.type == 'transpose':
                    if prevSkipNode != None:
                        prevSkipNode.outputs[0] = node.outputs[0]
                    else:
                        prevNode.outputs[0] = node.outputs[0]
                    prevOutput = node.outputs[0]
                    nodesToRemove.append(node)
                    fusedAnOp = True
                elif (prevNode.type == 'mul' or prevNode.type == 'add' or prevNode.type == 'muladd') \
                        and node.type == 'conv' and prevOutput == node.inputs[0] \
                        and tensorReadCount[prevOutput] == 1:
                    weight_shape = self.tensor_shapes[node.inputs[1]]
                    K = weight_shape[0]
                    N = weight_shape[3] if len(weight_shape) == 2 else np.prod(weight_shape[1:3])
                    if prevNode.type == 'muladd':
                        scale = np.frombuffer(self.binaries[prevNode.inputs[1]], dtype=npType)
                        offset = np.frombuffer(self.binaries[prevNode.inputs[2]], dtype=npType)
                    elif prevNode.type == 'mul':
                        scale = np.frombuffer(self.binaries[prevNode.inputs[1]], dtype=npType)
                        offset = np.repeat(np.array([0], dtype=npType), K)
                    elif prevNode.type == 'add':
                        scale = np.repeat(np.array([1], dtype=npType), K)
                        offset = np.frombuffer(self.binaries[prevNode.inputs[1]], dtype=npType)
                    weight = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                    if len(node.inputs) == 2:
                        bias = np.repeat(np.array([0], dtype=npType), K)
                        tensor = IrTensor()
                        tensor.setName(node.inputs[1] + '__bias')
                        if self.all_F032:
                            tensor.setInfo('F032', [1, K])
                        else:
                            tensor.setInfo('F016', [1, K])
                        self.addVariable(tensor)
                        self.addBinary(tensor.name, np.getbuffer(bias))
                        node.inputs.append(tensor.name)
                    else:
                        bias = np.frombuffer(self.binaries[node.inputs[2]], dtype=npType)
                    bias = bias + offset * np.sum(np.split(weight, K),axis=1)
                    weight = weight * np.repeat(scale, N)
                    self.addBinary(node.inputs[1], np.getbuffer(weight))
                    self.addBinary(node.inputs[2], np.getbuffer(bias))
                    node.inputs[0] = prevNode.inputs[0]
                    nodesToRemove.append(prevNode)
                    prevNode = node
                    prevSkipNode = None
                    prevOutput = node.outputs[0]
                    fusedAnOp = True
                elif prevNode.type == 'conv' and prevOutput == node.inputs[0] \
                        and (node.type == 'mul' or node.type == 'add' or node.type == 'muladd') \
                        and tensorReadCount[prevOutput] == 1:
                    weight_shape = self.tensor_shapes[prevNode.inputs[1]]
                    K = weight_shape[0]
                    N = weight_shape[3] if len(weight_shape) == 2 else np.prod(weight_shape[1:4])
                    if node.type == 'muladd':
                        scale = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                        offset = np.frombuffer(self.binaries[node.inputs[2]], dtype=npType)
                    elif node.type == 'mul':
                        scale = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                        offset = np.repeat(np.array([0], dtype=npType), K)
                    elif node.type == 'add':
                        scale = np.repeat(np.array([1], dtype=npType), K)
                        offset = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                    weight = np.frombuffer(self.binaries[prevNode.inputs[1]], dtype=npType)
                    if len(prevNode.inputs) == 2:
                        bias = np.repeat(np.array([0], dtype=npType), K)
                        tensor = IrTensor()
                        tensor.setName(prevNode.inputs[1] + '__bias')
                        if self.all_F032:
                            tensor.setInfo('F032', [1, K])
                        else:
                            tensor.setInfo('F016', [1, K])
                        self.addVariable(tensor)
                        self.addBinary(tensor.name, np.getbuffer(bias))
                        prevNode.inputs.append(tensor.name)
                    else:
                        bias = np.frombuffer(self.binaries[prevNode.inputs[2]], dtype=npType)
                    bias = bias * scale + offset
                    weight = weight * np.repeat(scale, N)
                    self.addBinary(prevNode.inputs[1], np.getbuffer(weight))
                    self.addBinary(prevNode.inputs[2], np.getbuffer(bias))
                    if prevSkipNode != None:
                        prevSkipNode.outputs[0] = node.outputs[0]
                    else:
                        prevNode.outputs[0] = node.outputs[0]
                    prevOutput = node.outputs[0]
                    nodesToRemove.append(node)
                    fusedAnOp = True
                elif prevSkipNode == None and prevNode.type == 'conv' and \
                     prevOutput == node.inputs[0] and tensorReadCount[prevOutput] == 1 and \
                     (node.type == 'max_pool' or node.type == 'avg_pool' or node.type == 'global_avg_pool'):
                    prevSkipNode = node
                    prevOutput = node.outputs[0]
                elif node.type == 'relu' and \
                     (prevNode.type == 'conv' or prevNode.type == 'max_pool' or \
                     prevNode.type == 'avg_pool' or prevNode.type == 'global_avg_pool'):
                    prevNode.attr.set('mode', 1)
                    if prevSkipNode != None:
                        prevSkipNode.outputs[0] = node.outputs[0]
                    else:
                        prevNode.outputs[0] = node.outputs[0]
                    prevOutput = node.outputs[0]
                    nodesToRemove.append(node)
                    fusedAnOp = True
                elif prevNode.type == 'add' and node.type == 'add' and \
                        prevOutput == node.inputs[0] and tensorReadCount[prevOutput] == 1:
                    ck = np.frombuffer(self.binaries[prevNode.inputs[1]], dtype=npType)
                    offset = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                    offset = offset + ck
                    self.addBinary(node.inputs[1], np.getbuffer(offset))
                    node.inputs[0] = prevNode.inputs[0]
                    nodesToRemove.append(prevNode)
                    prevNode = node
                    prevSkipNode = None
                    prevOutput = node.outputs[0]
                    fusedAnOp = True
                elif prevNode.type == 'add' and node.type == 'mul' and \
                        prevOutput == node.inputs[0] and tensorReadCount[prevOutput] == 1:
                    offset = np.frombuffer(self.binaries[prevNode.inputs[1]], dtype=npType)
                    scale = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                    offset = scale * offset
                    self.addBinary(prevNode.inputs[1], np.getbuffer(offset))
                    node.type = 'muladd'
                    node.inputs.append(prevNode.inputs[1])
                    node.inputs[0] = prevNode.inputs[0]
                    nodesToRemove.append(prevNode)
                    prevNode = node
                    prevSkipNode = None
                    prevOutput = node.outputs[0]
                    fusedAnOp = True
                elif prevNode.type == 'mul' and node.type == 'add' and \
                        prevOutput == node.inputs[0] and tensorReadCount[prevOutput] == 1:
                    scale = self.tensor_shapes[prevNode.inputs[1]]
                    offset = self.tensor_shapes[node.inputs[1]]
                    prevNode.type = 'muladd'
                    prevNode.inputs.append(node.inputs[1])
                    prevNode.outputs[0] = node.outputs[0]
                    prevOutput = node.outputs[0]
                    nodesToRemove.append(node)
                    fusedAnOp = True
                elif prevNode.type == 'mul' and node.type == 'mul' and \
                        prevOutput == node.inputs[0] and tensorReadCount[prevOutput] == 1:
                    mk = np.frombuffer(self.binaries[prevNode.inputs[1]], dtype=npType)
                    scale = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                    scale = scale * mk
                    self.addBinary(node.inputs[1], np.getbuffer(scale))
                    node.inputs[0] = prevNode.inputs[0]
                    nodesToRemove.append(prevNode)
                    prevNode = node
                    prevSkipNode = None
                    prevOutput = node.outputs[0]
                    fusedAnOp = True
                elif prevNode.type == 'add' and node.type == 'muladd' and \
                        prevOutput == node.inputs[0] and tensorReadCount[prevOutput] == 1:
                    ck = np.frombuffer(self.binaries[prevNode.inputs[1]], dtype=npType)
                    scale = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                    offset = np.frombuffer(self.binaries[node.inputs[2]], dtype=npType)
                    offset = offset + scale * ck
                    self.addBinary(node.inputs[2], np.getbuffer(offset))
                    node.inputs[0] = prevNode.inputs[0]
                    nodesToRemove.append(prevNode)
                    prevNode = node
                    prevSkipNode = None
                    prevOutput = node.outputs[0]
                    fusedAnOp = True
                elif prevNode.type == 'mul' and node.type == 'muladd' and \
                        prevOutput == node.inputs[0] and tensorReadCount[prevOutput] == 1:
                    mk = np.frombuffer(self.binaries[prevNode.inputs[1]], dtype=npType)
                    scale = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                    scale = scale * mk
                    self.addBinary(node.inputs[1], np.getbuffer(scale))
                    node.inputs[0] = prevNode.inputs[0]
                    nodesToRemove.append(prevNode)
                    prevNode = node
                    prevSkipNode = None
                    prevOutput = node.outputs[0]
                    fusedAnOp = True
                elif prevNode.type == 'muladd' and node.type == 'add' and \
                        prevOutput == node.inputs[0] and tensorReadCount[prevOutput] == 1:
                    ck = np.frombuffer(self.binaries[prevNode.inputs[2]], dtype=npType)
                    offset = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                    offset = offset + ck
                    self.addBinary(prevNode.inputs[2], np.getbuffer(offset))
                    prevNode.outputs[0] = node.outputs[0]
                    prevOutput = node.outputs[0]
                    nodesToRemove.append(node)
                    fusedAnOp = True
                elif prevNode.type == 'muladd' and node.type == 'mul' and \
                        prevOutput == node.inputs[0] and tensorReadCount[prevOutput] == 1:
                    mk = np.frombuffer(self.binaries[prevNode.inputs[1]], dtype=npType)
                    ck = np.frombuffer(self.binaries[prevNode.inputs[2]], dtype=npType)
                    scale = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                    offset = scale * ck
                    scale = scale * mk
                    self.addBinary(prevNode.inputs[1], np.getbuffer(scale))
                    self.addBinary(prevNode.inputs[2], np.getbuffer(offset))
                    prevNode.outputs[0] = node.outputs[0]
                    prevOutput = node.outputs[0]
                    nodesToRemove.append(node)
                    fusedAnOp = True
                elif prevNode.type == 'muladd' and node.type == 'muladd' and \
                        prevOutput == node.inputs[0] and tensorReadCount[prevOutput] == 1:
                    mk = np.frombuffer(self.binaries[prevNode.inputs[1]], dtype=npType)
                    ck = np.frombuffer(self.binaries[prevNode.inputs[2]], dtype=npType)
                    scale = np.frombuffer(self.binaries[node.inputs[1]], dtype=npType)
                    offset = np.frombuffer(self.binaries[node.inputs[2]], dtype=npType)
                    offset = offset + scale * ck
                    scale = scale * mk
                    self.addBinary(node.inputs[1], np.getbuffer(scale))
                    self.addBinary(node.inputs[2], np.getbuffer(offset))
                    node.inputs[0] = prevNode.inputs[0]
                    nodesToRemove.append(prevNode)
                    prevNode = node
                    prevSkipNode = None
                    prevOutput = node.outputs[0]
                    fusedAnOp = True
                else:
                    prevSkipNode = None
                    prevNode = node
                    prevOutput = node.outputs[0]
            for node in nodesToRemove:
                self.nodes.remove(node)
        self.removeUnusedTensors()

    def sliceGroups(self):
        if self.all_F032:
            npType = np.float32
        else:
            npType = np.float16                       
        for idx, node in enumerate(self.nodes):
            if node.type == 'conv' and node.attr.get('group') > 1:
                input = self.tensor_dict[node.inputs[0]]
                output = self.tensor_dict[node.outputs[0]]
                weight = self.tensor_dict[node.inputs[1]]
                groups = node.attr.get('group')
                C = input.shape[1] // groups
                K = output.shape[1] // groups
                outputShape = [int(v) for v in output.shape]
                inputShape = [int(v) for v in input.shape]
                weightShape = [int(v) for v in weight.shape]
                weightBinary = np.frombuffer(self.binaries[weight.name], dtype=npType).copy().reshape((weightShape[0],weightShape[1],weightShape[2],weightShape[3]))
                outputShape[1] = K
                inputShape[1] = C
                weightShape[0] = K
                bias = None
                if len(node.inputs) > 2:
                    bias = self.tensor_dict[node.inputs[2]]
                    biasShape = [v for v in bias.shape]
                    biasBinary = np.frombuffer(self.binaries[bias.name], dtype=npType).copy().reshape(biasShape[0],biasShape[1])
                    biasShape[1] = K
                joutputs = []
                jinputs = []
                jweights = []
                jbiases = []
                for jgrp in range(groups):
                    # joutput
                    outputName = '%s__grp_%d' % (output.name, jgrp)
                    local = IrTensor()
                    local.setName(outputName)
                    local.setInfo(output.type, outputShape)
                    self.addLocal(local)
                    joutputs.append(outputName)
                    # jinput
                    inputName = '%s__grp_%d' % (input.name, jgrp)
                    local = IrTensor()
                    local.setName(inputName)
                    local.setInfo(input.type, inputShape)
                    self.addLocal(local)
                    jinputs.append(inputName)
                    # jweight
                    weightName = '%s__grp_%d' % (weight.name, jgrp)
                    local = IrTensor()
                    local.setName(weightName)
                    local.setInfo(weight.type, weightShape)
                    self.addVariable(local)
                    jweights.append(weightName)
                    # slice weights binary and add them to dict
                    jweightBinary = weightBinary[jgrp*K:jgrp*K+K,:,:,:].copy()
                    self.addBinary(weightName, np.getbuffer(jweightBinary))
                    if bias is not None:
                        biasName = '%s__grp_%d' % (bias.name, jgrp)
                        local = IrTensor()
                        local.setName(biasName)
                        local.setInfo(bias.type, biasShape)
                        self.addVariable(local)
                        jbiases.append(biasName)
                        # slice bias binary and add them to dict
                        jbiasBinary = biasBinary[:,jgrp*K:jgrp*K+K].copy()
                        self.addBinary(biasName, np.getbuffer(jbiasBinary))
                self.removeTensor(weight.name)
                if bias is not None:
                    self.removeTensor(bias.name)
                for jgrp in reversed(range(groups)):
                    jnode = IrNode()
                    jnode.set('conv', [jinputs[jgrp], jweights[jgrp]] if bias is None else \
                        [jinputs[jgrp], jweights[jgrp], jbiases[jgrp]], [joutputs[jgrp]], node.attr)
                    jnode.attr.set('group', 1)
                    self.nodes.insert(idx, jnode)
                jattr = IrAttr()
                jnode = IrNode()
                jnode.set('slice', [node.inputs[0]], jinputs, jattr)
                self.nodes.insert(idx, jnode)
                node.set('concat', joutputs, [node.outputs[0]], IrAttr())

    def toFile(self,outputFolder):
        if not os.path.isdir(outputFolder):
            os.mkdir(outputFolder)
        irDescFile = outputFolder + '/graph.nnir'
        print('OK: creating IR description in ' + irDescFile + ' ...')
        with open(irDescFile, 'w') as f:
            for tensor in self.inputs:
                f.write('input|' + tensor.toString() + '\n')
            for tensor in self.outputs:
                f.write('output|' + tensor.toString() + '\n')
            for tensor in self.initializers:
                f.write('initializer|' + tensor.toString() + '\n')
            for tensor in self.locals:
                f.write('local|' + tensor.toString() + '\n')
            for node in self.nodes:
                f.write(node.toString() + '\n')
        binaryFolder = outputFolder + '/binary'
        print('OK: creating IR binaries in ' + binaryFolder + ' ...')
        if not os.path.isdir(binaryFolder):
            os.mkdir(binaryFolder)
        for binary in self.binaries:
            binaryFile = binaryFolder + '/' + binary + '.raw'
            with open(binaryFile, 'wb') as f:
                f.write(self.binaries[binary])

    def fromFile(self,inputFolder):
        irDescFile = inputFolder + '/graph.nnir'
        if not os.path.isfile(irDescFile):
            print('ERROR: unable to open: ' + irDescFile)
            sys.exit(1)
        print('OK: reading IR description from ' + irDescFile + ' ...')
        with open(irDescFile, 'r') as f:
            for line in f:
                line = line.strip()
                s = line.split('|')
                if s[0] == 'input':
                    tensor = IrTensor()
                    tensor.fromString(s[1])
                    self.addInput(tensor)
                elif s[0] == 'output':
                    tensor = IrTensor()
                    tensor.fromString(s[1])
                    self.addOutput(tensor)
                elif s[0] == 'initializer':
                    tensor = IrTensor()
                    tensor.fromString(s[1])
                    self.addVariable(tensor)
                elif s[0] == 'local':
                    tensor = IrTensor()
                    tensor.fromString(s[1])
                    self.addLocal(tensor)
                elif s[0] == 'node':
                    node = IrNode()
                    node.fromString(line)
                    self.addNode(node)
                else:
                    raise ValueError("Unsupported IR command: {}".format(s[0]))
        binaryFolder = inputFolder + '/binary'
        print('OK: reading IR binaries from ' + binaryFolder + ' ...')
        for tensor in self.initializers:
            binaryFile = binaryFolder + '/' + tensor.name + '.raw'
            with open(binaryFile, 'rb') as f:
                self.binaries[tensor.name] = f.read()
        self.updateLocals()
        self.removeUnusedTensors()
