import os, sys
import nnef
import numpy as np
from collections import OrderedDict
from nnir import *

nnef2ir_attr = {
    'axis' : 'axis',
    'axes' : 'axes',
    'size' : 'kernel_shape',
    'padding' : 'pads',
    'stride' : 'strides',
    'dilation' : 'dilations',
    'groups' : 'group',
    'epsilon' : 'epsilon',
    'alpha' : 'alpha',
    'beta' : 'beta',
    'transposeA' : 'transA',
    'transposeB' : 'transB',
    'bias' : 'bias',
    'border' : 'border_mode',
    'shape' : 'shape',
    'offset' : 'offset'
}

nnef2ir_op_type = {
    'conv'                                  : 'conv',
    'deconv'                                : 'conv_transpose',
    'batch_normalization'                   : 'batch_norm',
    'avg_pool'                              : 'avg_pool',
    'max_pool'                              : 'max_pool',
    'mean_reduce'                           : 'global_avg_pool',
    'relu'                                  : 'relu',
    'add'                                   : 'add',
    'mul'                                   : 'mul',
    'sub'                                   : 'sub',
    'matmul'                                : 'gemm',
    'softmax'                               : 'softmax',
    'local_response_normalization'          : 'lrn',
    'slice'                                 : 'slice',
    'concat'                                : 'concat',
    'leaky_relu'                            : 'leaky_relu',
    'reshape'                               : 'reshape',
    'transpose'                             : 'transpose',
    'copy'                                  : 'copy'
}

def flatten(nested_list):
    flatten_list = []
    for values in nested_list:
        if isinstance(values,list): 
            flatten_list.extend(flatten(values))
        else: 
            flatten_list.append(values)
    return flatten_list

def nnef_name_to_ir_name(nnef_name):
    return '_'.join(('_'.join(nnef_name.split('/')).split('-')))

def nnef_attr_to_ir_attr(nnef_attribs):
    global nnef2ir_attr
    attr = IrAttr()
    for attrib in nnef_attribs:
        if attrib in nnef2ir_attr:
            if attrib == 'size':
                size = [size for size in nnef_attribs[attrib]]
                if len(size) == 4:
                    size = size[2:]
                attr.set(nnef2ir_attr[attrib], size)  
            elif attrib == 'padding':
                padding = [pad for pads in nnef_attribs[attrib] for pad in pads]
                if len(padding) == 8:
                    padding = padding[4:]    
                elif len(padding) == 0:
                    padding = [0, 0, 0, 0]

                new_padding = [padding[2], padding[0], padding[3], padding[1]]
                attr.set(nnef2ir_attr[attrib], new_padding)  
            elif attrib == 'stride':
                stride = [stride for stride in nnef_attribs[attrib]]
                if len(stride) == 4:
                    stride = stride[2:]
                elif len(stride) == 0:
                    stride = [1, 1]
                attr.set(nnef2ir_attr[attrib], stride)  
            elif attrib == 'dilation':
                dilation = [dilation for dilation in nnef_attribs[attrib]]
                if len(dilation) == 4:
                    dilation = dilation[2:]
                elif len(dilation) == 0:
                    dilation = [1, 1]
                attr.set(nnef2ir_attr[attrib], dilation)  
            elif attrib == 'transposeA' or attrib == 'transposeB':
                if nnef_attribs[attrib]:
                    attr.set(nnef2ir_attr[attrib], 1)
                else:
                    attr.set(nnef2ir_attr[attrib], 0)
            else:
                attr.set(nnef2ir_attr[attrib], nnef_attribs[attrib])
        else:
            raise ValueError("Unsupported NNEF attribute: {}".format(attrib))
    return attr

def nnef_tensor_to_ir_tensor(nnef_tensor):
    nnir_tensor = IrTensor()
    nnir_tensor.setName(nnef_tensor.name)
    nnir_tensor.setInfo('F032', nnef_tensor.shape)
    return nnir_tensor

def nnef_op_to_ir_node(nnef_graph, nnef_operation):
    global nnef2ir_op_type
    node = IrNode()
    if nnef_operation.name in nnef2ir_op_type:
        type = nnef2ir_op_type[nnef_operation.name]
    else:
        print('ERROR: NNEF operation "%s" not supported yet' % (nnef_operation.name))
        sys.exit(1)

    if nnef_operation.name == 'conv':
        bias = nnef_operation.inputs['bias']
        if bias == 0.0:
            del nnef_operation.inputs['bias']

        filter_tensor = nnef_graph.tensors[nnef_operation.inputs['filter']]
        nnef_operation.attribs.update({'size': [filter_tensor.shape[3], filter_tensor.shape[2]]})

    if nnef_operation.name == 'matmul':
        nnef_operation.attribs.update({'beta': 0.0})
        
    inputs = [nnef_operation.inputs[nnef_name_to_ir_name(name)] for name in nnef_operation.inputs]
    outputs = [nnef_operation.outputs[nnef_name_to_ir_name(name)] for name in nnef_operation.outputs]    
    
    if nnef_operation.name == 'batch_normalization':
        input_tensor = nnef_operation.inputs['input']
        variance = nnef_operation.inputs['variance']
        scale = nnef_operation.inputs['scale']
        mean = nnef_operation.inputs['mean']
        offset = nnef_operation.inputs['offset']
        inputs = [input_tensor, scale, offset, mean, variance]
        
    # flatten the lists
    inputs = flatten(inputs)
    outputs = flatten(outputs)
    
    node.set(type, inputs, outputs, nnef_attr_to_ir_attr(nnef_operation.attribs))

    return node
    
def nnef_graph_to_ir_graph(nnef_graph):

    graph = IrGraph()
    
    # add input(s)
    for tensor_name in nnef_graph.inputs:
        tensor = nnef_graph.tensors[tensor_name]
        graph.addInput(nnef_tensor_to_ir_tensor(tensor))

    # add output(s)
    for tensor_name in nnef_graph.outputs:
        tensor = nnef_graph.tensors[tensor_name]
        graph.addOutput(nnef_tensor_to_ir_tensor(tensor))

    for operation in nnef_graph.operations:
        if operation.name == 'external':
            continue
        # add variable(s)
        if operation.name == 'variable':
            tensor_name = operation.outputs['output']
            tensor = nnef_graph.tensors[tensor_name]
            graph.addVariable(nnef_tensor_to_ir_tensor(tensor))
            graph.addBinary(tensor_name, tensor.data)
        else:
            # add node(s)
            for name in operation.outputs:
                tensor = nnef_graph.tensors[operation.outputs[name]]
                graph.addLocal(nnef_tensor_to_ir_tensor(tensor))
                node = nnef_op_to_ir_node(nnef_graph, operation)
                graph.addNode(node)
    graph.updateLocals()
    return graph

def nnef2ir(inputFolder, outputFolder):
    nnef_graph = nnef.load_model(inputFolder)
    graph = nnef_graph_to_ir_graph(nnef_graph)
    graph.toFile(outputFolder)

def main ():
    if len(sys.argv) < 3:
        print('Usage: python nnef_to_nnir.py <nnefInputFolder> <outputFolder>')
        sys.exit(1)
    inputFolder = sys.argv[1]
    outputFolder = sys.argv[2]

    print('reading NNEF model from ' + inputFolder + '...')
    nnef2ir(inputFolder, outputFolder)
    print('Done')
    
if __name__ == '__main__':
    main()

