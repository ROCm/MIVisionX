import nnef
import math
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
    'linear'                                : 'gemm',
    'softmax'                               : 'softmax',
    'local_response_normalization'          : 'lrn',
    'slice'                                 : 'slice',
    'concat'                                : 'concat',
    'leaky_relu'                            : 'leaky_relu',
    'reshape'                               : 'reshape',
    'squeeze'                               : 'reshape',
    'unsqueeze'                             : 'reshape',
    'transpose'                             : 'transpose',
    'copy'                                  : 'copy'
}
  
nnef2ir_data_type = {
    'float16' : 'F016',
    'float32' : 'F032',
    'float64' : 'F064',
    'uint8'   : 'U008',
    'uint16'  : 'U016',
    'int16'   : 'I016',
    'int32'   : 'I032'
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

def nnef_attr_to_ir_attr(nnef_tensor, nnef_operation):
    nnef_attribs = nnef_operation.attribs
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
                    input_tensor = nnef_tensor[nnef_operation.inputs['input']]
                    if 'filter' in nnef_operation.inputs:
                        filter_tensor = nnef_tensor[nnef_operation.inputs['filter']]
                        f_H = filter_tensor.shape[2]
                        f_W = filter_tensor.shape[3]
                    else:
                        size = nnef_attribs['size']
                        f_H = size[2]
                        f_W = size[3]

                    output_tensor = nnef_tensor[nnef_operation.outputs['output']]
                    temp_stride = nnef_attribs['stride']
                    strides = [temp_stride for temp_stride in nnef_attribs[attrib]]
                    if len(strides) == 4:
                        strides = strides[2:]
                    elif len(strides) == 0:
                        strides = [1, 1]
                    temp_dilation = nnef_attribs['dilation']
                    dilations = [temp_dilation for temp_dilation in nnef_attribs[attrib]]
                    if len(dilations) == 4:
                        dilations = dilations[2:]
                    elif len(dilations) == 0:
                        dilations = [1, 1]
                    in_H = input_tensor.shape[2]
                    in_W = input_tensor.shape[3]
                    out_H = output_tensor.shape[2]
                    out_W = output_tensor.shape[3]
                    s_H = strides[0]
                    s_W = strides[1]
                    d_H = dilations[0]
                    d_W = dilations[1]
                    fd_H = (f_H - 1) * d_H + 1
                    fd_W = (f_W - 1) * d_W + 1
                    tp_H = ((out_H - 1) * s_H + fd_H - in_H) / 2
                    tp_W = ((out_W - 1) * s_W + fd_W - in_W) / 2
                    p_H = (int)(math.floor(tp_H))
                    q_H = (int)(math.ceil(tp_H))
                    p_W = (int)(math.floor(tp_W))
                    q_W = (int)(math.ceil(tp_W))
                    padding = [p_H, q_H, p_W, q_W]

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
            elif attrib == 'groups' and nnef_attribs[attrib] == 0:
                    input_tensor = nnef_tensor[nnef_operation.inputs['input']]
                    attr.set(nnef2ir_attr[attrib], input_tensor.shape[1])
            else:
                attr.set(nnef2ir_attr[attrib], nnef_attribs[attrib])
        else:
            raise ValueError("Unsupported NNEF attribute: {}".format(attrib))
    return attr

def nnef_tensor_to_ir_tensor(nnef_tensor):
    nnir_tensor = IrTensor()
    nnir_tensor.setName(nnef_tensor.name)
    if len(nnef_tensor.shape) == 0:
        nnef_tensor.shape.append(1)
    if hasattr(nnef_tensor.data, 'dtype'):
        if str(nnef_tensor.data.dtype) in nnef2ir_data_type:
            nnir_tensor.setInfo(nnef2ir_data_type[str(nnef_tensor.data.dtype)], nnef_tensor.shape)    
        else:
            raise ValueError("ERROR: Data type {} not supported yet".format(nnef_tensor.data.dtype))
    else:
        nnir_tensor.setInfo('F032', nnef_tensor.shape)    
    return nnir_tensor

def nnef_op_to_ir_node(nnef_graph, nnef_operation):
    global nnef2ir_op_type
    node = IrNode()
    if nnef_operation.name in nnef2ir_op_type:
        type = nnef2ir_op_type[nnef_operation.name]
    else:
        raise ValueError("ERROR: NNEF operation {} not supported yet".format(nnef_operation.name))

    if nnef_operation.name == 'conv':
        filter_tensor = nnef_graph.tensors[nnef_operation.inputs['filter']]
        nnef_operation.attribs.update({'size': [filter_tensor.shape[3], filter_tensor.shape[2]]})

    if nnef_operation.name == 'matmul':
        nnef_operation.attribs.update({'beta': 0.0})
    
    if nnef_operation.name == 'squeeze':
        input_shape = nnef_graph.tensors[nnef_operation.inputs['input']].shape
        axes = nnef_operation.attribs['axes']
        output_shape = [input_shape[i] for i in range(len(input_shape)) if i not in axes]
        del nnef_operation.attribs['axes']
        nnef_operation.attribs.update({'shape': output_shape})
    
    if nnef_operation.name == 'unsqueeze':
        input_shape = nnef_graph.tensors[nnef_operation.inputs['input']].shape
        axes = nnef_operation.attribs['axes']
        output_shape = input_shape
        if len(output_shape) < 4:
            for i in range(len(axes)):
                output_shape.insert(axes[i], 1)
        del nnef_operation.attribs['axes']
        nnef_operation.attribs.update({'shape': output_shape})

    if nnef_operation.name == 'linear':
        nnef_operation.attribs.update({'transposeB': 1})
        
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
    
    node.set(type, inputs, outputs, nnef_attr_to_ir_attr(nnef_graph.tensors, nnef_operation))

    return node
    
def nnef_graph_to_ir_graph(nnef_graph):

    graph = IrGraph()

    scalarCount = 0
    hasFP64 = False
    hasINT32 = False

    for operation in nnef_graph.operations:
        if operation.name == 'external':
            continue
        # add variable(s)
        if operation.name == 'variable':
            tensor_name = operation.outputs['output']
            tensor = nnef_graph.tensors[tensor_name]
            dtype = tensor.data.dtype
            if dtype == 'float64':
                hasFP64 = True
            elif dtype == 'int32':
                hasINT32 = True
            graph.addVariable(nnef_tensor_to_ir_tensor(tensor))
            graph.addBinary(tensor_name, tensor.data)
        else:
            for name in operation.outputs:
                output_tensor = nnef_graph.tensors[operation.outputs[name]]
                graph.addLocal(nnef_tensor_to_ir_tensor(output_tensor))
                # handle 0-d tensor(s)
                for input in operation.inputs:
                    if isinstance(operation.inputs[input], float):
                        scalarCount += 1
                        scalar_name = 'scalar_' + str(scalarCount)
                        if input == 'bias':
                            filter_tensor = nnef_graph.tensors[operation.inputs['filter']]
                            shape = [1, filter_tensor.shape[0]]
                        elif input == 'x' or input == 'y' or input == 'z':                            
                            shape = output_tensor.shape
                            if len(output_tensor.shape) == 1:
                                shape = [1, output_tensor.shape[0]]
                            while len(shape) < 4:
                                shape.append(1)
                        else:
                            shape = output_tensor.shape
                        scalar_tensor = IrTensor()
                        scalar_tensor.setName(scalar_name)
                        scalar_tensor.setInfo('F032', shape)
                        tensor_data = np.full(shape, operation.inputs[input], dtype=np.float32)
                        graph.addVariable(scalar_tensor)                    
                        graph.addBinary(scalar_name, tensor_data)
                        operation.inputs[input] = scalar_name

                node = nnef_op_to_ir_node(nnef_graph, operation)
                graph.addNode(node)

    # add input(s)
    for tensor_name in nnef_graph.inputs:
        tensor = nnef_graph.tensors[tensor_name]
        graph.addInput(nnef_tensor_to_ir_tensor(tensor))

    # add output(s)
    for tensor_name in nnef_graph.outputs:
        tensor = nnef_graph.tensors[tensor_name]
        while len(tensor.shape) < 4:
            tensor.shape.append(1)
        graph.addOutput(nnef_tensor_to_ir_tensor(tensor))

    graph.updateLocals()

    if hasFP64:
        print('This nnir graph contains float64 bit tensors. Quantizing to float32 bit tensors ...')
        graph.convertFp32()
    
    if hasINT32:
        print('This nnir graph contains int32 bit tensors. Converting to float32 bit tensors ...')
        graph.convertFp32()
    return graph

def nnef2ir(inputFolder, outputFolder):
    nnef_graph = nnef.load_graph(inputFolder)
    nnef.infer_shapes(nnef_graph)
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

