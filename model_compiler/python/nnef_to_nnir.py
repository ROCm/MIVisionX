import os, sys
import nnef
from nnir import *

def generateGraph(graph, outputFolder, label):
    fileName = outputFolder + '/graph.nnir'
    print('creating ' + fileName + '...')
    # with open(fileName, 'wb') as f:
    #     f.write(
            
    #     )

def generateBinaries(inputFolder, outputFolder, ops):
    binaryFolder = outputFolder + '/' + label
    print('creating variables in ' + binaryFolder + ' ...')
    if not os.path.isdir(binaryFolder):
        os.mkdir(binaryFolder)
    #for tensor in ////:
    #    fileName = binaryFolder + '/' + tensor.name + '.raw'
    #    with open(fileName, 'wb') as f:
    #        binary = nnef._read_tensor_provisional(tensor)
    #        f.write(binary)

def nnef_tensor_info_to_date(item):
    tensor = IrTensor()
    tensor.setName(item)
    tensor.setInfo()

def nnef_graph_to_ir_graph(inputFolder, attrs, ops):
    binaryFolder = inputFolder + '/binary'
    graph = IrGraph()
    initializerList = []
    for proto, args in ops:
        if proto.name == 'variable':
            inputs, attribs, outputs, dtype = nnef.split_args(args, params=proto.params, results=proto.results, split_attribs=True)
            for idx, item in enumerate(outputs.values()):
                initializerList.append(item)
                print(item + proto.name)
                #graph.addVariable()
                fileName = binaryFolder + '/' + item + '.dat'
                binary = nnef._read_tensor_provisional(fileName)
                graph.addBinary(item, binary)

    graph.updateLocals()
    return graph

def nnef2ir(inputFolder, outputFolder):
    attrs, ops = nnef.parse_file(inputFolder + 'graph.nnef')
    #attrs, ops = nnef.parse_file('/home/hansel/Hansel/NNEF-Tools/parser/cpp/examples/resnet_flat.txt')
    #generateBinaries(inputFolder, outputFolder, ops)
    graph = nnef_graph_to_ir_graph(inputFolder, attrs, ops)
    graph.toFile(output_folder)

def main ():
    if len(sys.argv) < 3:
        print('Usage: python nnef2nnir.py <nnefInputFolder> <outputFolder>')
        sys.exit(1)
    inputFolder = sys.argv[1]
    outputFolder = sys.argv[2]
    print('reading NNEF model from ' + inputFolder + '...')
    nnef2ir(inputFolder, outputFolder)
    #file1 = open(inputFolder + 'binary/conv1_b.dat', "r")
    #data = nnef._read_tensor_provisional(file1)
    #nnef.format_graph(inputFolder + 'graph.nnef')
    #attribs, ops = nnef.parse_file(input = inputFolder + 'graph.nnef')
    #print(data)
    #fileName = 'ex2.raw'
    #with open(fileName, 'wb') as f:
    #    f.write(data)
    #attribs, ops = nnef.parse_string(input = inputFolder + 'graph.nnef')


if __name__ == '__main__':
    main()

