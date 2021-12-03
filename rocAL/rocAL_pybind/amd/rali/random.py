from amd.rali.global_cfg import Node, add_node
import rali_pybind as b


def coin_flip(*inputs,probability=0.5, device=None):
    values = [0, 1]
    frequencies = [1-probability, probability]
    output_array = b.CreateIntRand(values, frequencies)
    return output_array
