import rocal_pybind as b


def coin_flip(*inputs,probability=0.5, device=None):
    values = [0, 1]
    frequencies = [1-probability, probability]
    output_array = b.CreateIntRand(values, frequencies)
    return output_array

def uniform(*inputs,range=[-1, 1], device=None):
    output_param = b.CreateFloatUniformRand(range[0], range[1])
    return output_param
