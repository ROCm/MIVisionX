

class Node:
    def __init__(self):
        self.prev = []
        self.next = []

        self.node_name = None
        self.submodule_name = None
        self.rali_c_func_call = None
        self.kwargs_pybind = None
        self.kwargs = None
        self.is_output = False
        self.has_output_image = False
        self.has_input_image = False
        self.CMN = False # Set True only for CMN Node
        self.output_image = None # To store the output image
        self.visited = False # To know if we have already visited the Node
        self.augmentation_node = False #When It has an image output <reader nodes are an exception>
        self.input_image = []

    def __repr__(self):
        return '{} {}'.format(self.__class__.__name__, self.node_name)

    def set_output_image(self, image):
        self.output_image = image

    def set_visited(self, bool_value):
        self.visited = bool_value

def add_node(prev_node,current_node):
    prev_node.next.append(current_node)
    current_node.prev.append(prev_node)

def add_meta_data_node(prev_node,current_node):
    prev_node.next.current_node
    current_node.prev.prev_node

#special nodes <for handling meta data>
class MetaDataNode:
    def __init__(self):
        self.prev = None
        self.next = None

        self.node_name = "OneHotLabel"
        self.bool_variable = "_oneHotEncoding"
        self.kwargs = None
        self.kwargs_pybind = None
        self.visited = False # To know if we have already visited the Node
        self.has_rali_c_func_call = False
        self.rali_c_func_call = None

    def __repr__(self):
        return '{} {}'.format(self.__class__.__name__, self.node_name)

    def set_visited(self, bool_value):
        self.visited = bool_value
