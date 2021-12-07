import sys
from amd.rali import readers
from amd.rali import decoders
import inspect
import amd.rali.types as types
import rali_pybind as b
from amd.rali.pipeline import Pipeline


def fish_eye(*inputs, device=None):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False}
    fisheye_image = b.FishEye(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (fisheye_image)

def brightness(*inputs, brightness=1.0, bytes_per_sample_hint=0, image_type=0,
               preserve=False, seed=-1, device=None):

    kwargs_pybind = {"input_image0": inputs[0],
                     "is_output": False, "alpha": None, "beta": None}
    brightness_image = b.Brightness(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (brightness_image)

def resize(*inputs, bytes_per_sample_hint=0, image_type=0, interp_type=1, mag_filter= 1, max_size = [0.0, 0.0], min_filter = 1,
            minibatch_size=32, preserve=False, resize_longer=0.0, resize_shorter= 0.0, resize_x = 0.0, resize_y = 0.0,
            save_attrs=False, seed=1, temp_buffer_hint=0, device = None):

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "dest_width": resize_x, "dest_height": resize_y,
                     "is_output": False}
    resized_image = b.Resize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (resized_image)