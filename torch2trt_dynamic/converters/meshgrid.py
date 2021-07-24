import torch
from torch2trt_dynamic.torch2trt_dynamic import (tensor_trt_get_shape_trt,
                                                 tensorrt_converter, trt_)

from .repeat import _convert_repeat_impl


@tensorrt_converter('torch.meshgrid')
def convert_meshgrid(ctx):
    input_list = ctx.method_args
    output = ctx.method_return

    num_inputs = len(input_list)
    input_trt_list = [
        trt_(ctx.network, input_tensor) for input_tensor in input_list
    ]
    input_shape_trt_list = [
        tensor_trt_get_shape_trt(ctx.network, input_trt)
        for input_trt in input_trt_list
    ]

    output_shape_trt = ctx.network.add_concatenation(
        input_shape_trt_list).get_output(0)

    one_trt = trt_(ctx.network, torch.ones(1, dtype=torch.int32))
    for index, input_trt in enumerate(input_trt_list):
        shuffle_shape_trt = [one_trt] * index
        shuffle_shape_trt += [input_shape_trt_list[index]]
        shuffle_shape_trt += [one_trt] * (num_inputs - 1 - index)
        shuffle_shape_trt = \
            ctx.network.add_concatenation(shuffle_shape_trt).get_output(0)
        layer = ctx.network.add_shuffle(input_trt)
        layer.set_input(1, shuffle_shape_trt)
        input_trt_list[index] = layer.get_output(0)

    for input_trt, out in zip(input_trt_list, output):
        out._trt = _convert_repeat_impl(ctx, input_trt, output_shape_trt)
