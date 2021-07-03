import tensorrt as trt
import torch

from torch2trt_dynamic.torch2trt_dynamic import (get_arg, slice_shape_trt,
                                                 tensor_trt_get_shape_trt,
                                                 tensorrt_converter, trt_)


def _unsqueeze_input(ctx, input_trt, dim):
    ones_trt = trt_(ctx.network,
                    torch.ones(dim - len(input_trt.shape), dtype=torch.int32))
    input_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt)
    input_shape_trt = ctx.network.add_concatenation(
        [ones_trt, input_shape_trt]).get_output(0)
    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, input_shape_trt)
    input_trt = layer.get_output(0)
    return input_trt


def _convert_repeat_impl(ctx, input_trt, output_shape_trt):
    dim = output_shape_trt.shape[0]

    if len(input_trt.shape) < dim:
        input_trt = _unsqueeze_input(ctx, input_trt, dim)

    zeros_trt = trt_(ctx.network, torch.zeros(dim, dtype=torch.int32))
    ones_trt = trt_(ctx.network, torch.ones(dim, dtype=torch.int32))

    layer = ctx.network.add_slice(input_trt, [0] * dim, [1] * dim, [1] * dim)
    layer.set_input(1, zeros_trt)
    layer.set_input(2, output_shape_trt)
    layer.set_input(3, ones_trt)
    layer.mode = trt.SliceMode.WRAP

    output_trt = layer.get_output(0)

    return output_trt


@tensorrt_converter('torch.Tensor.repeat')
def convert_repeat(ctx):
    input = ctx.method_args[0]
    repeats = ctx.method_args[1]
    if isinstance(repeats, int):
        repeats = ctx.method_args[1:]

    output = ctx.method_return

    input_trt = trt_(ctx.network, input)

    # compute output shape
    input_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt)
    repeat_times_trt = [trt_(ctx.network, rep) for rep in repeats]
    repeat_times_trt = ctx.network.add_concatenation(
        repeat_times_trt).get_output(0)

    output_shape_trt = ctx.network.add_elementwise(
        input_shape_trt, repeat_times_trt,
        trt.ElementWiseOperation.PROD).get_output(0)

    # convert repeat
    output_trt = _convert_repeat_impl(ctx, input_trt, output_shape_trt)

    output._trt = output_trt


@tensorrt_converter('torch.Tensor.expand')
def convert_expand(ctx):
    input = ctx.method_args[0]
    if isinstance(ctx.method_args[1], int):
        sizes = ctx.method_args[1:]
    else:
        sizes = ctx.method_args[1]

    output = ctx.method_return

    input_trt = trt_(ctx.network, input)

    dim = len(sizes)

    # unsqueeze if necessary
    if len(input_trt.shape) < dim:
        input_trt = _unsqueeze_input(ctx, input_trt, dim)
    input_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt)

    # compute output shape
    output_shape_trt = []
    for i, s in enumerate(sizes):
        if s > 0:
            output_shape_trt.append(trt_(ctx.network, s))
        else:
            output_shape_trt.append(
                slice_shape_trt(ctx.network, input_shape_trt, i, 1))

    output_shape_trt = ctx.network.add_concatenation(
        output_shape_trt).get_output(0)

    # convert repeat
    output_trt = _convert_repeat_impl(ctx, input_trt, output_shape_trt)

    output._trt = output_trt


@tensorrt_converter('torch.Tensor.expand_as')
def convert_expand_as(ctx):
    input = ctx.method_args[0]
    other = get_arg(ctx, 'other', pos=1, default=None)

    input_trt = trt_(ctx.network, input)
    other_trt = trt_(ctx.network, other)
    output = ctx.method_return

    # compute output shape
    output_shape_trt = tensor_trt_get_shape_trt(ctx.network, other_trt)

    # convert repeat
    output_trt = _convert_repeat_impl(ctx, input_trt, output_shape_trt)

    output._trt = output_trt
