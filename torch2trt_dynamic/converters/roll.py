import tensorrt as trt
import torch

from torch2trt_dynamic.torch2trt_dynamic import (get_arg, slice_shape_trt,
                                                 tensor_trt_get_shape_trt,
                                                 tensorrt_converter, trt_)


@tensorrt_converter('torch.roll')
@tensorrt_converter('torch.Tensor.roll')
def convert_roll(ctx):
    x = ctx.method_args[0]
    shifts = get_arg(ctx, 'shifts', pos=1, default=None)
    dims = get_arg(ctx, 'dims', pos=2, default=None)
    output = ctx.method_return

    input_dim = x.dim()
    input_trt = trt_(ctx.network, x)
    input_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt)

    if dims is not None:
        dims = [int((input_dim + dim) % input_dim) for dim in dims]

    # if dims is None, the output should be flatten
    need_flatten = (dims is None)
    if need_flatten:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (-1, )
        input_trt = layer.get_output(0)
        dims = (0, )

    zero_trt = trt_(ctx.network, 0)
    slice_step_trt = trt_(ctx.network,
                          torch.ones((input_dim, ), dtype=torch.int32))
    for shift, dim in zip(shifts, dims):
        shift_trt = trt_(ctx.network, shift)
        dim_size_trt = slice_shape_trt(ctx.network, input_shape_trt, dim, 1)
        assert dim_size_trt is not None

        if shift < 0:
            # shift could be negitive, make it positive
            # shift = shift + dim_size
            shift_trt = ctx.network.add_elementwise(
                shift_trt, dim_size_trt,
                trt.ElementWiseOperation.SUM).get_output(0)

        shift_trt = ctx.network.add_elementwise(
            dim_size_trt, shift_trt,
            trt.ElementWiseOperation.SUB).get_output(0)

        slice_0_start_trt = []
        slice_0_size_trt = []
        slice_1_start_trt = []
        slice_1_size_trt = []
        if dim > 0:
            pre_dim_start_trt = trt_(ctx.network,
                                     torch.zeros((dim, ), dtype=torch.int32))
            pre_dim_size_trt = slice_shape_trt(ctx.network, input_shape_trt, 0,
                                               dim)
            slice_0_start_trt.append(pre_dim_start_trt)
            slice_0_size_trt.append(pre_dim_size_trt)
            slice_1_start_trt.append(pre_dim_start_trt)
            slice_1_size_trt.append(pre_dim_size_trt)

        slice_1_remain_trt = ctx.network.add_elementwise(
            dim_size_trt, shift_trt,
            trt.ElementWiseOperation.SUB).get_output(0)
        slice_0_start_trt.append(zero_trt)
        slice_0_size_trt.append(shift_trt)
        slice_1_start_trt.append(shift_trt)
        slice_1_size_trt.append(slice_1_remain_trt)
        if dim < input_dim - 1:
            post_dim_start_trt = trt_(
                ctx.network,
                torch.zeros((input_dim - dim - 1, ), dtype=torch.int32))
            post_dim_size_trt = slice_shape_trt(ctx.network, input_shape_trt,
                                                dim + 1, input_dim - dim - 1)
            slice_0_start_trt.append(post_dim_start_trt)
            slice_0_size_trt.append(post_dim_size_trt)
            slice_1_start_trt.append(post_dim_start_trt)
            slice_1_size_trt.append(post_dim_size_trt)

        slice_0_start_trt = ctx.network.add_concatenation(
            slice_0_start_trt).get_output(0)
        slice_0_size_trt = ctx.network.add_concatenation(
            slice_0_size_trt).get_output(0)
        slice_1_start_trt = ctx.network.add_concatenation(
            slice_1_start_trt).get_output(0)
        slice_1_size_trt = ctx.network.add_concatenation(
            slice_1_size_trt).get_output(0)

        layer = ctx.network.add_slice(input_trt, input_dim * [0],
                                      input_dim * [1], input_dim * [1])

        layer.set_input(1, slice_0_start_trt)
        layer.set_input(2, slice_0_size_trt)
        layer.set_input(3, slice_step_trt)
        slice_0_trt = layer.get_output(0)

        layer = ctx.network.add_slice(input_trt, input_dim * [0],
                                      input_dim * [1], input_dim * [1])

        layer.set_input(1, slice_1_start_trt)
        layer.set_input(2, slice_1_size_trt)
        layer.set_input(3, slice_step_trt)
        slice_1_trt = layer.get_output(0)

        layer = ctx.network.add_concatenation([slice_1_trt, slice_0_trt])
        layer.axis = dim
        input_trt = layer.get_output(0)

    # recover from flatten if needed
    if need_flatten:
        layer = ctx.network.add_shuffle(input_trt)
        layer.set_input(1, input_shape_trt)
        input_trt = input_trt

    output._trt = input_trt
