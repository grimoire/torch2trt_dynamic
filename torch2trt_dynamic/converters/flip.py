import torch
import tensorrt as trt

from ..torch2trt_dynamic import (get_arg, slice_shape_trt,
                                 tensor_trt_get_shape_trt, tensorrt_converter,
                                 trt_)


@tensorrt_converter('torch.flip')
@tensorrt_converter('torch.Tensor.flip')
def convert_flip(ctx):
    input = ctx.method_args[0]
    dims = get_arg(ctx, 'dims', pos=1, default=0)
    if isinstance(dims, int):
        dims = ctx.method_args[1:]

    input_dim = len(input.shape)
    dims = [input_dim + dim if dim < 0 else dim for dim in dims]

    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    input_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt)

    zero_trt = trt_(ctx.network, input.new_zeros(1, dtype=torch.int32))
    one_trt = trt_(ctx.network, input.new_ones(1, dtype=torch.int32))
    minus_one_trt = trt_(ctx.network,
                         -1 * input.new_ones(1, dtype=torch.int32))
    starts_trt = [zero_trt for _ in range(input_dim)]
    steps_trt = [one_trt for _ in range(input_dim)]

    for d in dims:
        tmp_slice_trt = slice_shape_trt(ctx.network, input_shape_trt, d, 1)
        starts_trt[d] = ctx.network.add_elementwise(
            tmp_slice_trt, one_trt, trt.ElementWiseOperation.SUB).get_output(0)
        steps_trt[d] = minus_one_trt

    starts_trt = ctx.network.add_concatenation(starts_trt).get_output(0)
    steps_trt = ctx.network.add_concatenation(steps_trt).get_output(0)

    layer = ctx.network.add_slice(input_trt, [0] * input_dim, [1] * input_dim,
                                  [0] * input_dim)
    layer.set_input(1, starts_trt)
    layer.set_input(2, input_shape_trt)
    layer.set_input(3, steps_trt)

    output._trt = layer.get_output(0)
