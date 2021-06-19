import numpy as np
import tensorrt as trt

from ..torch2trt_dynamic import (tensor_trt_get_shape_trt, tensorrt_converter,
                                 torch_dim_to_trt_axes, trt_)


@tensorrt_converter('torch.nn.LayerNorm.forward')
def convert_LayerNorm(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]

    normalized_shape = module.normalized_shape
    weight = module.weight
    bias = module.bias
    eps = module.eps

    output = ctx.method_return

    eps_np = np.array([eps], dtype=np.float32)
    keep_dims = True

    input_trt = trt_(ctx.network, input)

    if len(input.shape) == 3:
        input_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt)
        new_input_shape_trt = ctx.network.add_concatenation(
            [trt_(ctx.network, 1), input_shape_trt]).get_output(0)
        layer = ctx.network.add_shuffle(input_trt)
        layer.set_input(1, new_input_shape_trt)
        input_trt = layer.get_output(0)

    reduce_axes = torch_dim_to_trt_axes(
        tuple(
            range(
                len(input_trt.shape) - len(normalized_shape),
                len(input_trt.shape))))

    mean_trt = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG,
                                      reduce_axes, keep_dims).get_output(0)

    # compute variance over spatial (include eps, to reduce layer count)
    delta_trt = ctx.network.add_elementwise(
        input_trt, mean_trt, trt.ElementWiseOperation.SUB).get_output(0)

    var_trt = ctx.network.add_scale(delta_trt, trt.ScaleMode.UNIFORM,
                                    np.zeros_like(eps_np),
                                    np.ones_like(eps_np),
                                    2 * np.ones_like(eps_np)).get_output(0)
    var_trt = ctx.network.add_reduce(var_trt, trt.ReduceOperation.AVG,
                                     reduce_axes, keep_dims).get_output(0)

    # compute sqrt(var + eps)
    var_trt = ctx.network.add_scale(var_trt, trt.ScaleMode.UNIFORM, eps_np,
                                    np.ones_like(eps_np),
                                    0.5 * np.ones_like(eps_np)).get_output(0)

    # compute final result
    result_trt = ctx.network.add_elementwise(
        delta_trt, var_trt, trt.ElementWiseOperation.DIV).get_output(0)

    if len(input.shape) == 3:
        layer = ctx.network.add_shuffle(result_trt)
        layer.set_input(1, input_shape_trt)
        result_trt = layer.get_output(0)

    if weight is not None:
        assert weight.ndim <= input.ndim
        while weight.ndim < input.ndim:
            weight = weight.unsqueeze(0)
        weight_trt = trt_(ctx.network, weight)
        layer = ctx.network.add_elementwise(result_trt, weight_trt,
                                            trt.ElementWiseOperation.PROD)
        result_trt = layer.get_output(0)

    if bias is not None:
        assert bias.ndim <= input.ndim
        while bias.ndim < input.ndim:
            bias = bias.unsqueeze(0)
        bias_trt = trt_(ctx.network, bias)
        layer = ctx.network.add_elementwise(result_trt, bias_trt,
                                            trt.ElementWiseOperation.SUM)
        result_trt = layer.get_output(0)

    output._trt = result_trt
