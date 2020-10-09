import tensorrt as trt
from ..torch2trt_dynamic import *
# from torch2trt.plugins import *


# @tensorrt_converter('torch.nn.LayerNorm.forward')
# def convert_LayerNorm(ctx):
#     module = ctx.method_args[0]
    
#     normalized_shape = module.normalized_shape
#     weight = module.weight.detach().cpu().numpy()
#     bias = module.bias.detach().cpu().numpy()
#     eps = module.eps

#     input = ctx.method_args[1]
#     input_trt = trt_(ctx.network, input)
#     output = ctx.method_return

#     plugin = create_layernorm_plugin("layernorm_" + str(id(module)),
#                                      normalized_shape=normalized_shape,
#                                      W=weight,
#                                      B=bias,
#                                      eps=eps
#                                      )

#     custom_layer = ctx.network.add_plugin_v2(
#         inputs=[input_trt], plugin=plugin)

#     output._trt = custom_layer.get_output(0)


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
    reduce_axes = torch_dim_to_trt_axes(tuple(range(input.ndim - len(normalized_shape), input.ndim)))

    mean_trt = ctx.network.add_reduce(input._trt, trt.ReduceOperation.AVG, reduce_axes, keep_dims).get_output(0)

    # compute variance over spatial (include eps, to reduce layer count)
    delta_trt = ctx.network.add_elementwise(input._trt, mean_trt, trt.ElementWiseOperation.SUB).get_output(0)
    var_trt = ctx.network.add_scale(delta_trt, trt.ScaleMode.UNIFORM, np.zeros_like(eps_np), np.ones_like(eps_np), 2 * np.ones_like(eps_np)).get_output(0)
    var_trt = ctx.network.add_reduce(var_trt, trt.ReduceOperation.AVG, reduce_axes, keep_dims).get_output(0)

    # compute sqrt(var + eps)
    var_trt = ctx.network.add_scale(var_trt, trt.ScaleMode.UNIFORM, eps_np, np.ones_like(eps_np), 0.5 * np.ones_like(eps_np)).get_output(0)
    
    # compute final result
    result_trt = ctx.network.add_elementwise(delta_trt, var_trt, trt.ElementWiseOperation.DIV).get_output(0)
    
    # compute affine (if applicable)
    if weight is not None:
        unsqueeze_size = input.ndim - len(normalized_shape)
        weight_np = weight.detach().cpu().numpy()
        weight_np = np.expand_dims(weight_np, 0)
        bias_np = bias.detach().cpu().numpy()
        bias_np = np.expand_dims(bias_np, 0)
        result_trt = ctx.network.add_scale(result_trt, trt.ScaleMode.CHANNEL, bias_np, weight_np, np.ones_like(bias_np)).get_output(0)

    output._trt = result_trt