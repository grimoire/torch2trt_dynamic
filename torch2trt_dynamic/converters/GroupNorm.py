from ..torch2trt_dynamic import *
from ..plugins import *


@tensorrt_converter('torch.nn.GroupNorm.forward')
def convert_GroupNorm(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]

    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    num_channels = module.num_channels
    num_groups = module.num_groups
    weight = module.weight.detach().cpu().numpy()
    bias = module.bias.detach().cpu().numpy()
    eps = module.eps

    plugin = create_groupnorm_plugin("groupnorm_" + str(id(module)),
                                     num_groups=num_groups,
                                     num_channels=num_channels,
                                     W=weight,
                                     B=bias,
                                     eps=eps
                                     )

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)

# @tensorrt_converter('torch.nn.GroupNorm.forward')
# def convert_GroupNorm(ctx):
#     module = ctx.method_args[0]
#     input = ctx.method_args[1]

#     input_trt = trt_(ctx.network, input)
#     output = ctx.method_return

#     num_channels = module.num_channels
#     num_groups = module.num_groups
#     weight = module.weight.detach().cpu().numpy()
#     bias = module.bias.detach().cpu().numpy()
#     eps = module.eps

#     input_shape_trt = ctx.network.add_shape(input_trt).get_output(0)
#     input_batch_trt = ctx.network.add_slice(input_shape_trt, [0], [1], [1]).get_output(0)
#     input_channel_trt = ctx.network.add_slice(input_shape_trt, [1], [1], [1]).get_output(0)
#     input_hw_trt = ctx.network.add_slice(input_shape_trt, [2], [2], [1]).get_output(0)
    
#     group_length = num_channels//num_groups
#     num_groups_trt = trt_(ctx.network, torch.tensor([num_groups],dtype=torch.int32).to(input.device))
#     group_length_trt = trt_(ctx.network, torch.tensor([group_length],dtype=torch.int32).to(input.device))

#     new_shape_trt = ctx.network.add_concatenation([input_batch_trt, num_groups_trt, group_length_trt, input_hw_trt]).get_output(0)
#     layer = ctx.network.add_shuffle(input_trt)
#     layer.set_input(1, new_shape_trt)
#     new_input_trt = layer.get_output(0)

#     group_trts = []
#     eps_np = np.array([eps], dtype=np.float32)
#     keep_dims = True
#     reduce_axes = torch_dim_to_trt_axes(tuple(range(2,5)))

#     mean_trt = ctx.network.add_reduce(new_input_trt, trt.ReduceOperation.AVG, reduce_axes, keep_dims).get_output(0)

#     # compute variance over spatial (include eps, to reduce layer count)
#     delta_trt = ctx.network.add_elementwise(new_input_trt, mean_trt, trt.ElementWiseOperation.SUB).get_output(0)
#     var_trt = ctx.network.add_scale(delta_trt, trt.ScaleMode.UNIFORM, np.zeros_like(eps_np), np.ones_like(eps_np), 2 * np.ones_like(eps_np)).get_output(0)
#     var_trt = ctx.network.add_reduce(var_trt, trt.ReduceOperation.AVG, reduce_axes, keep_dims).get_output(0)

#     # compute sqrt(var + eps)
#     var_trt = ctx.network.add_scale(var_trt, trt.ScaleMode.UNIFORM, eps_np, np.ones_like(eps_np), 0.5 * np.ones_like(eps_np)).get_output(0)
        
#     # compute final result
#     norm_input_trt = ctx.network.add_elementwise(delta_trt, var_trt, trt.ElementWiseOperation.DIV).get_output(0)
#     layer = ctx.network.add_shuffle(norm_input_trt)
#     layer.set_input(1, input_shape_trt)
#     norm_input_trt = layer.get_output(0)

#     layer = ctx.network.add_scale(norm_input_trt, trt.ScaleMode.CHANNEL, bias, weight, np.ones_like(bias))
#     output._trt = layer.get_output(0)
