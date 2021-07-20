from ..plugins import create_groupnorm_plugin
from ..torch2trt_dynamic import tensorrt_converter, trt_


@tensorrt_converter('torch.nn.GroupNorm.forward')
def convert_GroupNorm(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]

    input_trt = trt_(ctx.network, input)
    weight_trt = trt_(ctx.network, module.weight)
    bias_trt = trt_(ctx.network, module.bias)
    output = ctx.method_return

    num_groups = module.num_groups
    eps = module.eps

    plugin = create_groupnorm_plugin(
        'groupnorm_' + str(id(module)), num_groups=num_groups, eps=eps)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt, weight_trt, bias_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)
