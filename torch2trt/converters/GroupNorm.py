from torch2trt.torch2trt import *
from torch2trt.plugins import *


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
