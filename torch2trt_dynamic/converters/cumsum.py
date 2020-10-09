from ..torch2trt_dynamic import *

from ..plugins import *

@tensorrt_converter('torch.cumsum')
@tensorrt_converter('torch.Tensor.cumsum')
def convert_cumsum(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=0)
    cum_type = 0

    if dim<0:
        dim = len(input.shape)+dim
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    plugin = create_torchcum_plugin("cumsum_" + str(id(input)),
                                            dim=dim,
                                            cum_type=cum_type
                                            )
    
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)

