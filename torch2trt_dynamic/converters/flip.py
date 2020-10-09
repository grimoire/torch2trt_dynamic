from ..torch2trt_dynamic import *
from ..plugins import *

@tensorrt_converter('torch.flip')
@tensorrt_converter('torch.Tensor.flip')
def convert_flip(ctx):
    input = ctx.method_args[0]
    dims = get_arg(ctx, 'dims', pos=1, default=0)
    if isinstance(dims, int):
        dims = [dims]
    
    dims = [len(input.shape)+dim if dim<0 else dim for dim in dims]

    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    plugin = create_torchflip_plugin("flip_" + str(id(input)),
                                  dims=dims
                                  )
    
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)

