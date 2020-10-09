from ..torch2trt_dynamic import *
from ..plugins import *


@tensorrt_converter('torch.Tensor.gather')
@tensorrt_converter('torch.gather')
def convert_gather(ctx):
    inputs = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=0)
    index = get_arg(ctx, 'index', pos=2, default=None)
    output = ctx.method_return
    
    inputs_trt = trt_(ctx.network, inputs)
    index_trt = trt_(ctx.network, index)

    plugin = create_torchgather_plugin("torch_gather_"+str(id(inputs)),
                                        dim=dim)
            
    layer = ctx.network.add_plugin_v2(
        inputs=[inputs_trt, index_trt], plugin=plugin)

    output._trt = layer.get_output(0)