from ..torch2trt_dynamic import *
import tensorrt as trt

@tensorrt_converter('torch.index_select')
@tensorrt_converter('torch.Tensor.index_select')
def convert_index_select(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    index = get_arg(ctx, 'index', pos=2, default=None)
    
    input_trt = trt_(ctx.network, input)
    index_trt = trt_(ctx.network, index)
    output = ctx.method_return

    layer = ctx.network.add_gather(input_trt, index_trt, dim)
    output._trt = layer.get_output(0)
