from torch2trt.torch2trt import *
import tensorrt as trt



@tensorrt_converter('torch.index_select')
@tensorrt_converter('torch.Tensor.index_select')
def convert_index_select(ctx):
    input = ctx.method_args[0]
    dim = ctx.method_args[1]
    index = ctx.method_args[2]
    
    input_trt = trt_(ctx.network, input)
    index_trt = trt_(ctx.network, index)
    output = ctx.method_return

    layer = ctx.network.add_gather(input_trt, index_trt, dim)
    output._trt = layer.get_output(0)
