from ..torch2trt_dynamic import *


@tensorrt_converter('torch.nn.Dropout.forward')
@tensorrt_converter('torch.nn.Dropout2d.forward')
@tensorrt_converter('torch.nn.Dropout3d.forward')
def convert_Identity(ctx):
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    output._trt = input_trt