from ..torch2trt_dynamic import *


@tensorrt_converter('torch.Tensor.cuda')
@tensorrt_converter('torch.Tensor.detach')
@tensorrt_converter('torch.Tensor.contiguous')
@tensorrt_converter('torch.nn.functional.dropout')
@tensorrt_converter('torch.nn.functional.dropout2d')
@tensorrt_converter('torch.nn.functional.dropout3d')
def convert_identity(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    output._trt = input_trt
