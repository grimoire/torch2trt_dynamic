from torch2trt_dynamic.torch2trt_dynamic import *
from .cast_type import *


@tensorrt_converter('torch.Tensor.to')
def convert_Tensor_to(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return

    input_trt = trt_(ctx.network, input)
    if output.dtype == input.dtype:
        output._trt = input_trt
    else:
        data_type = output.dtype
        if data_type == torch.int64:
            data_type = torch.int32
        
        output_trt = trt_cast(ctx.network, input_trt, data_type)
        output._trt = output_trt