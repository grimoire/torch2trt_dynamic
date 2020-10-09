from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test
from .transpose import convert_transpose


@tensorrt_converter('torch.Tensor.t')
def convert_t(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    # permutation -1 because TRT does not include batch dim

    if len(input.shape)==1:
        layer = ctx.network.add_identity(input_trt)
        output._trt = layer.get_output(0)
    else:
        ctx.method_args = [input, 1, 0]
        ctx.method_kwargs = {}
        convert_transpose(ctx)