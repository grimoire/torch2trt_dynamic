from torch2trt_dynamic.torch2trt_dynamic import *
from .div import convert_div


@tensorrt_converter('torch.reciprocal')
def convert_reciprocal(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return

    ctx.method_args = [1, input]
    ctx.method_kwargs = {}
    convert_div(ctx)
