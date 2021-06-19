import torch

from ..torch2trt_dynamic import get_arg, tensorrt_converter
from .Linear import convert_Linear


@tensorrt_converter('torch.nn.functional.linear')
def convert_linear(ctx):
    old_method_args = ctx.method_args
    old_method_kwargs = ctx.method_kwargs

    input = ctx.method_args[0]
    weight = get_arg(ctx, 'weight', pos=1, default=None)
    bias = get_arg(ctx, 'bias', pos=2, default=None)
    output = ctx.method_return

    in_channels = weight.shape[1]
    out_channels = weight.shape[0]
    module = torch.nn.Linear(in_channels, out_channels, bias is not None)
    module.weight = torch.nn.Parameter(weight)
    if bias is not None:
        module.bias = torch.nn.Parameter(bias)

    ctx.method_args = [module, input]
    ctx.method_kwargs = {}
    convert_Linear(ctx)

    ctx.method_args = old_method_args
    ctx.method_kwargs = old_method_kwargs
    ctx.method_return = output
