from torch2trt_dynamic.torch2trt_dynamic import get_arg, tensorrt_converter

from .cat import convert_cat
from .unsqueeze import convert_unsqueeze


@tensorrt_converter('torch.stack')
def convert_stack(ctx):
    inputs = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=0)
    output = ctx.method_return

    unsqueeze_inputs = []
    for input in inputs:
        unsqueeze_input = input.unsqueeze(dim=dim)
        ctx.method_args = (input, dim)
        ctx.method_return = unsqueeze_input
        convert_unsqueeze(ctx)
        unsqueeze_inputs.append(unsqueeze_input)

    ctx.method_args = (unsqueeze_inputs, dim)
    ctx.method_return = output

    convert_cat(ctx)
