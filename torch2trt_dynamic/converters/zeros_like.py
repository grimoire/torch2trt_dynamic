from torch2trt_dynamic.torch2trt_dynamic import *
from .mul import convert_mul
from .add import convert_add
from .cast_type import *


@tensorrt_converter('torch.zeros_like')
def convert_zeros_like(ctx):
    input = ctx.method_args[0]
    dtype = get_arg(ctx, 'dtype', pos=1, default=torch.float32)
    output = ctx.method_return
    input_trt = trt_(ctx.network, input)

    old_method_args = ctx.method_args
    old_method_kwargs = ctx.method_kwargs

    # mul zero
    input_mul_zero = input*0
    ctx.method_args = [input, 0]
    ctx.method_kwargs = {}
    ctx.method_return = input_mul_zero
    convert_mul(ctx)

    convert_type_func = None
    if dtype==torch.float32:
        convert_type_func = convert_float
    elif dtype==torch.int32 or dtype==torch.long:
        convert_type_func = convert_int
    elif dtype==torch.bool:
        convert_type_func = convert_bool
    else:
        print("unsupported convert type:{}".format(dtype))
    
    if convert_type_func is not None:
        input_as_type = input_mul_zero.to(dtype)
        ctx.method_args = [input_mul_zero, dtype]
        ctx.method_return = input_as_type
        convert_type_func(ctx)
        ctx.method_args = [input_as_type, 0]
        ctx.method_kwargs = {}
        ctx.method_return = output
        convert_add(ctx)

    ctx.method_args = old_method_args
    ctx.method_kwargs = old_method_kwargs
    ctx.method_return = output