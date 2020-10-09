from ..torch2trt_dynamic import *
from .mul import convert_mul
from .add import convert_add
from .cast_type import *

@tensorrt_converter('torch.full_like')
def convert_full_like(ctx):
    input = ctx.method_args[0]
    fill_value = get_arg(ctx, "fill_value", pos=1, default=0)
    dtype = get_arg(ctx, 'dtype', pos=3, default=torch.float32)
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

    # add fill_value
    input_add_one = input_mul_zero+fill_value
    ctx.method_args = [input_mul_zero, fill_value]
    ctx.method_kwargs = {}
    ctx.method_return = input_add_one
    convert_add(ctx)

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
        input_as_type = input_add_one.to(dtype)
        ctx.method_args = [input_add_one, dtype]
        ctx.method_return = input_as_type
        convert_type_func(ctx)
        ctx.method_args = [input_as_type, 0]
        ctx.method_kwargs = {}
        ctx.method_return = output
        convert_add(ctx)

    ctx.method_args = old_method_args
    ctx.method_kwargs = old_method_kwargs
    ctx.method_return = output