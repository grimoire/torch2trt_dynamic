from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test


@tensorrt_converter('torch.where')
def convert_where(ctx):
    condition = get_arg(ctx, 'condition', pos=0, default=None)
    x = get_arg(ctx, 'x', pos=1, default=None)
    y = get_arg(ctx, 'y', pos=2, default=None)

    condition_trt = trt_(ctx.network, condition)
    x_trt = trt_(ctx.network, x)
    y_trt = trt_(ctx.network, y)
    output = ctx.method_return

    layer = ctx.network.add_select(condition_trt, x_trt, y_trt)
    output_trt = layer.get_output(0)

    output._trt = output_trt


@tensorrt_converter('torch.Tensor.where')
def convert_Tensor_where(ctx):
    x = ctx.method_args[0]
    condition = get_arg(ctx, 'condition', pos=1, default=None)
    y = get_arg(ctx, 'y', pos=2, default=None)

    ctx.method_args = [condition, x, y]
    ctx.method_kwargs = {}
    convert_where(ctx)
