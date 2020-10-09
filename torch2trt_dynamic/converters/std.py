from torch2trt_dynamic.torch2trt_dynamic import *
from .mean import convert_mean
from .sub import convert_sub
from .mul import convert_mul

from torch2trt_dynamic.module_test import add_module_test


@tensorrt_converter('torch.std')
@tensorrt_converter('torch.Tensor.std')
def convert_std(ctx):
    old_method_args = ctx.method_args
    old_method_kwargs = ctx.method_kwargs

    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    unbiased = get_arg(ctx, 'unbiased', pos=2, default=True)
    keepdim = get_arg(ctx, 'keepdim', pos=3, default=False)

    # compute mean
    if dim is not None:
        mean_val = input.mean(dim, True)
        ctx.method_args = [input, dim, True]
        ctx.method_kwargs = []
        ctx.method_return = mean_val
        convert_mean(ctx)
    else:
        mean_val = input.mean()
        ctx.method_args = [input, None, False]
        ctx.method_kwargs = []
        ctx.method_return = mean_val
        convert_mean(ctx)
    
    # compute x-mean
    x_minus_mean = input-mean_val
    ctx.method_args = [input, mean_val]
    ctx.method_return = x_minus_mean
    convert_sub(ctx)

    # compute (x-mean)*(x-mean)
    x_pow = x_minus_mean*x_minus_mean
    ctx.method_args = [x_minus_mean, x_minus_mean]
    ctx.method_return = x_pow
    convert_mul(ctx)

    # compute average
    x_pow_trt = trt_(ctx.network, x_pow)
    # get dims from args or kwargs
    if dim is None:
        dim = tuple(range(len(input.shape)))
        
    # convert list to tuple
    if isinstance(dim, list):
        dim = tuple(dim)
        
    if not isinstance(dim, tuple):
        dim = (dim, )

    dim = tuple([d if d>=0 else len(input.shape)+d for d in dim])
        
    # create axes bitmask for reduce layer
    axes = 0
    for d in dim:
        axes |= 1<<d
    
    if unbiased:
        layer = ctx.network.add_reduce(x_pow_trt, trt.ReduceOperation.SUM, axes, keepdim)
        sum_trt = layer.get_output(0)
        # compute reduce size
        shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt, dim[0], 1)
        layer = ctx.network.add_identity(shape_trt)
        layer.set_output_type(0, trt.float32)
        shape_trt = layer.get_output(0)
        for d in dim[1:]:
            other_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt, d, 1)
            layer = ctx.network.add_identity(other_shape_trt)
            layer.set_output_type(0, trt.float32)
            other_shape_trt = layer.get_output(0)
            layer = ctx.network.add_elementwise(shape_trt, other_shape_trt, trt.ElementWiseOperation.PROD)
            layer.set_output_type(0, trt.float32)
            shape_trt = layer.get_output(0)
        # reduce size minus one
        one_trt = trt_(ctx.network, input.new_ones((1,)).float())
        layer = ctx.network.add_elementwise(shape_trt, one_trt, trt.ElementWiseOperation.SUB)
        layer.set_output_type(0, sum_trt.dtype)
        shape_minus_one_trt = layer.get_output(0)
        
        layer = ctx.network.add_shuffle(shape_minus_one_trt)
        layer.reshape_dims = (1,) * len(sum_trt.shape)
        shape_minus_one_trt = layer.get_output(0)

        # multi scale
        layer = ctx.network.add_elementwise(sum_trt, shape_minus_one_trt, trt.ElementWiseOperation.DIV)
        avg_trt = layer.get_output(0)
    else:
        layer = ctx.network.add_reduce(x_pow_trt, trt.ReduceOperation.AVG, axes, keepdim)
        avg_trt = layer.get_output(0)

    # reduce shape might be zero
    need_reshape = False
    if len(avg_trt.shape)==0:
        need_reshape = True
        layer = ctx.network.add_shuffle(avg_trt)
        layer.reshape_dims = (1,)
        avg_trt = layer.get_output(0)

    layer = ctx.network.add_unary(avg_trt, trt.UnaryOperation.SQRT)
    output._trt = layer.get_output(0)
    
    if need_reshape:
        layer = ctx.network.add_shuffle(output._trt)
        layer.reshape_dims = tuple()
        output._trt = layer.get_output(0)

    ctx.method_args = old_method_args
    ctx.method_kwargs = old_method_kwargs
    ctx.method_return = output