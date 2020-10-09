from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test


@tensorrt_converter('torch.Tensor.unsqueeze')
@tensorrt_converter('torch.unsqueeze')
def convert_unsqueeze(ctx):
        
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    if dim<0:
        dim = len(input.shape)+dim+1
    input_trt = trt_(ctx.network, input)
    shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    unsqueeze_trt = trt_(ctx.network, input.new_ones((1),dtype=torch.int32))
    output = ctx.method_return

    shape1_trt = None
    shape2_trt = None
    if dim == 0:
        shape2_trt = shape_trt
    elif dim == len(input.shape):
        shape1_trt = shape_trt
    else:
        slice1_start = [0]
        slice1_size = [dim]
        slice1_stride = [1]
        shape1_trt = ctx.network.add_slice(shape_trt, slice1_start, slice1_size, slice1_stride).get_output(0)
        slice2_start = [dim]
        slice2_size = [len(input.shape)-dim]
        slice2_stride = [1]
        shape2_trt = ctx.network.add_slice(shape_trt, slice2_start, slice2_size, slice2_stride).get_output(0)

    if shape1_trt == None:
        new_shape_trt = ctx.network.add_concatenation([unsqueeze_trt, shape2_trt]).get_output(0)
    elif shape2_trt == None:
        new_shape_trt = ctx.network.add_concatenation([shape1_trt, unsqueeze_trt]).get_output(0)
    else:
        new_shape_trt = ctx.network.add_concatenation([shape1_trt, unsqueeze_trt, shape2_trt]).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_shape_trt)
    output._trt = layer.get_output(0)