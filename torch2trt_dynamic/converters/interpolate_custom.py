import torch
from ..torch2trt_dynamic import *
from ..module_test import add_module_test
from .size import IntWarper

@tensorrt_converter('torch.nn.functional.interpolate')
def convert_interpolate(ctx):

    input = ctx.method_args[0]

    try:
        scale_factor = get_arg(ctx, 'scale_factor', pos=2, default=None)
    except KeyError:
        scale_factor = None
    if isinstance(scale_factor, int):
        scale_factor = float(scale_factor)
    if isinstance(scale_factor, float):
        scale_factor = tuple([scale_factor]*(len(input.shape)-2))

    try:
        size = get_arg(ctx, 'size', pos=1, default=None)
    except KeyError:
        size = None
    
    if isinstance(size, int):
        size = [size]

    try:
        mode = get_arg(ctx, 'mode', pos=3, default='nearest')
    except KeyError:
        mode = 'nearest'

    try:
        align_corners = get_arg(ctx, 'align_corners', pos=4, default=None)
    except KeyError:
        align_corners = False

    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    is_shape_tensor = False
    if size is not None:
        for s in size:
            if isinstance(s, IntWarper):
                is_shape_tensor = True
                break

    if is_shape_tensor:
        shape_trt = []
        # tuple(input.shape[:(len(input.shape)-len(size))]) + 
        size = tuple(size)
        for s in size:
            if isinstance(s, IntWarper):
                shape_trt.append(s._trt)
            else:
                const_shape_trt = trt_(ctx.network, input.new_tensor([s],dtype=torch.int32))
                shape_trt.append(const_shape_trt)
        pre_input_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt, 0, (len(input.shape)-len(size)))
        shape_trt = [pre_input_shape_trt] + shape_trt
        shape_trt = ctx.network.add_concatenation(shape_trt).get_output(0)

    layer = ctx.network.add_resize(input_trt)
    
    if is_shape_tensor:
        layer.set_input(1, shape_trt)
    elif scale_factor is not None:
        scale_factor = (1,)*2 + tuple(scale_factor)
        layer.scales = scale_factor
    else:
        layer.shape = tuple(output.shape)
    layer.align_corners = align_corners

    if mode=="nearest":
        layer.resize_mode = trt.ResizeMode.NEAREST
    elif mode=="linear":
        layer.resize_mode = trt.ResizeMode.LINEAR
    else:
        layer.resize_mode = trt.ResizeMode.LINEAR
        print("unknown interpolate type, use linear insteed.")

    output._trt = layer.get_output(0)

    

class InterpolateTest(torch.nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(InterpolateTest, self).__init__()
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        align_corners = None
        if (self.mode!='nearest'):
            align_corners = True
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=align_corners)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 6)])
def test_interpolate_size_int_nearest():
    return InterpolateTest(2, mode='nearest')

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 6)])
def test_interpolate_size_3d_nearest():
    return InterpolateTest((2,), mode='nearest')

@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
def test_interpolate_size_4d_nearest():
    return InterpolateTest((2, 3), mode='nearest')

@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4, 6)])
def test_interpolate_size_5d_nearest():
    return InterpolateTest((2, 3, 4), mode='nearest')

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
def test_interpolate_size_int_linear():
    return InterpolateTest(2, mode='bilinear')
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
def test_interpolate_size_4d_linear():
    return InterpolateTest((2, 3), mode='bilinear')


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 6)])
def test_interpolate_scale_int_nearest():
    return InterpolateTest(scale_factor=2., mode='nearest')

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 6)])
def test_interpolate_scale_3d_nearest():
    return InterpolateTest(scale_factor=(4.), mode='nearest')

@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
def test_interpolate_scale_4d_nearest():
    return InterpolateTest(scale_factor=(4., 5.), mode='nearest')

@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4, 6)])
def test_interpolate_scale_5d_nearest():
    return InterpolateTest(scale_factor=(4., 5., 6.), mode='nearest')