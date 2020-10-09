import torch
from torch2trt_dynamic.torch2trt_dynamic import *

def get_intwarper_trt(other, ctx):
    if isinstance(other, IntWarper):
        return other._trt
    elif isinstance(other, int):
        return ctx.network.add_constant((1,), np.array([other], dtype=np.int32)).get_output(0)
    else:
        return other

class IntWarper(int):
    pass
    

class ShapeWarper(tuple):
    def numel(self):
        return torch.Size(self).numel()

def create_shape_warper(shape, trt, ctx):
    trt_shape = ctx.network.add_shape(trt).get_output(0)
    new_shape = []
    for i in range(len(shape)):
        int_warper=  IntWarper(shape[i])
        trt_int = ctx.network.add_slice(trt_shape,[i],[1],[1]).get_output(0)
        int_warper._trt = trt_int
        new_shape.append(int_warper)
    shape_warper = ShapeWarper(new_shape)
    return shape_warper

@tensorrt_converter('torch.Tensor.size')
def convert_size(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    
    input_trt = trt_(ctx.network, input)
    shape = input.size()

    if dim is None:
        shape_warper = create_shape_warper(shape, input_trt, ctx)
        ctx.method_return = shape_warper
    else:
        shape_warper = create_shape_warper(shape, input_trt, ctx)
        ctx.method_return = shape_warper[dim]

@tensorrt_converter('torch2trt_dynamic.converters.size.ShapeWarper.numel')
def convert_shapewarper_numel(ctx):
    shape = ctx.method_args[0]
    
    num = ctx.method_return
    
    num_trt = shape[i]._trt
    for i in range(1, len(shape)):
        other_trt = shape[i]._trt
        num_trt = ctx.network.add_elementwise(num_trt, other_trt, trt.ElementWiseOperation.PROD).get_output(0)
    intwarper = IntWarper(num)
    intwarper._trt = num_trt

    ctx.method_return = intwarper

@tensorrt_converter("torch2trt_dynamic.converters.size.IntWarper.__add__")
def convert_intwarper_add(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = get_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(self._trt, trt_other, trt.ElementWiseOperation.SUM).get_output(0)
        ret = IntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret


@tensorrt_converter("torch2trt_dynamic.converters.size.IntWarper.__radd__")
def convert_intwarper_radd(ctx):
    convert_intwarper_add(ctx)

    
@tensorrt_converter("torch2trt_dynamic.converters.size.IntWarper.__mul__")
def convert_intwarper_mul(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = get_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(self._trt, trt_other, trt.ElementWiseOperation.PROD).get_output(0)
        ret = IntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret

@tensorrt_converter("torch2trt_dynamic.converters.size.IntWarper.__rmul__")
def convert_intwarper_rmul(ctx):
    convert_intwarper_mul(ctx)


    
@tensorrt_converter("torch2trt_dynamic.converters.size.IntWarper.__sub__")
def convert_intwarper_sub(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = get_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(self._trt, trt_other, trt.ElementWiseOperation.SUB).get_output(0)
        ret = IntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret
    
@tensorrt_converter("torch2trt_dynamic.converters.size.IntWarper.__rsub__")
def convert_intwarper_rsub(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = get_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(trt_other, self._trt, trt.ElementWiseOperation.SUB).get_output(0)
        ret = IntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret

@tensorrt_converter("torch2trt_dynamic.converters.size.IntWarper.__floordiv__")
def convert_intwarper_floordiv(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = get_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(self._trt, trt_other, trt.ElementWiseOperation.FLOOR_DIV).get_output(0)
        ret = IntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret

        
@tensorrt_converter("torch2trt_dynamic.converters.size.IntWarper.__rfloordiv__")
def convert_intwarper_rfloordiv(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = get_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(trt_other, self._trt, trt.ElementWiseOperation.FLOOR_DIV).get_output(0)
        ret = IntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret

@tensorrt_converter("torch2trt_dynamic.converters.size.IntWarper.__pow__")
def convert_intwarper_pow(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = get_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(self._trt, trt_other, trt.ElementWiseOperation.POW).get_output(0)
        ret = IntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret

        
@tensorrt_converter("torch2trt_dynamic.converters.size.IntWarper.__rpow__")
def convert_intwarper_rpow(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = get_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(trt_other, self._trt, trt.ElementWiseOperation.POW).get_output(0)
        ret = IntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret
