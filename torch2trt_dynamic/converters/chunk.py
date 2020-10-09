from ..torch2trt_dynamic import *
from ..module_test import add_module_test
from .split import convert_split


@tensorrt_converter('torch.chunk')
@tensorrt_converter('torch.Tensor.chunk')
def convert_chunk(ctx):

    # https://github.com/pytorch/pytorch/blob/b90fc52c687a6851047f18ec9d06fb998efe99dd/aten/src/ATen/native/TensorShape.cpp

    input = get_arg(ctx, 'input', 0, None)
    input_trt = trt_(ctx.network, input)
    chunks = get_arg(ctx, 'chunks', 1, 0)
    dim = get_arg(ctx, 'dim', 2, 0)
    outputs = ctx.method_return

    if len(outputs)!=chunks:
        convert_split(ctx)
        return

    input_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_trt)
    head_shape_trt = slice_shape_trt(ctx.network, input_shape_trt, 0, dim)
    chunk_shape_trt = slice_shape_trt(ctx.network, input_shape_trt, dim, 1, 1)
    tail_shape_trt = slice_shape_trt(ctx.network, input_shape_trt, dim+1)

    chunk_trt = trt_(ctx.network, int(chunks))
    one_trt = trt_(ctx.network, 1)
    zero_trt = trt_(ctx.network, 0)

    # chunk 0~n-2
    chunk_size_trt = ctx.network.add_elementwise(chunk_shape_trt, chunk_trt, trt.ElementWiseOperation.SUM).get_output(0)
    chunk_size_trt = ctx.network.add_elementwise(chunk_size_trt, one_trt, trt.ElementWiseOperation.SUB).get_output(0)
    chunk_size_trt = ctx.network.add_elementwise(chunk_size_trt, chunk_trt, trt.ElementWiseOperation.FLOOR_DIV).get_output(0)
    
    # chunk n-1
    chunk_last_trt = ctx.network.add_elementwise(chunk_trt, one_trt, trt.ElementWiseOperation.SUB).get_output(0)
    chunk_last_trt = ctx.network.add_elementwise(chunk_size_trt, chunk_last_trt, trt.ElementWiseOperation.PROD).get_output(0)
    chunk_last_trt = ctx.network.add_elementwise(chunk_shape_trt, chunk_last_trt, trt.ElementWiseOperation.SUB).get_output(0)

    stride_trt = ctx.network.add_concatenation([one_trt]*len(input.shape)).get_output(0)
    if head_shape_trt is not None:
        head_start_trt = ctx.network.add_concatenation([zero_trt]*dim).get_output(0)

    if tail_shape_trt is not None:
        tail_start_trt = ctx.network.add_concatenation([zero_trt]*(len(input.shape)-1-dim)).get_output(0)

    start_trt = []
    size_trt = []
    chunk_start_trt = zero_trt
    if head_shape_trt is not None:
        start_trt.append(head_start_trt)
        size_trt.append(head_shape_trt)
    start_trt.append(chunk_start_trt)
    size_trt.append(chunk_size_trt)
    if tail_shape_trt is not None:
        start_trt.append(tail_start_trt)
        size_trt.append(tail_shape_trt)
    start_trt = ctx.network.add_concatenation(start_trt).get_output(0)
    size_trt = ctx.network.add_concatenation(size_trt).get_output(0)
    
    input_dim = len(input.shape)
    for i in range(chunks-1):
        layer = ctx.network.add_slice(input_trt, [0]*input_dim, [1]*input_dim, [1]*input_dim)
        layer.set_input(1, start_trt)
        layer.set_input(2, size_trt)
        layer.set_input(3, stride_trt)
        outputs[i]._trt = layer.get_output(0)
        
        start_trt = []
        chunk_start_trt = ctx.network.add_elementwise(chunk_start_trt, chunk_size_trt, trt.ElementWiseOperation.SUM).get_output(0)
        if head_shape_trt is not None:
            start_trt.append(head_start_trt)
        start_trt.append(chunk_start_trt)
        if tail_shape_trt is not None:
            start_trt.append(tail_start_trt)
        start_trt = ctx.network.add_concatenation(start_trt).get_output(0)

    
    size_trt = []
    if head_shape_trt is not None:
        size_trt.append(head_shape_trt)
    size_trt.append(chunk_last_trt)
    if tail_shape_trt is not None:
        size_trt.append(tail_shape_trt)
    size_trt = ctx.network.add_concatenation(size_trt).get_output(0)

    layer = ctx.network.add_slice(input_trt, [0]*input_dim, [1]*input_dim, [1]*input_dim)
    layer.set_input(1, start_trt)
    layer.set_input(2, size_trt)
    layer.set_input(3, stride_trt)
    outputs[chunks-1]._trt = layer.get_output(0)


        
class TorchChunk(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(TorchChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs
        
    def forward(self, x):
        return torch.chunk(x, *self.args, **self.kwargs)
    

class TensorChunk(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(TensorChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs
        
    def forward(self, x):
        return x.chunk(*self.args, **self.kwargs)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_1_1():
    return TorchChunk(1, 1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_2_1():
    return TorchChunk(2, 1)

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_3_1():
    return TorchChunk(3, 1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_3_2():
    return TorchChunk(3, 2)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_tensor_chunk_3_2():
    return TensorChunk(3, 2)