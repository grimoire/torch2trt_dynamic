from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test


@tensorrt_converter('torch.mean')
@tensorrt_converter('torch.Tensor.mean')
def convert_mean(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    keep_dims = get_arg(ctx, 'keepdim', pos=2, default=False)
    
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

    layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, axes, keep_dims)
    output._trt = layer.get_output(0)

    
class Mean(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, x):
        return x.mean(self.dim, self.keepdim)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_channel():
    return Mean(1, False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_tuple():
    return Mean((1, 2), False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_keepdim():
    return Mean(1, True)