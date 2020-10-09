from torch2trt_dynamic.torch2trt_dynamic import *
import tensorrt as trt
from torch2trt_dynamic.module_test import add_module_test


@tensorrt_converter('torch.topk')
@tensorrt_converter('torch.Tensor.topk')
def convert_topk(ctx):

    input = ctx.method_args[0]

    k=get_arg(ctx, 'k', pos=1, default = 1)
    axis = get_arg(ctx, 'dim', pos=2, default = len(input.shape)-1)
    if axis is None:
        axis = len(input.shape)-1
    largest = get_arg(ctx, 'largest', pos=3, default = True)
    topkOp = trt.TopKOperation.MAX if largest else trt.TopKOperation.MIN
    
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    
    # can only use topk on dim>=2
    need_unsqueeze = len(input_trt.shape)==1
    if need_unsqueeze:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (1,) + tuple(input_trt.shape)
        input_trt = layer.get_output(0)
        axis+=1

    layer = ctx.network.add_topk(input_trt, topkOp, k, 1<<axis)

    output0_trt = layer.get_output(0)
    output1_trt = layer.get_output(1)

    # recovery
    if need_unsqueeze:
        layer = ctx.network.add_shuffle(output0_trt)
        layer.reshape_dims = tuple(output0_trt.shape)[1:]
        output0_trt = layer.get_output(0)

        layer = ctx.network.add_shuffle(output1_trt)
        layer.reshape_dims = tuple(output1_trt.shape)[1:]
        output1_trt = layer.get_output(0)
        
    output[0]._trt = output0_trt
    output[1]._trt = output1_trt


class TopkTestModule(torch.nn.Module):
    def __init__(self, k, dim, largest):
        super(TopkTestModule, self).__init__()
        self.k=k
        self.dim=dim
        self.largest = largest

    def forward(self, x):
        return x.topk(k=self.k, dim=self.dim, largest=self.largest)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 20, 4, 6)], max_workspace_size=1<<20)
@add_module_test(torch.float32, torch.device('cuda'), [(1, 20, 6)], max_workspace_size=1<<20)
@add_module_test(torch.float32, torch.device('cuda'), [(1, 20)], max_workspace_size=1<<20)
def test_topk_dim1():
    return TopkTestModule(10, 1, True)

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 20, 6)], max_workspace_size=1<<20)
@add_module_test(torch.float32, torch.device('cuda'), [(1, 6, 20)], max_workspace_size=1<<20)
def test_topk_dim2():
    return TopkTestModule(10, 2, True)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 20, 4, 6)], max_workspace_size=1<<20)
@add_module_test(torch.float32, torch.device('cuda'), [(1, 20, 6)], max_workspace_size=1<<20)
@add_module_test(torch.float32, torch.device('cuda'), [(1, 20)], max_workspace_size=1<<20)
def test_topk_largest_false():
    return TopkTestModule(10, 1, False)