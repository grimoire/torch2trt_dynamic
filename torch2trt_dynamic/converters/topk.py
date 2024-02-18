import tensorrt as trt
import torch
from torch2trt_dynamic.module_test import add_module_test
from torch2trt_dynamic.torch2trt_dynamic import (bind_arguments,
                                                 tensorrt_converter, trt_)

from .size import IntWarper


def _dummy_topk(input, k, dim=None, largest=True, sorted=True, *, out=None):
    pass


@tensorrt_converter('torch.topk')
@tensorrt_converter('torch.Tensor.topk')
def convert_topk(ctx):
    arguments = bind_arguments(_dummy_topk, ctx)
    input = arguments['input']
    k = arguments['k']
    dim = arguments['dim']
    largest = arguments['largest']

    if dim is None:
        dim = len(input.shape) - 1
    if dim < 0:
        dim = len(input.shape) + dim

    def __add_unsqueeze_layer(input_trt, dim):
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (1, ) + tuple(input_trt.shape)
        input_trt = layer.get_output(0)
        dim += 1
        return input_trt, dim

    def __add_topk_layer(k, dim):
        topkOp = trt.TopKOperation.MAX if largest else trt.TopKOperation.MIN

        k_trt = None
        if isinstance(k, IntWarper):
            k_trt = trt_(ctx.network, k)
            layer = ctx.network.add_shuffle(k_trt)
            layer.reshape_dims = tuple()
            k_trt = layer.get_output(0)

        if isinstance(k, int) and k > 3840:
            print('Clamp k to 3840.')
            k = 3840

        layer = ctx.network.add_topk(input_trt, topkOp, k, 1 << dim)

        if k_trt is not None:
            layer.set_input(1, k_trt)

        output0_trt = layer.get_output(0)
        output1_trt = layer.get_output(1)
        return output0_trt, output1_trt

    def __add_squeeze_layer(output_trt):
        layer = ctx.network.add_shuffle(output_trt)
        layer.reshape_dims = tuple(output_trt.shape)[1:]
        return layer.get_output(0)

    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    # can only use topk on dim>=2
    need_unsqueeze = len(input_trt.shape) == 1
    if need_unsqueeze:
        input_trt, dim = __add_unsqueeze_layer(input_trt, dim)

    output0_trt, output1_trt = __add_topk_layer(k, dim)

    # recovery
    if need_unsqueeze:
        output0_trt = __add_squeeze_layer(output0_trt)
        output1_trt = __add_squeeze_layer(output1_trt)

    output[0]._trt = output0_trt
    output[1]._trt = output1_trt


class TopkTestModule(torch.nn.Module):

    def __init__(self, k, dim, largest):
        super(TopkTestModule, self).__init__()
        self.k = k
        self.dim = dim
        self.largest = largest

    def forward(self, x):
        return x.topk(k=self.k, dim=self.dim, largest=self.largest)


@add_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 20, 4, 6)],
    max_workspace_size=1 << 20)
@add_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 20, 6)],
    max_workspace_size=1 << 20)
@add_module_test(
    torch.float32, torch.device('cuda'), [(1, 20)], max_workspace_size=1 << 20)
def test_topk_dim1():
    return TopkTestModule(10, 1, True)


@add_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 4, 20, 6)],
    max_workspace_size=1 << 20)
@add_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 6, 20)],
    max_workspace_size=1 << 20)
def test_topk_dim2():
    return TopkTestModule(10, 2, True)


@add_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 20, 4, 6)],
    max_workspace_size=1 << 20)
@add_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 20, 6)],
    max_workspace_size=1 << 20)
@add_module_test(
    torch.float32, torch.device('cuda'), [(1, 20)], max_workspace_size=1 << 20)
def test_topk_largest_false():
    return TopkTestModule(10, 1, False)
