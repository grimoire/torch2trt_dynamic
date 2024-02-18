import pytest
import torch
from torch import nn
from torch2trt_dynamic import module2trt


class _TestStaticKModel(nn.Module):

    def __init__(self, k, dim, largest) -> None:
        super().__init__()
        self.k = k
        self.dim = dim
        self.largest = largest

    def forward(self, input):
        val, index = input.topk(k=self.k, dim=self.dim, largest=self.largest)
        return val, index


class _TestDynamicModel(nn.Module):

    def __init__(self, k, dim, largest) -> None:
        super().__init__()
        self.k = k
        self.dim = dim
        self.largest = largest

    def forward(self, input):
        new_k = input.size(self.dim)
        k = min(self.k, new_k)
        val, index = input.topk(k=k, dim=self.dim, largest=self.largest)
        return val, index


class TestTopk:

    @pytest.fixture
    def shape(self, request):
        yield request.param

    @pytest.fixture
    def dim(self, request):
        yield request.param

    @pytest.fixture
    def k(self, request):
        yield request.param

    @pytest.fixture
    def largest(self, request):
        yield request.param

    @pytest.fixture
    def input(self, shape):
        yield torch.rand(shape).cuda()

    @pytest.mark.parametrize('shape,dim', [
        ((5, 10), 0),
        ((5, 10), 1),
        ((5, ), 0),
    ])
    @pytest.mark.parametrize('k', [3])
    @pytest.mark.parametrize('largest', [True, False])
    def test_static(self, input, k, dim, largest):
        model = _TestStaticKModel(k, dim, largest)

        dummy_input = torch.zeros_like(input)
        trt_model = module2trt(model, args=[dummy_input])

        with torch.inference_mode():
            gt = model(input)
            out = trt_model(input)
        torch.testing.assert_close(out[0], gt[0])
        torch.testing.assert_close(out[1].to(torch.int64), gt[1])

    @pytest.mark.parametrize('shape,dim', [
        ((5, 10), 0),
        ((5, 10), 1),
        ((5, ), 0),
    ])
    @pytest.mark.parametrize('k', [6])
    @pytest.mark.parametrize('largest', [True, False])
    def test_dynamic(self, input, k, dim, largest):
        model = _TestDynamicModel(k, dim, largest)

        dummy_input = torch.zeros_like(input)
        trt_model = module2trt(model, args=[dummy_input])

        with torch.inference_mode():
            gt = model(input)
            out = trt_model(input)
        torch.testing.assert_close(out[0], gt[0])
        torch.testing.assert_close(out[1].to(torch.int64), gt[1])
