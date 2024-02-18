import pytest
import torch
from torch import nn
from torch2trt_dynamic import module2trt


class _TestModel(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input, index):
        return torch.gather(input, self.dim, index)


class TestGather:

    @pytest.fixture
    def input(self):
        yield torch.rand(3, 4, 5).cuda()

    @pytest.fixture
    def dim(self, request):
        yield request.param

    @pytest.fixture
    def index(self, input, dim):
        max_val = input.size(dim)
        yield torch.randint(max_val, (3, 4, 5)).cuda()

    @pytest.mark.parametrize('dim', [0, 1, 2])
    def test_gather(self, input, dim, index):
        model = _TestModel(dim)
        dummy_input = torch.zeros_like(input)
        dummy_index = torch.zeros_like(index)
        trt_model = module2trt(model,
                               args=[dummy_input, dummy_index])

        with torch.inference_mode():
            gt = model(input, index)
            out = trt_model(input, index)
        torch.testing.assert_close(out, gt)
