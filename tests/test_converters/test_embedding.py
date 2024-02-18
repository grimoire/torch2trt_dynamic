import pytest
import torch
from torch import nn
from torch2trt_dynamic import module2trt


class _TestModel(nn.Module):

    def __init__(self, num, dim) -> None:
        super().__init__()
        self.embeding = nn.Embedding(num, dim)

    def forward(self, input):
        return self.embeding(input)


class TestGather:

    @pytest.fixture
    def dim(self):
        yield 4

    @pytest.fixture
    def num(self):
        yield 10

    @pytest.fixture
    def batch(self):
        yield 2

    @pytest.fixture
    def input(self, batch, num):
        yield torch.randint(num, (batch, 6)).cuda()

    def test_gather(self, input, dim, num):
        model = _TestModel(num, dim).eval().cuda()
        dummy_input = torch.zeros_like(input)
        trt_model = module2trt(model, args=[dummy_input])

        with torch.inference_mode():
            gt = model(input)
            out = trt_model(input)
        torch.testing.assert_close(out, gt)
