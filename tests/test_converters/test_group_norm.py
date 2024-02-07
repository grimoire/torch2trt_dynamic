import pytest
import torch
from torch import nn
from torch2trt_dynamic import module2trt


class _TestModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(*args, **kwargs)

    def forward(self, input):
        return self.gn(input)


class TestGroupNorm:

    @pytest.fixture
    def num_channels(self):
        yield 4

    @pytest.fixture
    def input(self, num_channels):
        yield torch.rand(2, num_channels, 8, 16).cuda()

    @pytest.fixture
    def num_groups(self):
        yield 2

    def test_group_norm(self, input, num_groups):
        num_channels = input.size(1)
        model = _TestModel(num_groups, num_channels)
        model = model.eval().cuda()
        dummy_input = torch.zeros_like(input)
        trt_model = module2trt(model,
                               args=[dummy_input])

        with torch.inference_mode():
            gt = model(input)
            out = trt_model(input)
        torch.testing.assert_close(out, gt)
