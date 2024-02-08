import pytest
import torch
from torch import nn
from torch2trt_dynamic import BuildEngineConfig, module2trt
from torch.nn import functional as F


class _TestModel(nn.Module):

    def __init__(self, mode, padding_mode, align_corners) -> None:
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input, grid):
        return F.grid_sample(
            input,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners)


class TestGridSample:

    @pytest.fixture
    def hw_in(self, request):
        yield request.param

    @pytest.fixture
    def hw_out(self, request):
        yield request.param

    @pytest.fixture
    def mode(self, request):
        yield request.param

    @pytest.fixture
    def padding_mode(self, request):
        yield request.param

    @pytest.fixture
    def align_corners(self, request):
        yield request.param

    @pytest.fixture
    def batch(self):
        yield 2

    @pytest.fixture
    def channel(self):
        yield 4

    @pytest.fixture
    def deep_in(self):
        yield 4

    @pytest.fixture
    def deep_out(self):
        yield 2

    @pytest.fixture
    def input4d(self, batch, channel, hw_in):
        yield torch.rand(batch, channel, *hw_in).cuda()

    @pytest.fixture
    def input5d(self, batch, channel, deep_in, hw_in):
        yield torch.rand(batch, channel, deep_in, *hw_in).cuda()

    @pytest.fixture
    def grid4d(self, batch, hw_out):
        lin_w = torch.linspace(-1, 1, hw_out[1])[:, None].repeat(1, hw_out[0])
        lin_h = torch.linspace(-1, 1, hw_out[0]).repeat(hw_out[1], 1)
        grid = torch.stack([lin_w, lin_h], dim=-1)
        grid = grid[None].repeat(batch, 1, 1, 1)
        yield grid.cuda()

    @pytest.fixture
    def grid5d(self, batch, deep_out, hw_out):
        lin_d = torch.linspace(-1, 1,
                               deep_out)[:, None,
                                         None].repeat(1, hw_out[1], hw_out[0])
        lin_w = torch.linspace(-1, 1,
                               hw_out[1])[None, :,
                                          None].repeat(deep_out, 1, hw_out[0])
        lin_h = torch.linspace(-1, 1, hw_out[0])[None, None, :].repeat(
            deep_out, hw_out[1], 1)
        grid = torch.stack([lin_w, lin_h, lin_d], dim=-1)
        grid = grid[None].repeat(batch, 1, 1, 1, 1)
        yield grid.cuda()

    @pytest.fixture
    def model(self, mode, padding_mode, align_corners):
        kwargs = dict(
            mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        yield _TestModel(**kwargs)

    def make_config(self, input, grid):
        input_shape = tuple(input.shape)
        input_post = input_shape[2:]
        input_post_max = [x * 2 for x in input_post]
        input_post_min = [x // 2 for x in input_post]
        input_max = (*input_shape[:2], *input_post_max)
        input_min = (*input_shape[:2], *input_post_min)
        grid_shape = tuple(grid.shape)
        grid_post = grid_shape[1:-1]
        grid_post_max = [x * 2 for x in grid_post]
        grid_post_min = [x // 2 for x in grid_post]
        grid_max = (grid_shape[0], *grid_post_max, grid_shape[-1])
        grid_min = (grid_shape[0], *grid_post_min, grid_shape[-1])
        config = BuildEngineConfig(
            shape_ranges=dict(
                input=dict(min=input_min, opt=input_shape, max=input_max),
                grid=dict(min=grid_min, opt=grid_shape, max=grid_max)))
        return config

    @pytest.mark.parametrize('hw_in,hw_out', [
        ((8, 16), (16, 32)),
        ((16, 32), (8, 16)),
    ])
    @pytest.mark.parametrize('mode', ['bilinear', 'nearest', 'bicubic'])
    @pytest.mark.parametrize('padding_mode', ['zeros', 'border', 'reflection'])
    @pytest.mark.parametrize('align_corners', [True, False])
    def test_grid_sample_4d(self, input4d, grid4d, model):

        dummy_input = torch.zeros_like(input4d)
        dummy_grid = torch.zeros_like(grid4d)
        config = self.make_config(dummy_input, dummy_grid)
        trt_model = module2trt(
            model, args=[dummy_input, dummy_grid], config=config)

        args = [input4d, grid4d]
        with torch.inference_mode():
            gt = model(*args)
            out = trt_model(*args)
        torch.testing.assert_close(out, gt)
