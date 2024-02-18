import tensorrt as trt

from ..torch2trt_dynamic import get_arg, tensorrt_converter, trt_

_MODE_MAP = dict(
    bilinear=trt.ResizeMode.LINEAR,
    nearest=trt.ResizeMode.NEAREST,
    bicubic=trt.ResizeMode.CUBIC)

_PAD_MODE_MAP = dict(
    zeros=trt.SampleMode.FILL,
    border=trt.SampleMode.CLAMP,
    reflection=trt.SampleMode.REFLECT)


@tensorrt_converter('torch.nn.functional.grid_sample')
def convert_grid_sample(ctx):
    input = ctx.method_args[0]
    grid = get_arg(ctx, 'grid', pos=1, default=None)
    mode = get_arg(ctx, 'mode', pos=2, default='bilinear')
    padding_mode = get_arg(ctx, 'padding_mode', pos=3, default='zeros')
    align_corners = get_arg(ctx, 'align_corners', pos=4, default=False)

    output = ctx.method_return

    input_trt = trt_(ctx.network, input)
    grid_trt = trt_(ctx.network, grid)

    mode = _MODE_MAP[mode]
    padding_mode = _PAD_MODE_MAP[padding_mode]

    layer = ctx.network.add_grid_sample(input_trt, grid_trt)
    layer.interpolation_mode = mode
    layer.sample_mode = padding_mode
    layer.align_corners = align_corners

    output._trt = layer.get_output(0)
