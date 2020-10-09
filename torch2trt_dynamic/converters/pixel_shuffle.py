from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.pixel_shuffle')
def convert_pixel_shuffle(ctx):

    input = ctx.method_args[0]
    upscale_factor = get_arg(ctx, "upscale_factor", pos=1, default=None)

    input_trt = trt_(ctx.network, input)
    input_shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    output = ctx.method_return

    batch_shape_trt = ctx.network.add_slice(
        input_shape_trt, [0], [1], [1]).get_output(0)
    channel_shape_trt = ctx.network.add_slice(
        input_shape_trt, [1], [1], [1]).get_output(0)
    height_shape_trt = ctx.network.add_slice(
        input_shape_trt, [2], [1], [1]).get_output(0)
    width_shape_trt = ctx.network.add_slice(
        input_shape_trt, [3], [1], [1]).get_output(0)

    upscale_shape_trt = trt_(ctx.network, torch.tensor(
        [upscale_factor], dtype=torch.int32).to(input.device))
    upscale_p2_trt = ctx.network.add_elementwise(
        upscale_shape_trt, upscale_shape_trt, trt.ElementWiseOperation.PROD).get_output(0)
    new_channel_shape_trt = ctx.network.add_elementwise(
        channel_shape_trt, upscale_p2_trt, trt.ElementWiseOperation.FLOOR_DIV).get_output(0)

    # (b, c0, s, s, h, w)
    pre_shape_trt = ctx.network.add_concatenation([batch_shape_trt,
                                                   new_channel_shape_trt,
                                                   upscale_shape_trt,
                                                   upscale_shape_trt,
                                                   height_shape_trt,
                                                   width_shape_trt]).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, pre_shape_trt)
    layer.second_transpose = (0,1,4,2,5,3)

    permute_trt = layer.get_output(0)

    new_height_shape_trt = ctx.network.add_elementwise(
        height_shape_trt, upscale_shape_trt, trt.ElementWiseOperation.PROD).get_output(0)
    new_width_shape_trt = ctx.network.add_elementwise(
        width_shape_trt, upscale_shape_trt, trt.ElementWiseOperation.PROD).get_output(0)

    post_shape_trt = ctx.network.add_concatenation([batch_shape_trt,
                                                   new_channel_shape_trt,
                                                   new_height_shape_trt,
                                                   new_width_shape_trt]).get_output(0)
    
    layer = ctx.network.add_shuffle(permute_trt)
    layer.set_input(1, post_shape_trt)
    output._trt = layer.get_output(0)
