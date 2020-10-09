from ..torch2trt_dynamic import *
from ..plugins import *


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

    if mode == 'bilinear':
        mode = trt.ResizeMode.LINEAR
    elif mode == 'nearest':
        mode = trt.ResizeMode.NEAREST
    
    if padding_mode == 'zeros':
        padding_mode = 0
    elif padding_mode == 'border':
        padding_mode = 1
    elif padding_mode == 'reflection':
        padding_mode = 2

    plugin = create_gridsample_plugin("torch_gridsample_"+str(id(input)),
                                        mode=mode,
                                        padding_mode=padding_mode,
                                        align_corners=align_corners)
            
    layer = ctx.network.add_plugin_v2(
        inputs=[input_trt, grid_trt], plugin=plugin)

    output._trt = layer.get_output(0)