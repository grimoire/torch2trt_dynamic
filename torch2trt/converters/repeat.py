from torch2trt.torch2trt import *

from plugins import *

@tensorrt_converter('torch.Tensor.repeat')
def convert_repeat(ctx):
    input = ctx.method_args[0]
    shape = ctx.method_args[1]
    if not isinstance(shape, torch.Size):
        shape = ctx.method_args[1:]

    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    plugin = create_repeat_plugin("repeat_" + str(id(input)),
                                  repeat_shape=shape
                                  )
    
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)


@tensorrt_converter('torch.Tensor.expand_as')
def convert_expand_as(ctx):
    input = ctx.method_args[0]
    shape = ctx.method_args[1].shape
    shape = tuple(shape)
    input_shape_length = len(input.shape)
    shape = shape[:len(shape)-input_shape_length] + (1,)*input_shape_length
    
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    plugin = create_repeat_plugin("repeat_" + str(id(input)),
                                  repeat_shape=shape
                                  )
                                  
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)
    