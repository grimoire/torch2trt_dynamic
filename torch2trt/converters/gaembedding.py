from torch2trt.torch2trt import *

from plugins import *
import mmdet


@tensorrt_converter('mmdet.models.plugins.GeneralizedAttention.get_position_embedding')
def convert_Repeat(ctx):
    module = ctx.method_args[0]
    input1 = ctx.method_args[1]
    input2 = ctx.method_args[2]
    q_stride = ctx.method_args[3]
    kv_stride = ctx.method_args[4]
    feat_dim = ctx.method_args[6]
    if len(ctx.method_args)>7:
        wave_length = ctx.method_args[7]
    else:
        wave_length = 1000
    position_magnitude = module.position_magnitude

    input1_trt = trt_(ctx.network, input1)
    input2_trt = trt_(ctx.network, input2)
    output = ctx.method_return

    plugin = create_gaembedding_plugin("gaemb_" + str(id(module)),
                                       position_magnitude,
                                       q_stride,
                                       kv_stride,
                                       feat_dim,
                                       wave_length)
                                       
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input1_trt, input2_trt], plugin=plugin)

    output[0]._trt = custom_layer.get_output(0)
    output[1]._trt = custom_layer.get_output(1)
