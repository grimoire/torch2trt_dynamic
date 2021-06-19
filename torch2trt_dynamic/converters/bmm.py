from ..plugins import *
from ..torch2trt_dynamic import *


@tensorrt_converter('torch.Tensor.bmm')
@tensorrt_converter('torch.bmm')
def convert_bmm(ctx):
    mat0 = ctx.method_args[0]
    mat1 = ctx.method_args[1]
    output = ctx.method_return

    mat0_trt = trt_(ctx.network, mat0)
    mat1_trt = trt_(ctx.network, mat1)

    plugin = create_torchbmm_plugin('torch_bmm_' + str(id(mat0)))

    layer = ctx.network.add_plugin_v2(
        inputs=[mat0_trt, mat1_trt], plugin=plugin)

    output._trt = layer.get_output(0)
