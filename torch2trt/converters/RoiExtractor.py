from torch2trt.torch2trt import *

from plugins import *
import mmdet


@tensorrt_converter('mmdet.models.roi_extractors.SingleRoIExtractor.forward')
def convert_roiextractor(ctx):
    module = ctx.method_args[0]
    feats = ctx.method_args[1]
    rois = ctx.method_args[2]
    
    out_size = module.roi_layers[0].out_size[0]
    sample_num = module.roi_layers[0].sample_num
    featmap_strides = module.featmap_strides
    finest_scale = module.finest_scale

    feats_trt = [trt_(ctx.network, f) for f in feats]
    rois_trt = trt_(ctx.network, rois)
    output = ctx.method_return

    plugin = create_roiextractor_plugin("roiextractor_" + str(id(module)),
                                       out_size,
                                       sample_num,
                                       featmap_strides,
                                       finest_scale)
                                       
    custom_layer = ctx.network.add_plugin_v2(
        inputs=[rois_trt] + feats_trt, plugin=plugin)

    output._trt = custom_layer.get_output(0)
