from .Conv2d import *


@tensorrt_converter('mmdet.models.utils.conv_ws.ConvWS2d.forward')
def convert_ConvWS2d(ctx):
    convert_Conv2d(ctx)