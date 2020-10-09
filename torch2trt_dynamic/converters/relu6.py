from torch2trt_dynamic.torch2trt_dynamic import *
from .ReLU6 import *


@tensorrt_converter('torch.nn.functional.relu6')
def convert_relu6(ctx):
    ctx.method_args = (torch.nn.ReLU6(),) + ctx.method_args
    convert_ReLU6(ctx)