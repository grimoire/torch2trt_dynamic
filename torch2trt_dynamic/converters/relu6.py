import torch
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter

from .ReLU6 import convert_ReLU6


@tensorrt_converter('torch.nn.functional.relu6')
def convert_relu6(ctx):
    ctx.method_args = (torch.nn.ReLU6(), ) + ctx.method_args
    convert_ReLU6(ctx)
