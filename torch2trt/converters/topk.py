from torch2trt.torch2trt import *
import tensorrt as trt


@tensorrt_converter('torch.topk')
@tensorrt_converter('torch.Tensor.topk')
def convert_topk(ctx):
    support_dynamic_shape = False
    if hasattr(ctx, "support_dynamic_shape"):
        support_dynamic_shape = ctx.support_dynamic_shape

    input = ctx.method_args[0]

    if len(ctx.method_args)>1:
        k = ctx.method_args[1]
    else:
        k = ctx.method_kwargs['k']

    axis = len(input.shape)-1
    if len(ctx.method_args)>2:
        axis = ctx.method_args[2]
    if 'dim' in ctx.method_kwargs:
        axis = ctx.method_kwargs['dim']
    
    topkOp = trt.TopKOperation.MAX
    if len(ctx.method_args)>3:
        if ctx.method_args[3]==False:
            topkOp = trt.TopKOperation.MIN
    if 'largest' in ctx.method_kwargs:
        if ctx.method_args[3]==False:
            topkOp = trt.TopKOperation.MIN
    
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    
    # can only use topk on dim>=2
    need_unsqueeze = len(input_trt.shape)==1
    if need_unsqueeze:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (1,) + tuple(input_trt.shape)
        input_trt = layer.get_output(0)
        axis+=1

    if not support_dynamic_shape:
        axis-=1

    layer = ctx.network.add_topk(input_trt, topkOp, k, 1<<axis)

    output0_trt = layer.get_output(0)
    output1_trt = layer.get_output(1)

    # recovery
    if need_unsqueeze:
        layer = ctx.network.add_shuffle(output0_trt)
        layer.reshape_dims = tuple(output0_trt.shape)[1:]
        output0_trt = layer.get_output(0)

        layer = ctx.network.add_shuffle(output1_trt)
        layer.reshape_dims = tuple(output1_trt.shape)[1:]
        output1_trt = layer.get_output(0)
        
    output[0]._trt = output0_trt
    output[1]._trt = output1_trt