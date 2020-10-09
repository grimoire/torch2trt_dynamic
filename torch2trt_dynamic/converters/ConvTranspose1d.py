from torch2trt_dynamic.torch2trt_dynamic import *


@tensorrt_converter('torch.nn.ConvTranspose1d.forward')
def convert_ConvTranspose1d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, 1)
    else:
        kernel_size = kernel_size + (1,)

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, 1)
    else:
        stride = stride + (1,)

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, 0)
    else:
        padding = padding + (0,)
        
    kernel = module.weight.detach().cpu().numpy()[..., None]
    
    bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()[..., None]

    # unsqueeze(3)
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = (0,0,0,1)
    input_trt = layer.get_output(0)

    # deconv
    layer = ctx.network.add_deconvolution(
        input=input_trt,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride = stride
    layer.padding = padding
    
    if module.groups is not None:
        layer.num_groups = module.groups

    output_trt = layer.get_output(0)

    # squeeze(3)
    layer = ctx.network.add_shuffle(output_trt)
    layer.reshape_dims = (0,0,0)
    output_trt = layer.get_output(0)

    output._trt = output_trt