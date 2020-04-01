from torch2trt.torch2trt import *

@tensorrt_converter('torch.addcmul')
@tensorrt_converter('torch.Tensor.addcmul')
def convert_addcmul(ctx):
    tensor0 = ctx.method_args[0]

    value = 1
    next_tensor_offset = 0
    if len(ctx.method_args)==4:
        value = ctx.method_args[1]
        next_tensor_offset = 1
    if 'value' in ctx.method_kwargs:
        value = ctx.method_kwargs['value']

    tensor1 = ctx.method_args[1+next_tensor_offset]
    tensor2 = ctx.method_args[2+next_tensor_offset]

    input0_trt, input1_trt, input2_trt = trt_(ctx.network, tensor0, tensor1, tensor2)
    output = ctx.method_return

    output_mul_trt = ctx.network.add_elementwise(input1_trt, input2_trt, trt.ElementWiseOperation.PROD).get_output(0)
    if value!=1:
        input_value_trt = trt_(ctx.network, value)
        output_mul_trt = ctx.network.add_elementwise(input_value_trt, output_mul_trt, trt.ElementWiseOperation.PROD).get_output(0)
    
    output_trt = ctx.network.add_elementwise(input0_trt, output_mul_trt, trt.ElementWiseOperation.SUM).get_output(0)

    output._trt = output_trt