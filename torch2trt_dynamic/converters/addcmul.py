from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test

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
    if value!=1 or value!=1.:
        # input_value_trt = trt_(ctx.network, value)
        # output_mul_trt = ctx.network.add_elementwise(input_value_trt, output_mul_trt, trt.ElementWiseOperation.PROD).get_output(0)
        shift = np.zeros([1], np.float32)
        scale = np.array([value], np.float32)
        if len(tensor0.shape)<4:
            input_shape_trt = ctx.network.add_shape(input0_trt).get_output(0)
            add_dim = 4-len(tensor0.shape)
            add_trt = trt_(ctx.network, torch.ones([add_dim], dtype=torch.int32))
            new_input_shape_trt = ctx.network.add_concatenation([add_trt, input_shape_trt]).get_output(0)
            layer = ctx.network.add_shuffle(output_mul_trt)
            layer.set_input(1, new_input_shape_trt)
            output_mul_trt = layer.get_output(0)
        output_mul_trt = ctx.network.add_scale(output_mul_trt, trt.ScaleMode.UNIFORM, shift, scale).get_output(0)
        
        if len(tensor0.shape)<4:
            layer = ctx.network.add_shuffle(output_mul_trt)
            layer.set_input(1, input_shape_trt)
            output_mul_trt = layer.get_output(0)
    
    output_trt = ctx.network.add_elementwise(input0_trt, output_mul_trt, trt.ElementWiseOperation.SUM).get_output(0)

    output._trt = output_trt



class AddcmulTestModule(torch.nn.Module):
    def __init__(self, value):
        super(AddcmulTestModule, self).__init__()
        self.value = value

    def forward(self, x, y, z):
        return torch.addcmul(x, self.value, y, z)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 5), (1, 4, 5), (1, 4, 5)])
def test_addcmul():
    return AddcmulTestModule(2)

