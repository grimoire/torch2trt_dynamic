from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
import tensorrt as trt


@tensorrt_converter('torch.matmul')
def convert_matmul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return

    mm_op = trt.MatrixOperation.NONE
    layer = ctx.network.add_matrix_multiply(input_a_trt, mm_op, input_b_trt, mm_op)
    output._trt = layer.get_output(0)


class MatmulTest(torch.nn.Module):
    def __init__(self):
        super(MatmulTest, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)



@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 6), (1, 2, 6, 7)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6), (1, 2, 6, 7)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 6), (1, 6, 7)])
# @add_module_test(torch.float32, torch.device('cuda'), [(4, 6), (6, 7)])
def test_matmul():
    return MatmulTest()