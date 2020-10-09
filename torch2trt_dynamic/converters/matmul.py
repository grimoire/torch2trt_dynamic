from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.module_test import add_module_test
import tensorrt as trt


@tensorrt_converter('torch.matmul')
@tensorrt_converter('torch.Tensor.matmul')
def convert_matmul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt = trt_(ctx.network, input_a)
    input_b_trt = trt_(ctx.network, input_b)
    # input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return

    need_reshape = (len(input_a.shape)>2 and len(input_b.shape)==1) or (len(input_b.shape)>2 and len(input_a.shape)==1)

    # change shape of input a
    if need_reshape:
        old_a_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_a_trt)
        old_b_shape_trt = tensor_trt_get_shape_trt(ctx.network, input_b_trt)
        matmul_dim_trt = slice_shape_trt(ctx.network, old_a_shape_trt, len(input_a.shape)-1)
        minus_one_trt = trt_(ctx.network, torch.tensor([-1], dtype=torch.int32).cuda())
        prefix_shape_trt = None

        if len(input_a.shape)>1:
            prefix_a_trt = slice_shape_trt(ctx.network, old_a_shape_trt, len(input_a.shape)-2, 1)
        else:
            prefix_a_trt = None

        if len(input_b.shape)>1:
            prefix_b_trt = slice_shape_trt(ctx.network, old_b_shape_trt, len(input_b.shape)-1, 1)
        else:
            prefix_b_trt = None

        if len(input_a.shape)>2:
            prefix_shape_trt = slice_shape_trt(ctx.network, old_a_shape_trt, 0, len(input_a.shape)-2)
            matmul_a_shape_trt = ctx.network.add_concatenation([minus_one_trt, matmul_dim_trt]).get_output(0)
            layer = ctx.network.add_shuffle(input_a_trt)
            layer.set_input(1, matmul_a_shape_trt)
            input_a_trt = layer.get_output(0)
        
        if len(input_b.shape)>2:
            if prefix_shape_trt is None:
                prefix_shape_trt = slice_shape_trt(ctx.network, old_b_shape_trt, 0, len(input_b.shape)-2)
            matmul_b_shape_trt = ctx.network.add_concatenation([matmul_dim_trt, minus_one_trt]).get_output(0)
            layer = ctx.network.add_shuffle(input_b_trt)
            layer.first_transpose = tuple([len(input_b.shape)-2] + list(range(len(input_b.shape)-2)) + [len(input_b.shape)-1])
            layer.set_input(1, matmul_b_shape_trt)
            input_b_trt = layer.get_output(0)

    # vector or mat
    mm_a_op = trt.MatrixOperation.NONE
    mm_b_op = trt.MatrixOperation.NONE
    if len(input_a_trt.shape)==1:
        mm_a_op = trt.MatrixOperation.VECTOR
    if len(input_b_trt.shape)==1:
        mm_b_op = trt.MatrixOperation.VECTOR
    layer = ctx.network.add_matrix_multiply(input_a_trt, mm_a_op, input_b_trt, mm_b_op)
    matmul_output = layer.get_output(0)

    # recovery shape
    if need_reshape:
        new_shape_trt = [prefix_shape_trt]

        if prefix_a_trt is not None:
            new_shape_trt.append(prefix_a_trt)
        
        if prefix_b_trt is not None:
            new_shape_trt.append(prefix_b_trt)
        if len(new_shape_trt)>1:
            new_shape_trt = ctx.network.add_concatenation(new_shape_trt).get_output(0)
        else:
            new_shape_trt = new_shape_trt[0]

        layer = ctx.network.add_shuffle(matmul_output)
        layer.set_input(1, new_shape_trt)
        matmul_output = layer.get_output(0)

    output._trt = matmul_output


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