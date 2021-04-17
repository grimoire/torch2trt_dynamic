import tensorrt as trt
from .permute import convert_permute
from torch2trt_dynamic.torch2trt_dynamic import *


def set_gate_parameter(func, index, gate_type_list, hidden_size, param_i,
                       param_h):
    for i, gate_type in enumerate(gate_type_list):
        func(index, gate_type, True,
             param_i[i * hidden_size:(i + 1) * hidden_size].copy())
        func(index, gate_type, False,
             param_h[i * hidden_size:(i + 1) * hidden_size].copy())


# from https://github.com/grimoire/torch2trt_dynamic/issues/19#issuecomment-817265423
# Thank you @iAlexKai
@tensorrt_converter('torch.nn.GRU.forward')
def convert_GRU(ctx):

    module = ctx.method_args[0]
    input_tensor = ctx.method_args[1]
    if len(ctx.method_args) == 3:
        init_state_tensor = ctx.method_args[2]
    output_0, output_1 = ctx.method_return[0], ctx.method_return[1]

    input_tensor_trt = trt_(ctx.network, input_tensor)
    init_state_tensor_trt = trt_(ctx.network, init_state_tensor)

    layer_count = module.num_layers
    hidden_size = module.hidden_size
    batch_first = module.batch_first
    if batch_first:
        max_seq_length = input_tensor.shape[1]
    else:
        permute_input_tensor = input_tensor.permute(1, 0, 2)
        convert_with_args(ctx, convert_permute, [input_tensor, 1, 0, 2], {},
                          permute_input_tensor)
        input_tensor_trt = trt_(ctx.network, permute_input_tensor)
        max_seq_length = permute_input_tensor.shape[1]

    permute_init_state_tensor = init_state_tensor.permute(1, 0, 2)
    convert_with_args(ctx, convert_permute, [init_state_tensor, 1, 0, 2], {},
                      permute_init_state_tensor)
    init_state_tensor_trt = trt_(ctx.network, permute_init_state_tensor)
    op = trt.RNNOperation.GRU
    layer = ctx.network.add_rnn_v2(input_tensor_trt, layer_count, hidden_size,
                                   max_seq_length, op)
    if len(ctx.method_args) == 3:
        layer.hidden_state = init_state_tensor_trt

    if module.bidirectional is True:
        layer.direction = trt.RNNDirection.BIDIRECTION

    gate_type_list = [
        trt.RNNGateType.RESET, trt.RNNGateType.UPDATE, trt.RNNGateType.HIDDEN
    ]
    for i in range(layer_count):
        iw = getattr(module, "weight_ih_l%s" % i).detach().cpu().numpy()
        hw = getattr(module, "weight_hh_l%s" % i).detach().cpu().numpy()

        rela_index = 2 * i if module.bidirectional is True else i

        set_gate_parameter(layer.set_weights_for_gate,
                           rela_index,
                           gate_type_list,
                           hidden_size=hidden_size,
                           param_i=iw,
                           param_h=hw)

        ib = getattr(module, "bias_ih_l%s" % i).detach().cpu().numpy()
        hb = getattr(module, "bias_hh_l%s" % i).detach().cpu().numpy()

        set_gate_parameter(layer.set_bias_for_gate,
                           rela_index,
                           gate_type_list,
                           hidden_size=hidden_size,
                           param_i=ib,
                           param_h=hb)

        if module.bidirectional is True:
            # ================reverse=====================
            iw_r = getattr(module,
                           "weight_ih_l%s_reverse" % i).detach().cpu().numpy()
            hw_r = getattr(module,
                           "weight_hh_l%s_reverse" % i).detach().cpu().numpy()

            set_gate_parameter(layer.set_weights_for_gate,
                               2 * i + 1,
                               gate_type_list,
                               hidden_size=hidden_size,
                               param_i=iw_r,
                               param_h=hw_r)

            ib_r = getattr(module,
                           "bias_ih_l%s_reverse" % i).detach().cpu().numpy()
            hb_r = getattr(module,
                           "bias_hh_l%s_reverse" % i).detach().cpu().numpy()

            set_gate_parameter(layer.set_bias_for_gate,
                               2 * i + 1,
                               gate_type_list,
                               hidden_size=hidden_size,
                               param_i=ib_r,
                               param_h=hb_r)

    gru_output_0 = layer.get_output(0)
    gru_output_1 = layer.get_output(1)
    if batch_first:
        output_0._trt = gru_output_0
    else:
        permuted_output_0 = output_0.permute(1, 0, 2)
        permuted_output_0._trt = gru_output_0
        convert_with_args(ctx, convert_permute, [permuted_output_0, 1, 0, 2],
                          {}, output_0)
    permuted_output_1 = output_1.permute(1, 0, 2)
    permuted_output_1._trt = gru_output_1
    convert_with_args(ctx, convert_permute, [permuted_output_1, 1, 0, 2], {},
                      output_1)
