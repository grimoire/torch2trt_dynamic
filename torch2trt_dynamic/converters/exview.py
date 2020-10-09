from ..torch2trt_dynamic import *
from ..module_test import add_module_test
# from torch2trt.plugins import *

# def convert_exview(ctx):
#     input = ctx.method_args[0]
#     tensors = ctx.method_args[1]
#     exps = ctx.method_args[2]
#     input_trt = trt_(ctx.network, input)
#     tensors_trt = [trt_(ctx.network, t) for t in tensors]
#     output = ctx.method_return

#     plugin = create_exview_plugin("exview_" + str(id(input)), exps)

#     layer_input = [input_trt] + tensors_trt
    
#     custom_layer = ctx.network.add_plugin_v2(
#         inputs=layer_input, plugin=plugin)

#     output._trt = custom_layer.get_output(0)


def next_symbol_exview(exp, start_pos):
    if start_pos >= len(exp):
        print("next_symbol_exview out of range", exp, start_pos)
        next_pos = start_pos
        symbol = -1
        isnumber = False
        return isnumber, symbol, next_pos

    next_pos = start_pos+1    
    if exp[start_pos]<'0' or exp[start_pos]>'9':
        symbol = exp[start_pos]
        isnumber = False
        return isnumber, symbol, next_pos
    
    symbol = int(exp[start_pos])
    while next_pos<len(exp) and exp[next_pos]>='0' and exp[next_pos]<='9':
        symbol = symbol*10 + int(exp[next_pos])
        next_pos += 1
    isnumber = True
    return isnumber, int(symbol), next_pos

def get_value_exview_impl(ctx, exp, inputs, start_pos):
    if start_pos >= len(exp):
        print("get_value_exview_impl out of range", exp, start_pos)
        return None, None
    
    isnumber, symbol, next_pos = next_symbol_exview(exp, start_pos)

    if isnumber:
        return trt_(ctx.network, torch.tensor([symbol],dtype=torch.int32).cuda(0)), next_pos
    if symbol.isalpha():
        desc_id = ord(symbol.lower())-ord('a')
        isnumber, symbol, next_pos = next_symbol_exview(exp, next_pos)
        if not isnumber:
            print("wrong expression1:", exp, "with symbol:", symbol)
            return None, next_pos
        return ctx.network.add_slice(inputs[desc_id],[symbol],[1],[1]).get_output(0), next_pos
    elif symbol == '(':
        result = parse_exview_string_impl(ctx, exp, inputs, start_pos+1)
        if next_pos>=len(exp) or exp[next_pos]!=')':
            print("wrong expression2:", exp, "with symbol:", symbol)
            return None, next_pos
        return result, next_pos+1

    else:
        print("wrong expression3:", exp, "with symbol:", symbol)
        return None, next_pos

    

def parse_exview_string_impl(ctx, exp, inputs, start_pos):
    if start_pos >= len(exp):
        print("parse_exview_string_impl out of range", exp, start_pos)
        return None, None

    return_value, next_pos = get_value_exview_impl(ctx, exp, inputs, start_pos)
    if return_value is None:
        return None, next_pos

    for _ in range(next_pos, len(exp)):
        isnumber, symbol, next_pos = next_symbol_exview(exp, next_pos)

        chr_sym = str(symbol)
        if not isnumber and chr_sym==')':
            next_pos-=1
            break

        result, next_pos = get_value_exview_impl(ctx, exp, inputs, next_pos)

        elementwise_op = None
        if chr_sym == "+":
            elementwise_op = trt.ElementWiseOperation.SUM
        if chr_sym == "-":
            elementwise_op = trt.ElementWiseOperation.SUB
        if chr_sym == "*":
            elementwise_op = trt.ElementWiseOperation.PROD
        if chr_sym == "/":
            elementwise_op = trt.ElementWiseOperation.FLOOR_DIV
        
        if elementwise_op is not None:
            return_value = ctx.network.add_elementwise(return_value, result, elementwise_op).get_output(0)
        if next_pos>=len(exp):
            break
    
    return return_value, next_pos


def parse_exview_string(ctx, exp, tensors_shape_trt):
    result, _ = parse_exview_string_impl(ctx, exp, tensors_shape_trt, 0)
    return result

def convert_exview(ctx):
    input = ctx.method_args[0]
    tensors = ctx.method_args[1]
    exps = ctx.method_args[2]
    input_trt = trt_(ctx.network, input)
    tensors_trt = [trt_(ctx.network, t) for t in tensors]
    output = ctx.method_return

    tensors_shape_trt = [ctx.network.add_shape(t).get_output(0) for t in tensors_trt]
    
    shape_trt = [parse_exview_string(ctx, exp, tensors_shape_trt) for exp in exps]
    shape_trt = ctx.network.add_concatenation(shape_trt).get_output(0)
    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, shape_trt)

    output._trt = layer.get_output(0)