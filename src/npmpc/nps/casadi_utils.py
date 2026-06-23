from typing import OrderedDict
import casadi as ca


def create_casadi_scaler(casadi_input: ca.MX, scaler_weights: OrderedDict) -> ca.MX:
    weight = ca.DM(scaler_weights['weight'].numpy())
    bias = ca.DM(scaler_weights['bias'].numpy())
    casadi_output = casadi_input * weight.T + bias.T
    return casadi_output

    
def create_casadi_mlp(casadi_input: ca.MX, name: str, network: dict, network_weights: OrderedDict) -> ca.MX:
    inner_activation = get_casadi_activation(network['inner_activation'])
    outer_activation = get_casadi_activation(network['mu_activation'])
    layer_id = 0
    while True:
        if f'{name}.{layer_id}.weight' in network_weights.keys():
            layer_weights = ca.DM(network_weights[f'{name}.{layer_id}.weight'].numpy())
        else: 
            break
        if f'{name}.{layer_id}.bias' in network_weights.keys():
            layer_bias = ca.DM(network_weights[f'{name}.{layer_id}.bias'].numpy())
        else:
            layer_bias = ca.DM.zeros(layer_weights.size1())
        linear_output = casadi_linear_layer(casadi_input, layer_weights, layer_bias)
        layer_id += 2
        if f'{name}.{layer_id}.weight' in network_weights.keys():
            casadi_input = inner_activation(linear_output)
        else:
            casadi_input = linear_output
    casadi_output = outer_activation(casadi_input)
    return casadi_output


def get_casadi_activation(activation: str):
    if activation == 'GELU':
        return casadi_gelu_layer
    elif activation == 'ReLU':
        return casadi_relu_layer
    elif activation == 'Sigmoid':
        return casadi_sigmoid_layer
    else:
        return casadi_no_layer
    
    
def casadi_linear_layer(input: ca.MX, weights: ca.MX, bias: ca.MX) -> ca.MX:
    return ca.mtimes(input, weights.T) + bias.T
    
def casadi_gelu_layer(input: ca.MX) -> ca.MX:
    return input * 0.5 * (1 + ca.erf(input / ca.sqrt(2)))

def casadi_relu_layer(input: ca.MX) -> ca.MX:
    return ca.fmax(0, input)

def casadi_sigmoid_layer(input: ca.MX) -> ca.MX:
    return 1 / (1 + ca.exp(-input))

def casadi_no_layer(input: ca.MX) -> ca.MX:
    return input

def casadi_softplus_layer(input: ca.MX) -> ca.MX:
    return ca.log(1 + ca.exp(input))