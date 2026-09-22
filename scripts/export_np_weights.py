"""Export the mu_decoder + scaler weights of a trained NeuralProcess checkpoint
to a YAML file, for the C++ port (src_cpp/npmpc/nps) to load via yaml-cpp.

Only what NeuralProcess.casadi_decoder needs is exported: the mu_decoder MLP
layers (weight/bias), and the x_direct_scaler / y_inverse_scaler vectors.
The encoder and sigma_decoder are not part of the CasADi decoder path and are
not exported.

No YAML library is required: the file format here (mappings, and flow-style
`[...]` lists of floats/rows) is simple enough to emit by hand.
"""
import argparse
from pathlib import Path

import torch


def _flow_list(values) -> str:
    return '[' + ', '.join(repr(float(v)) for v in values) + ']'


def _indent(text: str, spaces: int) -> str:
    prefix = ' ' * spaces
    return '\n'.join(prefix + line for line in text.splitlines())


def _mlp_layer_yaml(weight: list, bias: list) -> str:
    rows = '\n'.join(f'  - {_flow_list(row)}' for row in weight)
    return f'weight:\n{rows}\nbias: {_flow_list(bias)}'


def export_np_weights(checkpoint_path: str, output_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    params = checkpoint['params']
    decoder_sd = checkpoint['decoder']

    layers = []
    layer_id = 0
    while f'mu_decoder.{layer_id}.weight' in decoder_sd:
        weight = decoder_sd[f'mu_decoder.{layer_id}.weight']
        bias_key = f'mu_decoder.{layer_id}.bias'
        bias = decoder_sd[bias_key] if bias_key in decoder_sd else torch.zeros(weight.shape[0])
        layers.append((weight.tolist(), bias.tolist()))
        layer_id += 2

    sizes = params['sizes']
    x_scaler = checkpoint['x_direct_scaler']
    y_scaler = checkpoint['y_inverse_scaler']
    cnp_decoder = params['cnp_decoder']

    layers_yaml = '\n'.join(
        '- ' + _mlp_layer_yaml(weight, bias).replace('\n', '\n  ')
        for weight, bias in layers
    )

    yaml_text = f"""\
sizes:
  history: {sizes['history']}
  x: {sizes['x']}
  y: {sizes['y']}
  z: {sizes['z']}
cnp_decoder:
  inner_activation: {cnp_decoder.get('inner_activation', 'GELU')}
  mu_activation: {cnp_decoder.get('mu_activation', 'None')}
x_direct_scaler:
  weight: {_flow_list(x_scaler['weight'].tolist())}
  bias: {_flow_list(x_scaler['bias'].tolist())}
y_inverse_scaler:
  weight: {_flow_list(y_scaler['weight'].tolist())}
  bias: {_flow_list(y_scaler['bias'].tolist())}
mu_decoder_layers:
{_indent(layers_yaml, 2)}
"""

    Path(output_path).write_text(yaml_text)
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='model/model.pth')
    parser.add_argument('--output', default='model/np_weights.yaml')
    args = parser.parse_args()
    export_np_weights(args.checkpoint, args.output)
