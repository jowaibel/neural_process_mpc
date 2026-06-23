import torch


def create_mlp(input_dim: int, output_dim: int, layers: list, inner_activation: torch.nn.Module, final_activation: torch.nn.Module =None) -> torch.nn.Sequential:
    """Creates a multi-layer perceptron with the specified input and output dimensions and the specified number of layers."""
    mlp = torch.nn.Sequential(torch.nn.Linear(input_dim, layers[0], bias=False), inner_activation)
    for i in range(1, len(layers)):
        mlp.append(torch.nn.Linear(layers[i-1], layers[i], bias=False)).append(inner_activation)
    mlp.append(torch.nn.Linear(layers[-1], output_dim, bias=True))
    if final_activation is not None:
        mlp.append(final_activation)
    return mlp


def initialize_mlp(mlp: torch.nn.Sequential, init_func=torch.nn.init.kaiming_normal_) -> None:
    """Initializes the weights of the linear layers in the MLP using the specified initialization function."""
    for i in range(len(mlp)):
        if isinstance(mlp[i], torch.nn.Linear):
            init_func(mlp[i].weight, nonlinearity='relu')


ACTIVATIONS = {
    'GELU': torch.nn.GELU,
    'ReLU': torch.nn.ReLU,
    'Sigmoid': torch.nn.Sigmoid,
    'tanh': torch.nn.Tanh,
    'Softplus': torch.nn.Softplus,
}

def get_activation(name: str):
    """Return an activation instance, or None for unknown/None names."""
    cls = ACTIVATIONS.get(name)
    return cls() if cls is not None else None


class Scaler(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer('weight', weight.clone())
        self.register_buffer('bias', bias.clone())

    def forward(self, x):
        return x * self.weight + self.bias
