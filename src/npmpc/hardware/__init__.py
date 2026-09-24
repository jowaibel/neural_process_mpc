import os
import torch
from npmpc.utils.utils import get_project_root
from npmpc.hardware.QubeSimulator import QubeSimulator

HARDWARE_REGISTRY = {
    'sim': QubeSimulator,
}

def create_hardware(params: dict):
    root = get_project_root()
    p = torch.Tensor(params['p']) if params.get('p') is not None else None
    save_name = os.path.join(root, 'qube_servo2.npy')
    HardwareClass = HARDWARE_REGISTRY[params['target']]
    # Optional loop rate [Hz]; only passed when given, so the class default applies otherwise.
    kwargs = {'frequency': params['frequency']} if params.get('frequency') is not None else {}
    return HardwareClass(p=p, save_name=save_name, **kwargs)
