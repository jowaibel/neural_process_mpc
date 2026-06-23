from npmpc.dynamics import DYNAMICS_REGISTRY
from npmpc.dynamics.integrators import INTEGRATOR_REGISTRY


class DataGeneratorBase:
    params: dict

    def __init__(self, params: dict):
        self.params = params
        DynamicsClass = DYNAMICS_REGISTRY[params['system']]
        IntegratorClass = INTEGRATOR_REGISTRY[params['integrator']]
        self.oracle = IntegratorClass(DynamicsClass)

    def generate(self):
        raise NotImplementedError
        


from npmpc.data.generators.random import RandomDataGenerator
from npmpc.data.generators.furuta import FurutaDataGenerator

GENERATOR_REGISTRY = {
    'furuta': FurutaDataGenerator,
}
