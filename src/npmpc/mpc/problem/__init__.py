from npmpc.mpc.problem.FurutaMPC import FurutaMPC
from npmpc.mpc.problem.FurutaNPMPC import FurutaNPMPC

PROBLEM_REGISTRY = {
    ('furuta', 'equation'): FurutaMPC,
    ('furuta', 'neural'): FurutaNPMPC,
}
