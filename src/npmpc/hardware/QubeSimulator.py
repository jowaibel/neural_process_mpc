import logging
import multiprocessing
import time
import numpy as np
import torch

from npmpc.dynamics import DYNAMICS_REGISTRY
from npmpc.dynamics.integrators import INTEGRATOR_REGISTRY
from npmpc.hardware.QubeBase import QubeBase
from npmpc.utils.utils import attach_logging

logger = logging.getLogger(__name__)


class QubeSimulator(QubeBase):
    Kt = 0.042

    current_torque: float
    samples = 100

    def __init__(self, pendulum_type: str = 'furuta', p: torch.Tensor = None, save_name: str = None):
        super().__init__(pendulum_type, p, save_name)
        DynamicsClass = DYNAMICS_REGISTRY[self.pendulum_type]
        IntegratorClass = INTEGRATOR_REGISTRY.get('midpoint')
        self.integrator = IntegratorClass(DynamicsClass)

        # Sim-specific shared memory (`ready`/`state_value` come from QubeBase).
        self.torque_setpoint = multiprocessing.Value('d', 0.0)

        self.low_level_control_thread = multiprocessing.Process(
            target=self.low_level_controller,
            args=(self.state_value, self.torque_setpoint, self.ready))
        self.low_level_control_thread.start()

        while not self.is_ready():
            time.sleep(0.1)


    def set_torque_setpoint(self, torque_setpoint: float):
        with self.torque_setpoint.get_lock():
            self.torque_setpoint.value = torque_setpoint


    def low_level_controller(self, state_value, torque_setpoint, ready):
        attach_logging()
        save_file = None
        if self.save_name is not None:
            logger.info("Buffer saving started")
            save_file = open(self.save_name, "wb")

        logger.info("Simulator started")
        time_vector = time.time() * np.ones((self.samples, 1), dtype=np.float64)
        state_vector = torch.Tensor([[torch.pi, 0.0, 0.0, 0.0]] * self.samples)
        torque_vector = torch.Tensor([[0.0]] * self.samples)
        voltage_vector = torch.Tensor([[0.0]] * self.samples)

        index = 0

        with ready.get_lock():
            ready.value = 1

        while True:
            time_vector[index] = time.time()
            delta_t = time_vector[index, 0] - time_vector[((index - 1) % self.samples), 0]
            if delta_t < 1 / self.frequency:
                time.sleep(1 / self.frequency - delta_t)
                time_vector[index] = time.time()
                delta_t = time_vector[index, 0] - time_vector[((index - 1) % self.samples), 0]

            # Check termination condition ready == -1
            with ready.get_lock():
                tmp = ready.value
            if tmp == -1:
                if self.save_name is not None:
                    save_file.close()
                logger.info("Simulator terminated")
                return
            # Get the torque setpoint
            with torque_setpoint.get_lock():
                torque = torque_setpoint.value
            torque_vector[index] = torque

            step = self.integrator.integrate(x0=state_vector[((index - 1) % self.samples)],
                                             u=torque_vector[index].unsqueeze(-1),
                                             dt=delta_t,
                                             p=self.p)
            state_vector[index] = step[-1]

            with state_value.get_lock():
                state_value[:] = state_vector[index].numpy()

            if self.save_name is not None and index == self.samples - 1:
                np.concatenate((time_vector, state_vector, torque_vector, voltage_vector), axis=-1).tofile(save_file)

            index = (index + 1) % self.samples
