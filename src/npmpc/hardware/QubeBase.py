import multiprocessing
import torch


class QubeBase:
    """Shared lifecycle for Qube* devices: per-instance shared memory, state reads,
    and a terminate() that signals and joins the child control process.
    Subclasses spawn their own `low_level_control_thread` and extra shared variables."""
    pendulum_type: str
    p: torch.Tensor
    save_name: str
    low_level_control_thread: multiprocessing.Process = None

    frequency = 4000

    def __init__(self, pendulum_type: str = 'furuta',
                 p: torch.Tensor = None, save_name: str = None):
        self.pendulum_type = pendulum_type
        self.p = p
        self.save_name = save_name
        # Per-instance shared memory, re-created per instance to avoid stale state across runs.
        self.ready = multiprocessing.Value('i', 0)
        self.state_value = multiprocessing.Array('d', [0.0, 0.0, 0.0, 0.0])

    def is_ready(self) -> bool:
        with self.ready.get_lock():
            return self.ready.value == 1

    def get_state(self) -> torch.Tensor:
        with self.state_value.get_lock():
            return torch.Tensor(self.state_value[:])

    def set_torque_setpoint(self, torque_setpoint: float):
        """Send torque setpoint to the servo. Hardware-specific."""
        raise NotImplementedError

    def terminate(self):
        """Signal the child loop (ready=-1) and join it so it is gone before returning."""
        with self.ready.get_lock():
            self.ready.value = -1
        if self.low_level_control_thread is not None:
            self.low_level_control_thread.join(timeout=1.0)
            if self.low_level_control_thread.is_alive():
                # Last resort if the child didn't observe the signal in time.
                self.low_level_control_thread.terminate()
                self.low_level_control_thread.join()
