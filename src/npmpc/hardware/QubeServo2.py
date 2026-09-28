import logging
import multiprocessing
import time
import torch
from quanser.hardware import HIL, HILError
import numpy as np

from npmpc.dynamics.furuta import FurutaDynamics
from npmpc.hardware.QubeBase import QubeBase
from npmpc.utils.utils import attach_logging

logger = logging.getLogger(__name__)

    
class QubeServo2(QubeBase):
    
    Kproportional = 2.5
    Kintegral = 2500 
    
    # Torque constant 
    Kt = 0.042
    R = 8.4
    
    # Encoder channels
    r_encoder_channels = np.array([0,1], dtype=np.uint32)
    r_num_encoder_channels = len(r_encoder_channels)

    # Motor enable channel
    w_digital_channels = np.array([0], dtype=np.uint32)
    w_num_digital_channels = len(w_digital_channels)
    w_digital_buffer = np.ones(w_digital_channels.shape, dtype=np.int8)

    # Motor Current channels
    r_analog_channels = np.array([0], dtype=np.uint32)
    r_num_analog_channels = len(r_analog_channels)

    # Motor Voltage Command channels
    w_command_channels = np.array([0], dtype=np.uint32)
    w_num_command_channels = len(w_command_channels)

    # LED color channels
    w_color_channels = np.array([11000, 11001, 11002], dtype=np.uint32)
    w_num_color_channels = len(w_color_channels)

    # Motor tachometer channels
    r_speed_channels = np.array([14000], dtype=np.int32)
    r_num_speed_channels = len(r_speed_channels)

    samples = 100


    def __init__(self, pendulum_type='furuta', p: torch.Tensor=None, save_name=None,
                 frequency: float = None):
        super().__init__(pendulum_type, p, save_name)
        if self.pendulum_type != 'furuta':
            raise ValueError('Pendulum type not implemented')

        # Polling rate of the current-control loop [Hz]; defaults to QubeBase.frequency.
        if frequency is not None:
            self.frequency = frequency

        # Servo-specific shared memory (common `ready`/`state_value` come from QubeBase).
        self.current_value    = multiprocessing.Value('d', 0.0)
        self.current_setpoint = multiprocessing.Value('d', 0.0)

        self.low_level_control_thread = multiprocessing.Process(
            target=self.low_level_controller,
            args=(self.state_value, self.current_value, self.current_setpoint, self.ready))

        self.low_level_control_thread.start()

        while not self.is_ready():
            time.sleep(0.1)


    def set_current_setpoint(self, current_setpoint):
        with self.current_setpoint.get_lock():
            self.current_setpoint.value = current_setpoint


    def set_torque_setpoint(self, torque_setpoint):
        self.set_current_setpoint(torque_setpoint / self.Kt)


    def get_current(self):
        with self.current_value.get_lock():
            return self.current_value.value
        
    
    def low_level_controller(self, state_value, current_value, current_setpoint, ready):
        attach_logging()
        with ready.get_lock():
            ready.value = 0
        save_file = None
        if self.save_name is not None:
            logger.info("Buffer saving started")
            save_file = open(self.save_name, "wb")

        logger.info("Current controller started")
        qube = HIL()
        try:
            # Connect to the Quanser Qube Servo 2
            HIL.close_all()
            qube.open("qube_servo2_usb")
            # Enable the motor
            qube.write_digital(self.w_digital_channels, self.w_num_digital_channels, self.w_digital_buffer)
            # Set the encoder counts to pi
            counts = np.array([0,-1024], dtype=np.int32)
            qube.set_encoder_counts(self.r_encoder_channels, self.r_num_encoder_channels, counts)
            
            # Wait for the position encoder to be active
            tmp = np.zeros((self.r_num_encoder_channels), dtype=np.int32)
            while tmp[1] != -1024:
                qube.read_encoder(self.r_encoder_channels, self.r_num_encoder_channels, tmp)
                
            # Set led color to green
            qube.write_other(self.w_color_channels, self.w_num_color_channels, np.array([0, 1, 0], dtype=np.float64))
            
        except HILError as e:
            logger.error(e.get_error_message())
            return
            
        integral_component = 0.0
        if self.pendulum_type == 'furuta':
            state_vector, covariance_matrix = self.furuta_state_initializer()
        else:
            state_vector, covariance_matrix = None, None
            
        index = 0; active = False
        
        time_vector = time.time() * np.ones((self.samples,1), dtype=np.float64)
        encoder_vector = counts[0] * np.ones((self.samples,self.r_num_encoder_channels), dtype=np.int32)
        
        current_vector = np.zeros((self.samples,1), dtype=np.float64)
        voltage_vector = np.zeros((self.samples,1), dtype=np.float64)

        while True:
            time_vector[index] = time.time()
            delta_t = time_vector[index,0] - time_vector[((index-1) % self.samples),0]
            if delta_t < 1/self.frequency:
                time.sleep(1/self.frequency - delta_t)
                time_vector[index] = time.time()
                delta_t = time_vector[index,0] - time_vector[((index-1) % self.samples),0]
            
            try:
                qube.read(analog_channels=self.r_analog_channels, num_analog_channels=self.r_num_analog_channels, analog_buffer=current_vector[index], 
                          encoder_channels=self.r_encoder_channels, num_encoder_channels=self.r_num_encoder_channels, encoder_buffer=encoder_vector[index])
                # Invert pendulum axis direction 
                encoder_vector[index,1] = -encoder_vector[index,1]
            except HILError as e:
                logger.error(e.get_error_message())

            # Calculate the encoder velocity and position

            state_vector[index], covariance_matrix[index] = self.furuta_kalman_filter(state_vector[((index-1) % self.samples)], 
                                                                                      covariance_matrix[((index-1) % self.samples)], 
                                                                                      encoder_vector[index], current_vector[index], delta_t)
                
            # Run current controller
            voltage_vector[index], integral_component = self.current_controller(current_vector[index], current_setpoint, integral_component, delta_t, 0)
            
            # Check termination condition ready == -1
            with self.ready.get_lock():
                tmp = ready.value
            if tmp == -1:
                # Reset the current setpoint and ready flag
                with current_setpoint.get_lock(), ready.get_lock():
                    current_setpoint.value = 0.0
                    ready.value = 0
                    
                # Close the file
                if self.save_name is not None:
                    save_file.close()
                # Write 0 voltage to the motor
                qube.write_analog(self.w_command_channels, self.w_num_command_channels, np.array([0], dtype=np.float64))
                # Disable the motor
                qube.write_digital(self.w_digital_channels, self.w_num_digital_channels, np.zeros(self.w_digital_channels.shape, dtype=np.int8))
                # Set led color to green
                qube.write_other(self.w_color_channels, self.w_num_color_channels, np.array([1, 0, 0], dtype=np.float64))
                # Close the connection
                qube.close()
                logger.info("Current controller terminated")
                return
        
            # Send the voltage command to the motor
            try:
                if active:
                    qube.write_analog(self.w_command_channels, self.w_num_command_channels, voltage_vector[index])
            except HILError as e:
                logger.error(e.get_error_message())

            if self.save_name is not None and index == self.samples - 1:
                np.concatenate((time_vector, encoder_vector, state_vector, current_vector*self.Kt, voltage_vector), axis=-1).tofile(save_file)
                
            # Write the values to the shared memory
            with state_value.get_lock(), current_value.get_lock():
                state_value[:] = state_vector[index].numpy()
                current_value.value = current_vector[index]
                
            # Do not send command for first self.samples to allow the Kalman filter to converge 
            if index == self.samples - 1:
                active = True
                # All the samples are collected, set the ready flag
                with ready.get_lock():
                    ready.value = 1
                    
            index = (index + 1) % self.samples
            
    
    def current_controller(self, actual_current, current_setpoint, integral_component, delta_t, velocity):
        # Run current controller
        with current_setpoint.get_lock():
            tmp = current_setpoint.value
        current_error = tmp - actual_current[0]
        feed_forward_term = self.Kt * velocity + self.R * tmp
        integral_component = np.clip(integral_component + current_error * delta_t, -15/self.Kintegral, 15/self.Kintegral)
        return np.clip(feed_forward_term + self.Kproportional*current_error + self.Kintegral*integral_component, -15, 15), integral_component
    
    
    def furuta_state_initializer(self):
        # initializee the 2D pendulum dynamics
        self.dynamics = FurutaDynamics
        
        # initialize the state vector
        state_vector = torch.Tensor([[np.pi, 0.0, 0.0, 0.0]]*self.samples)
        covariance_matrix = torch.Tensor([[[1000, 0.0, 0.0, 0.0],[0.0, 1000, 0.0, 0.0],[0.0, 0.0, 1000, 0.0],[0.0, 0.0, 0.0, 1000]]]*self.samples)
        
        # initialize H, Q, R matrices
        self.kalman_H = torch.Tensor([[1, 0, 0, 0],[0, 1, 0, 0]])
        self.kalman_Q = torch.diag_embed(torch.Tensor([0,0,50*np.pi,50*np.pi]))
        self.kalman_R = 2 * torch.eye(2) * np.pi / 1024
        
        return state_vector, covariance_matrix
        
        
    def furuta_kalman_filter(self, state_vector, covariance_matrix, position_update, current_update, delta_time):
        # Initializations
        F = torch.eye(4) + torch.Tensor([[0, 0, delta_time, 0],[0, 0, 0, delta_time],[0, 0, 0, 0],[0, 0, 0, 0]])
        torque_update = torch.Tensor(self.Kt * current_update)
        position_update = torch.Tensor(np.array([position_update[::-1]])*np.pi/1024)
        
        # Prediction phase - state update
        state_dot = self.dynamics.dynamics(x=state_vector, u=torque_update, p=self.p)
        state_vector = state_vector + state_dot * delta_time
        
        # Prediction phase - covariance update
        covariance_matrix = F @ covariance_matrix @ F.T + self.kalman_Q * delta_time
        
        # Update phase - Kalman gain
        position_error = position_update - state_vector[...,0:2]
        position_noise = self.kalman_H @ covariance_matrix @ self.kalman_H.T + self.kalman_R
        kalman_gain = covariance_matrix @ self.kalman_H.T @ position_noise.inverse()
        
        # Update phase - state update
        state_vector = state_vector + (position_error @ kalman_gain.T).squeeze(0)
        
        # Update phase - covariance update
        covariance_matrix = (torch.eye(4) - kalman_gain @ self.kalman_H) @ covariance_matrix
        
        return state_vector, covariance_matrix
    