"""Waypoint task only; rate and thrust references come exclusively from PX4 uORB."""
import numpy as np

from .backend import SimulatorError
from .waypoint import WaypointReference


class NativePX4Reference(WaypointReference):
    def __init__(self, config, backend):
        super().__init__(config)
        self.backend = backend

    def reset(self, state, rng):
        # Gazebo world NED and EKF local NED share axes, but not their origin.
        self.local_offset = np.asarray(state['px4_local_position']) - np.asarray(state['position_ned'])
        super().reset(state, rng)

    def _set_waypoint(self, point, position):
        super()._set_waypoint(point, position)
        self.backend.set_position_goal(self.position_target_ned + self.local_offset, self.heading)

    def command(self, state, max_thrust_ratio, rate_limit):
        # Never fall back to a Python controller or synthesize rate setpoints.
        if not state.get('px4_position_control'):
            raise SimulatorError('PX4 left armed native position/attitude control')
        age = state['px4_now_us'] - state['px4_reference_us']
        if not 0 <= age <= 50000:
            raise SimulatorError(f'Stale PX4 rate setpoint: {age} us')
        rates = np.asarray(state['px4_rate_target'], dtype=float)
        thrust_z = float(state['px4_thrust_body_z'])
        if rates.shape != (3,) or not np.isfinite(rates).all() or not np.isfinite(thrust_z):
            raise SimulatorError('Invalid PX4 rate/thrust reference')
        self._update_arrival_gate(state)
        self.last_rate_reference = rates.copy()
        self.attitude_target = np.asarray(state['px4_attitude_target'], dtype=float)
        w, x, y, z = self.attitude_target
        self.euler_target = np.array([np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                                      np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                                      np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))])
        # PX4 normalized collective thrust is linear in force; MPC_THR_HOVER
        # is fixed to the configured hover thrust during this training setup.
        self.unclipped_thrust = -thrust_z / self.backend.px4_hover_command
        if self.unclipped_thrust < 0:
            raise SimulatorError('Unexpected positive body-z thrust from PX4')
        return rates.copy(), self.unclipped_thrust
