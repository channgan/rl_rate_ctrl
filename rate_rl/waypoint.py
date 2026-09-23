"""Position PD -> attitude/rate and thrust references; no motor allocation."""
import numpy as np
from .outer_loop import AltitudeAttitudeReference, euler_quaternion, wrap_yaw


def waypoint_gate_metadata():
    return dict(version='instant_arrival_mean_abs_tracking_error_fixed_heading_v6', distance_strictly_below_m=.2,
        tracking_error_mean_abs_strictly_below_deg_s=5.,
        consecutive_seconds=5., distance_required_throughout=False,
        feedback='post-action actual rates and the reference issued for that action',
        reset='any rate-error condition fails, waypoint switch or episode reset; distance does not reset timer',
        heading='set once from position to new waypoint; hold until next waypoint; retain heading for vertical legs')


class WaypointReference(AltitudeAttitudeReference):
    tracking_hold_seconds = 5.0
    tracking_tolerance_deg_s = 5.0

    def reset(self, state, rng):
        self.tracking_seconds = 0.
        self.last_rate_reference = None
        self.tracking_sample_us = state['sim_us']
        self.gate_transition = None
        self.rng = rng
        self.origin = np.asarray(state['position_ned'], dtype=float).copy()
        self.waypoints_reached = 0
        q = np.asarray(state['q_ned_frd'], dtype=float)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        self.heading = wrap_yaw(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        self.altitude_target = -self.origin[2]
        self.previous_altitude = self.altitude_target
        self.previous_us = state['sim_us']
        self.velocity_up = -float(state['linear_velocity_ned'][2])
        self.initial_thrust_ratio = 1.
        self._next_waypoint(self.origin)

    def _next_waypoint(self, position):
        while True:
            point = self.origin + self.rng.uniform([-5.,-5.,-1.], [5.,5.,1.])
            if np.linalg.norm(point-position) >= 2.:
                break
        self._set_waypoint(point, position)

    def _set_waypoint(self, point, position):
        """Latch the mission heading once; only position feedback changes in flight."""
        self.position_target_ned = np.asarray(point, dtype=float).copy()
        self.altitude_target = -point[2]
        horizontal = (self.position_target_ned-np.asarray(position))[:2]
        if np.linalg.norm(horizontal) > 1e-6:
            self.heading = wrap_yaw(np.arctan2(horizontal[1], horizontal[0]))

    def _update_arrival_gate(self, state):
        position = np.asarray(state['position_ned'], dtype=float)
        elapsed = (state['sim_us'] - self.tracking_sample_us) / 1e6
        distance = float(np.linalg.norm(self.position_target_ned-position))
        actual = np.rad2deg(np.asarray(state['rates_true'],dtype=float))
        desired = (np.rad2deg(self.last_rate_reference) if self.last_rate_reference is not None
                   else np.full(3,np.nan))
        mean_error = float(np.mean(np.abs(actual - desired)))
        # Do not admit exact-boundary values rounded down by rad/deg conversion.
        eligible = bool(mean_error < self.tracking_tolerance_deg_s
                        and not np.isclose(mean_error, self.tracking_tolerance_deg_s,
                                           rtol=0., atol=1e-12))
        if elapsed > 0:
            self.tracking_seconds = self.tracking_seconds + elapsed if eligible else 0.
        elif elapsed < 0 or not eligible:
            self.tracking_seconds = 0.
        self.tracking_sample_us = state['sim_us']
        switched = bool(elapsed > 0 and eligible and distance < .2
                        and self.tracking_seconds + 1e-9 >= self.tracking_hold_seconds)
        self.gate_transition = dict(distance_m=distance,eligible=eligible,
            settled_seconds=self.tracking_seconds,switched=switched,waypoint_before=self.waypoints_reached,
            actual_deg_s=actual.tolist(), desired_deg_s=(desired.tolist() if self.last_rate_reference is not None else None))
        if switched:
            self.waypoints_reached += 1
            self._next_waypoint(position)
            self.tracking_seconds = 0.
        self.gate_transition['waypoint_after'] = self.waypoints_reached
        return switched

    def command(self, state, max_thrust_ratio, rate_limit):
        position = np.asarray(state['position_ned'], dtype=float)
        velocity = np.asarray(state['linear_velocity_ned'], dtype=float)
        self._update_arrival_gate(state)
        horizontal_error = (self.position_target_ned-position)[:2]
        # NED horizontal acceleration, with damping and a 3 m/s^2 norm limit.
        accel = horizontal_error - 1.8*velocity[:2]
        accel *= min(1., 3./max(np.linalg.norm(accel), 1e-9))
        c = self.config
        up = c.gravity_m_s2 + np.clip(c.altitude_p*(self.altitude_target+position[2])
              + c.altitude_d*velocity[2], -c.max_vertical_accel, c.max_vertical_accel)
        # Rotate acceleration into the desired heading's forward/right frame.
        co, si = np.cos(self.heading), np.sin(self.heading)
        forward, right = np.array([[co,si],[-si,co]]) @ accel
        roll = np.arctan2(right, np.sqrt(up*up+forward*forward))
        pitch = np.arctan2(-forward, up)
        self.euler_target = np.array([roll, pitch, self.heading])
        self.attitude_target = euler_quaternion(*self.euler_target)
        rates, thrust = super().command(state, max_thrust_ratio, rate_limit)
        self.last_rate_reference = np.array(rates, copy=True)
        return rates, thrust

    def diagnostics(self):
        return dict(super().diagnostics(), position_target_ned=self.position_target_ned.copy(),
                    waypoints_reached=self.waypoints_reached,
                    waypoint_tracking_seconds=self.tracking_seconds)
