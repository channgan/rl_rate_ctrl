"""Simulation reference generator: altitude PD and quaternion attitude P.

It generates rate/thrust requests only. The learned policy still exclusively
commands all four motors. This is not the user's MPC or an XY position hold.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OuterLoopConfig:
    attitude_hold_seconds: float = 2.0
    roll_pitch_limit_deg: float = 45.0  # Combined tilt is also limited to this cone.
    yaw_limit_deg: float = 45.0
    initial_thrust_variation: float = 0.1
    fixed_altitude_target_m: float | None = None
    attitude_p: float = 4.0
    altitude_p: float = 2.0
    altitude_d: float = 2.0
    max_vertical_accel: float = 3.0
    gravity_m_s2: float = 9.8


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def wrap_yaw(yaw):
    return float((yaw + np.pi) % (2 * np.pi) - np.pi)


def canonical_rotation(q):
    q = np.asarray(q, dtype=float).copy()
    q /= np.linalg.norm(q)
    # Deterministic sign at the exactly-180-degree shortest-path tie.
    pivot = next((x for x in q if abs(x) > 1e-12), 1.)
    return -q if pivot < 0 else q


def euler_quaternion(roll, pitch, yaw):
    yaw = wrap_yaw(yaw)
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


class AltitudeAttitudeReference:
    def __init__(self, config=None):
        self.config = config or OuterLoopConfig()

    def reset(self, state, rng):
        self.altitude_target = -float(state["position_ned"][2])
        self.previous_altitude = self.altitude_target
        self.initial_thrust_ratio = float(rng.uniform(
            1 - self.config.initial_thrust_variation, 1 + self.config.initial_thrust_variation))
        # At a level, stationary reset this yields the sampled F/(m*g).
        # Thereafter the altitude loop remains responsible for the demand.
        self.altitude_target += self.config.gravity_m_s2 * (self.initial_thrust_ratio - 1) / self.config.altitude_p
        if self.config.fixed_altitude_target_m is not None:
            self.altitude_target = self.config.fixed_altitude_target_m
            self.initial_thrust_ratio = 1.0
        self.previous_us = state["sim_us"]
        self.velocity_up = -float(state["linear_velocity_ned"][2])
        self.sample(rng)

    def sample(self, rng):
        c = self.config
        while True:
            self.euler_target = np.deg2rad(rng.uniform(
                [-c.roll_pitch_limit_deg, -c.roll_pitch_limit_deg, -c.yaw_limit_deg],
                [c.roll_pitch_limit_deg, c.roll_pitch_limit_deg, c.yaw_limit_deg]))
            if np.cos(self.euler_target[0]) * np.cos(self.euler_target[1]) >= np.cos(np.deg2rad(c.roll_pitch_limit_deg)):
                break
        self.attitude_target = euler_quaternion(*self.euler_target)

    def command(self, state, max_thrust_ratio, rate_limit):
        """Return body-rate targets and total thrust / vehicle weight."""
        c = self.config
        q = np.asarray(state["q_ned_frd"], dtype=float)
        q = q / np.linalg.norm(q)
        error = canonical_rotation(multiply(q * [1, -1, -1, -1], self.attitude_target))
        rates = np.clip(2 * c.attitude_p * error[1:], -np.asarray(rate_limit), np.asarray(rate_limit))
        altitude = -float(state["position_ned"][2])
        dt = (state["sim_us"] - self.previous_us) * 1e-6
        if dt > 0:
            measured_velocity = -float(state["linear_velocity_ned"][2])
            alpha = 1 - np.exp(-dt / .02)
            self.velocity_up += alpha * (measured_velocity - self.velocity_up)
        self.previous_altitude, self.previous_us = altitude, state["sim_us"]
        accel = np.clip(c.altitude_p * (self.altitude_target - altitude)
                        - c.altitude_d * self.velocity_up,
                        -c.max_vertical_accel, c.max_vertical_accel)
        cos_tilt = 1 - 2 * (q[1]**2 + q[2]**2)
        requested = (1 + accel / c.gravity_m_s2) / max(float(cos_tilt), .1)
        self.unclipped_thrust = float(requested)
        return rates, float(np.clip(requested, 0, max_thrust_ratio))

    def diagnostics(self):
        return dict(altitude_target_m=self.altitude_target,
                    initial_thrust_ratio=self.initial_thrust_ratio,
                    outer_unclipped_thrust=getattr(self, "unclipped_thrust", None),
                    attitude_target_rpy_rad=self.euler_target.copy())
