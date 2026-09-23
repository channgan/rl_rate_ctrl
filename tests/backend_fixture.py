import numpy as np

class TestBackend:
    """Deterministic test double; production training never uses this class."""
    __test__ = False
    hover = .7
    config = dict(mode="ball_rig", mass_kg=1., gravity_m_s2=9.8,
                  max_rotor_rad_s=1000., thrust_coefficient=5e-6)

    def reset(self, seed=None):
        self.t = 1_000_000
        self.rates = [0., 0., 0.]
        self.true_rates = None
        self.rotors = np.full(4, self.hover)
        self.applied_speed_fraction = np.full(4, self.hover)
        return self.state()

    def state(self):
        return dict(sim_us=self.t, sample_us=self.t, rates=self.rates,
                    rates_true=self.rates if self.true_rates is None else self.true_rates,
                    position_ned=[0., 0., -3.], q_ned_frd=[1., 0., 0., 0.],
                    linear_velocity_ned=[0., 0., 0.], rotor_speed_fraction=self.rotors.copy(),
                    applied_pwm=self.applied_speed_fraction.copy())

    def step(self, action, dt):
        self.applied_speed_fraction = np.asarray(action).copy()
        self.t += round(dt * 1e6)
        return self.state()

    def close(self):
        pass
