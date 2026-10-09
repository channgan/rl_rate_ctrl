"""Rate-only policy identity; intentionally incompatible with direct-motor PPO."""
import numpy as np
from gymnasium import spaces
import torch

ACTION_CONTRACT = 'px4_native_thrust_allocator_torque_actor9_v2'
CRITIC_CONTRACT = 'ordinary_actor9_critic9_64x64_v2'
REWARD_CONTRACT = 'rate_only_torque_slew_limits_no_headroom_no_waypoint_bonus_v4'


def interface_metadata():
    return dict(version=ACTION_CONTRACT, observation_size=9, action_size=3,
        dtype='float32', rate_frame='FRD', source_rate_units='rad/s', source_rate_divisor=5.,
        observation='gyro rate/5 [3], (issued target-gyro rate)/5 [3], previous accepted requested normalized torque [3]',
        history='last requested vehicle_torque_setpoint.xyz acknowledged by the native bridge, [-1,1]; zero before the first policy action; not allocator-achieved torque',
        action='vehicle_torque_setpoint.xyz normalized FRD torque [-1,1]; not N*m',
        thrust='PX4 vehicle_rates_setpoint.thrust_body forwarded unchanged to vehicle_thrust_setpoint; never actor controlled',
        allocation='PX4 control_allocator remains active; mc_rate_control stopped',
        actor_requires_privileged_inputs=False)


def require_training_contract(model):
    if (getattr(model, 'action_contract', None) != ACTION_CONTRACT
            or getattr(model, 'critic_contract', None) != CRITIC_CONTRACT
            or getattr(model, 'reward_contract', None) != REWARD_CONTRACT):
        raise ValueError('Rate-only training requires a matching 9D observation / 3D torque checkpoint; '
                         'old 10D / 3D allocated-ESC-history and 12D / 4D PWM models must not be resumed')
    if getattr(model, 'interface_metadata', None) != interface_metadata():
        raise ValueError('Rate-only checkpoint input/action metadata differs from the current interface; '
                         'matching dimensions do not establish matching units or semantics')
    obs, act = model.observation_space, model.action_space
    if not (isinstance(obs, spaces.Box) and obs.shape == (9,) and obs.dtype == np.float32
            and isinstance(act, spaces.Box) and act.shape == (3,) and act.dtype == np.float32
            and np.all(act.low == -1.) and np.all(act.high == 1.)):
        raise ValueError('Rate-only checkpoint spaces do not match the declared interface')
    expected_low = np.array([-np.inf] * 6 + [-1.] * 3, dtype=np.float32)
    expected_high = np.array([np.inf] * 6 + [1.] * 3, dtype=np.float32)
    if not (np.array_equal(obs.low, expected_low) and np.array_equal(obs.high, expected_high)):
        raise ValueError('Rate-only observations require unbounded rate/error fields and [-1,1] requested torque history')
    if model.policy.net_arch != dict(pi=[64,64], vf=[64,64]):
        raise ValueError('Rate-only actor and critic must each use 64x64 networks')
    extractor = getattr(model.policy, 'mlp_extractor', None)
    for branch in ('policy_net', 'value_net'):
        layers = getattr(extractor, branch, None)
        if (getattr(model.policy, 'activation_fn', None) is not torch.nn.ReLU
                or not isinstance(layers, torch.nn.Sequential) or len(layers) != 4
                or not isinstance(layers[0], torch.nn.Linear)
                or not isinstance(layers[2], torch.nn.Linear)
                or not isinstance(layers[1], torch.nn.ReLU)
                or not isinstance(layers[3], torch.nn.ReLU)
                or (layers[0].in_features, layers[0].out_features,
                    layers[2].in_features, layers[2].out_features) != (9, 64, 64, 64)):
            raise ValueError('Rate-only actor and critic must implement 9 -> 64 -> 64 with ReLU')
