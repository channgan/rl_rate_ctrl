"""Explicit checkpoint migration of the native task's episode duration only."""
from copy import deepcopy

from .env import TaskConfig
from .defaults import CURRENT
from .reward_contract import mse_reward_interface_metadata

DEFAULT_NATIVE_EPISODE_STEPS = CURRENT.episode_steps


def migrate_episode_horizon(model, new_contract):
    old_contract = getattr(model, 'promotion_environment_contract', None)
    if not isinstance(old_contract, dict) or not isinstance(old_contract.get('task'), dict):
        raise ValueError('Episode migration requires a saved task/environment contract')
    old_task = old_contract['task']
    new_task = new_contract['task']
    if not old_task.get('native_torque') or not new_task.get('native_torque'):
        raise ValueError('Episode migration supports only the native torque task')
    if getattr(model, 'curriculum_state', {}).get('pending_evaluation') is not None:
        raise ValueError('Cannot change the horizon during pending evaluation')
    candidate = deepcopy(old_contract)
    candidate['task']['episode_seconds'] = new_task['episode_seconds']
    if candidate != new_contract:
        raise ValueError('Episode migration permits only episode_seconds; task, dt and simulator must match')
    old_reward = mse_reward_interface_metadata(TaskConfig(**old_task))
    new_reward = mse_reward_interface_metadata(TaskConfig(**new_task))
    if getattr(model, 'reward_interface_metadata', None) != old_reward:
        raise ValueError('Saved reward does not match its original episode configuration')
    candidate_reward = deepcopy(old_reward)
    candidate_reward['raw_settlement']['early_failure_extra_max'] = new_reward['raw_settlement']['early_failure_extra_max']
    if candidate_reward != new_reward:
        raise ValueError('Episode migration cannot change reward coefficients or success thresholds')
    record = dict(previous_seconds=old_task['episode_seconds'], current_seconds=new_task['episode_seconds'],
                  previous_steps=round(old_task['episode_seconds']/old_task['dt']),
                  current_steps=round(new_task['episode_seconds']/new_task['dt']),
                  previous_settlement=deepcopy(old_reward['raw_settlement']),
                  current_settlement=deepcopy(new_reward['raw_settlement']),
                  optimizer_preserved=True)
    # Change contract metadata only after all checks pass; no network, optimizer,
    # KL-controller or worker RNG state is changed here.
    model.promotion_environment_contract = deepcopy(new_contract)
    model.reward_interface_metadata = deepcopy(new_reward)
    return record
