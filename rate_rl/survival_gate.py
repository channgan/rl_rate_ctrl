"""A full training episode ends exploration; this is not reward success."""
from copy import deepcopy


EXPLORATION_MAX_TOTAL_STEPS = 300000


def exploration_step_budget(current_steps):
    remaining = EXPLORATION_MAX_TOTAL_STEPS - current_steps
    if remaining <= 0:
        raise ValueError('Exploration checkpoint already reached the total step cap; start late phase')
    return remaining


class FullEpisodeGate:
    def __init__(self, horizon_steps, previous=None, required_consecutive=10):
        if type(horizon_steps) is not int or horizon_steps <= 0:
            raise ValueError('A positive episode horizon is required')
        self.horizon_steps = horizon_steps
        if type(required_consecutive) is not int or required_consecutive <= 0:
            raise ValueError('A positive consecutive episode count is required')
        self.required_consecutive = required_consecutive
        self.consecutive = 0
        self.first_episode = None
        if previous is not None:
            if previous.get('horizon_steps') != horizon_steps:
                raise ValueError('Saved survival gate horizon differs')
            # A one-episode certificate cannot satisfy the new ten-episode rule.
            if previous.get('required_consecutive', 1) == required_consecutive:
                self.first_episode = deepcopy(previous.get('first_episode'))
                self.consecutive = int(previous.get('consecutive', int(self.first_episode is not None)))

    @property
    def passed(self):
        return self.consecutive >= self.required_consecutive

    def observe(self, info, done, training_step):
        if self.passed or not done:
            return
        episode = info.get('episode')
        if (episode and episode['l'] >= self.horizon_steps
                and info.get('failure') in (None, 'tracking_rmse')
                and (info.get('is_success') or info.get('failure') == 'tracking_rmse')):
            self.consecutive += 1
            if self.passed:
                self.first_episode = dict(training_step=int(training_step),
                                      episode_steps=int(episode['l']),
                                      outcome=info.get('failure') or 'success')
        else:
            self.consecutive = 0
            self.first_episode = None

    def state(self):
        return dict(horizon_steps=self.horizon_steps, passed=self.passed,
                    required_consecutive=self.required_consecutive, consecutive=self.consecutive,
                    first_episode=deepcopy(self.first_episode))
