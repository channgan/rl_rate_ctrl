"""PPO with physically separate actor-only and privileged-value input paths."""
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ActorObservationExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=12)

    def forward(self, observations):
        return observations["actor"].float()


class CriticObservationExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=35)

    def forward(self, observations):
        return observations["critic"].float()


class AsymmetricMlpExtractor(nn.Module):
    """Retain SB3's actor/value names while allowing different input widths."""
    def __init__(self, net_arch, activation_fn):
        super().__init__()
        if not isinstance(net_arch, dict) or set(net_arch) != {"pi", "vf"}:
            raise ValueError("Asymmetric policy requires separate pi/vf layer lists")

        def branch(input_size, widths):
            modules = []
            for width in widths:
                if not isinstance(width, int) or width < 1:
                    raise ValueError("Network widths must be positive integers")
                modules.extend([nn.Linear(input_size, width), activation_fn()])
                input_size = width
            return nn.Sequential(*modules), input_size

        self.policy_net, self.latent_dim_pi = branch(12, net_arch["pi"])
        self.value_net, self.latent_dim_vf = branch(35, net_arch["vf"])

    def forward_actor(self, actor_features):
        return self.policy_net(actor_features)

    def forward_critic(self, critic_features):
        return self.value_net(critic_features)

    def forward(self, features):
        actor_features, critic_features = features
        return self.forward_actor(actor_features), self.forward_critic(critic_features)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """Actor sees 12 dimensions; critic sees its own 35, with no shared weights.

    SB3's forward/evaluate_actions/get_distribution/predict_values remain in
    charge of distributions and PPO semantics. Only key selection and the two
    input widths differ. Privileged data never enters the actor feature path.
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        if not isinstance(observation_space, spaces.Dict) or set(observation_space.spaces) != {"actor", "critic"}:
            raise ValueError("Expected Dict(actor=Box12, critic=Box35)")
        for name, size in (("actor", 12), ("critic", 35)):
            space = observation_space[name]
            if not isinstance(space, spaces.Box) or space.shape != (size,) or space.dtype != np.float32:
                raise ValueError(f"Expected float32 {name} observations of size {size}")
        if not isinstance(action_space, spaces.Box) or action_space.shape != (4,) or not (
                np.all(action_space.low == 0) and np.all(action_space.high == 1)):
            raise ValueError("Expected four normalised PWM actions in [0,1]")
        if kwargs.pop("share_features_extractor", False):
            raise ValueError("Actor and critic feature paths must remain separate")
        extractor_class = kwargs.pop("features_extractor_class", ActorObservationExtractor)
        if extractor_class is not ActorObservationExtractor or kwargs.get("features_extractor_kwargs"):
            raise ValueError("Asymmetric policy uses fixed, independent key selectors")
        if kwargs.get("use_sde", False) or kwargs.get("squash_output", False):
            raise ValueError("This interface uses an unsquashed diagonal Gaussian policy")
        kwargs.setdefault("net_arch", dict(pi=[64, 64], vf=[128, 128]))
        kwargs.setdefault("activation_fn", nn.ReLU)
        super().__init__(observation_space, action_space, lr_schedule,
                         share_features_extractor=False,
                         features_extractor_class=ActorObservationExtractor, **kwargs)

    def _build_mlp_extractor(self):
        # The parent constructs two feature extractors before building its
        # optimizer. Replace the value selector here, before any optimizer exists.
        self.vf_features_extractor = CriticObservationExtractor(self.observation_space)
        self.mlp_extractor = AsymmetricMlpExtractor(self.net_arch, self.activation_fn).to(self.device)

    def predict_actor(self, observation, deterministic=True):
        """Deployable inference from actor12 alone, following SB3's (action, state) API."""
        array = np.asarray(observation, dtype=np.float32)
        if array.ndim not in (1, 2) or array.shape[-1] != 12 or not np.all(np.isfinite(array)):
            raise ValueError("Expected finite float32 actor observation (12,) or (batch,12)")
        unbatched = array.ndim == 1
        self.set_training_mode(False)
        with torch.no_grad():
            tensor = torch.as_tensor(array.reshape(-1, 12), device=self.device)
            latent = self.mlp_extractor.forward_actor(tensor)
            distribution = self._get_action_dist_from_latent(latent)
            actions = distribution.get_actions(deterministic=deterministic).cpu().numpy()
        if not np.all(np.isfinite(actions)):
            raise ValueError("Actor produced non-finite actions")
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return (actions[0] if unbatched else actions), None
