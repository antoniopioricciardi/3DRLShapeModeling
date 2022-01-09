import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from perceiver_pytorch import PerceiverIO
import torch 
import numpy as np 
import poly
# class CustomDense(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
#         super(CustomDense, self).__init__(observation_space, features_dim)
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         n_input_channels = observation_space.shape[0]
#         self.Net = nn.Sequential(
#             nn.Linear(n_input_channels, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.Net(
#                 th.as_tensor(observation_space.sample()[None]).float()
#             ).shape[1]

#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.linear(self.Net(observations))

def get_perc(features_dim=128):
    return dict(
    features_extractor_class=poly.CustomPerc,
    features_extractor_kwargs=dict(features_dim=128)
    )

def get_mlp(pi, vf):
    pi = [x for x in pi]
    vf = [x for x in vf]
    return dict(activation_fn=th.nn.ReLU,
                    net_arch=[dict(pi=pi, vf=vf)])
    # return dict(activation_fn=th.nn.ReLU,
    #                  net_arch=[dict(pi=[32, 32], vf=[32, 32])])


class CustomPerc(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, num_points: int = 4):
        super(CustomPerc, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            # Run through a simple MLP
            extractors[key] = nn.Linear(subspace.shape[0], subspace.shape[0])
            total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)
        #self._features_dim = total_concat_size
        self.n_input_channels = total_concat_size
        self.n_flatten = total_concat_size

        self.Net = PerceiverIO(
            dim = self.n_input_channels,                    # dimension of sequence to be encoded
            queries_dim = self.n_input_channels,            # dimension of decoder queries
            logits_dim = self.n_input_channels,             # dimension of final logits
            depth = 2,                   # depth of net
            num_latents = 64,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 128,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 4,            # number of heads for latent self attention, 8
            cross_dim_head = 32,         # number of dimensions per cross attention head
            latent_dim_head = 32,        # number of dimensions per latent self attention head
            weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
            self_per_cross_attn = 2,     # number of self attention blocks per cross attention  
        )

        self.linear = nn.Sequential(nn.Linear(total_concat_size, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        out = th.cat(encoded_tensor_list, dim=1)

        feats = self.Net(th.reshape(out,(out.shape[0],-1,self.n_input_channels)), 
                         queries = th.reshape(out,(out.shape[0],-1,self.n_input_channels)))
        return self.linear(th.reshape(feats,(out.shape[0],-1)))


# policy_kwargs = dict(
#     features_extractor_class=CustomPerc,
#     features_extractor_kwargs=dict(features_dim=128),
# )
# model = PPO("MultiInputPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)

# model.learn(1000)