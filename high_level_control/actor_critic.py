import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256, 128]

    use_decoder = False


class PrivilegedFeatures(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.fc(x)

# class Actor(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size

#         self.fc = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.ELU(),
#             nn.Linear(512, 256),
#             nn.ELU(),
#             nn.Linear(256, 128),
#             nn.ELU(),
#             nn.Linear(128, output_size),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.fc(x)

# class Critic(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size

#         self.fc = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.ELU(),
#             nn.Linear(512, 256),
#             nn.ELU(),
#             nn.Linear(256, 128),
#             nn.ELU(),
#             nn.Linear(128, output_size),
#         )
    
#     def forward(self, x):
#         return self.fc(x)
    
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ELU(),
            nn.Linear(4096, 2048),
            nn.ELU(),
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.Linear(512, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ELU(),
            nn.Linear(4096, 2048),
            nn.ELU(),
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.Linear(512, output_size),
        )
    
    def forward(self, x):
        return self.fc(x)



class SharedLayers(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, output_size),
        )
    
    def forward(self, x):
        return self.fc(x)


class ActorCritic(nn.Module):
    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions):

        super().__init__()
        self.input_size = num_obs_history # num_obs #  + num_privileged_obs
        self.output_size = num_actions

        self.shared_memory =  SharedLayers(self.input_size, 256)
        self.privileged_features = PrivilegedFeatures(21, 8)
        self.actor = Actor(self.input_size , self.output_size)
        self.critic = Critic(self.input_size, 1)

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def act_evaluate(self, observations, privileged_obs, **kwargs):
        # state = self.shared_memory(torch.cat([observations], dim=-1))
        # state = observations[:, :3]
        # state = torch.cat([state, self.privileged_features(observations[:, 3:])], dim=-1)
        state = observations
        self.update_distribution(state)
        value = self.critic(state)
        return self.distribution.sample(), value

    def update_distribution(self, state):
        mean = self.actor(state)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, privileged_obs, **kwargs):
        # state = self.shared_memory(torch.cat([observations], dim=-1))
        # state = observations[:, :3]
        # state = torch.cat([state, self.privileged_features(observations[:, 3:])], dim=-1)
        state = observations
        self.update_distribution(state)
        return self.distribution.sample()
        
    def evaluate(self, observations, privileged_obs, **kwargs):
        # state = self.shared_memory(torch.cat([observations], dim=-1))
        # state = observations[:, :3]
        # state = torch.cat([state, self.privileged_features(observations[:, 3:])], dim=-1)
        state = observations
        return self.critic(state)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, privileged_obs, **kwargs):
        # state = self.shared_memory(torch.cat([observations], dim=-1))
        # state = observations[:, :3]
        # state = torch.cat([state, self.privileged_features(observations[:, 3:])], dim=-1)
        state = observations
        return self.actor(state)

    def get_latent(self, observations, privileged_obs, **kwargs):
        return self.shared_memory(torch.cat([observations], dim=-1))
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # def act_expert(self, ob, policy_info={}):
    #     return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    # def act_inference(self, ob, policy_info={}):
    #     return self.act_student(ob["obs_history"], policy_info=policy_info)

    # def act_student(self, observation_history, policy_info={}):
    #     latent = self.adaptation_module(observation_history)
    #     actions_mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
    #     policy_info["latents"] = latent.detach().cpu().numpy()
    #     return actions_mean

    # def act_teacher(self, observation_history, privileged_info, policy_info={}):
    #     actions_mean = self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))
    #     policy_info["latents"] = privileged_info
    #     return actions_mean

    

