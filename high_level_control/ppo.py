import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from high_level_control.actor_critic import ActorCritic
from high_level_control.rollout_storage import RolloutStorage

class PPO_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.

    selective_adaptation_module_loss = False


class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, one_step_models=None, device='cpu'):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPO_Args.learning_rate)

        self.one_step_models = []
        if one_step_models is not None:
            self.one_step_models = one_step_models
            self.one_step_optimizers = [optim.Adam(osm.parameters(), lr=1e-3) for osm in one_step_models]
        
        self.transition = RolloutStorage.Transition()

        self.learning_rate = PPO_Args.learning_rate

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape, low_level_obs_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, low_level_obs_shape, self.device)

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history):
        # Compute the actions and values
        actions, value =  self.actor_critic.act_evaluate(obs_history, privileged_obs)
        self.transition.actions = actions.detach() # self.actor_critic.act(obs_history).detach()
        self.transition.values = value.detach() # self.actor_critic.evaluate(obs_history, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        return self.transition.actions
    
    
    def process_env_step(self, next_observations, rewards, dones, infos):
        self.transition.next_observations = next_observations.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # self.transition.env_bins = infos["env_bins"]
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += PPO_Args.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        # self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        for obs_batch, next_obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, low_level_obs_batch, low_level_next_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            _, value_batch = self.actor_critic.act_evaluate(obs_history_batch, privileged_obs_batch, masks=masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # value_batch = self.actor_critic.evaluate(obs_history_batch, privileged_obs_batch, masks=masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if PPO_Args.desired_kl != None and PPO_Args.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > PPO_Args.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPO_Args.clip_param,
                                                                               1.0 + PPO_Args.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + PPO_Args.value_loss_coef * value_loss - PPO_Args.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO_Args.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            data_size = privileged_obs_batch.shape[0]
            num_train = int(data_size // 5 * 4)


        # mean_osm_loss = 0.
        # osm_loss_count = 0
        # if len(self.one_step_models) > 0:
        #     for i in range(1):
        #         hl_loss_fn = nn.MSELoss(reduction='mean')
        #         # (512 * 25) / 100 = 128 
        #         generator = self.storage.mini_batch_generator(100, PPO_Args.num_learning_epochs)
        #         for obs_batch, next_obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, low_level_obs_batch, low_level_next_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
        #             old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

        #             osm_idx = np.random.choice(np.arange(0, len(self.one_step_models)))
        #             osm_model = self.one_step_models[osm_idx]
        #             osm_optimizer = self.one_step_optimizers[osm_idx]
        #             # latent = None
        #             # with torch.no_grad():
        #             #     # obs_batch[:, :6], 
        #             #     latent = self.actor_critic.get_latent(obs_history_batch, privileged_obs_batch)
        #             #     latent_next = self.actor_critic.get_latent(torch.cat([obs_history_batch[:, obs_batch.size(1):], next_obs_batch], dim=-1), privileged_obs_batch)

        #             osm_model.train()
        #             osm_optimizer.zero_grad()
                    
        #             decoded_obs, decoded_next_obs, decoded_pred_next_obs, _ = osm_model(torch.cat([obs_batch[:, :6], obs_batch[:, 9:]], dim=-1), actions_batch, torch.cat([next_obs_batch[:, :6], next_obs_batch[:, 9:]], dim=-1))

        #             loss = hl_loss_fn(decoded_obs, torch.cat([obs_batch[:, :6], obs_batch[:, 9:]], dim=-1)) + hl_loss_fn(decoded_next_obs, torch.cat([next_obs_batch[:, :6], next_obs_batch[:, 9:]], dim=-1)) + hl_loss_fn(decoded_pred_next_obs, torch.cat([next_obs_batch[:, :6], next_obs_batch[:, 9:]], dim=-1))

        #             # out = osm_model(torch.cat([obs_batch[:, :6], obs_batch[:, 9:]], dim=-1), actions_batch)
        #             # loss = hl_loss_fn(out, torch.cat([next_obs_batch[:, :6], next_obs_batch[:, 9:]], dim=-1))
        #             loss.backward()
        #             osm_optimizer.step()
        #             mean_osm_loss += loss.item()
        #             osm_loss_count += 1

        # try:
        #     mean_osm_loss /= osm_loss_count
        # except ZeroDivisionError:
        #     mean_osm_loss = 0.
            
        num_updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        self.storage.clear()


        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student
