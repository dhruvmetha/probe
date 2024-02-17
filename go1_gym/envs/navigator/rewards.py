import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi



class CoRLRewards:
    def __init__(self, env):
        self.env = env
    
    # def __getattr__(self, name):
    #     return getattr(self.env, name)
    
    # ------------ reward functions----------------
    
    def _reward_terminal_distance_gs(self):
        # print(self.gs_buf * 1.0)
        return self.env.gs_buf * 1.0

    def _reward_terminal_time_out(self):
        return (self.env.time_out_buf & ~self.env.gs_buf) * 1.0

    def _reward_terminal_distance_travelled(self):
        return ((1/torch.exp(self.env.distance_travelled)) * (self.env.gs_buf * 1.0))/0.0018

    def _reward_torque_energy(self):
        return torch.sum(torch.square(self.env.legged_env.torques), dim=-1)

    def _reward_action_energy(self):
        return torch.sum(torch.square((torch.abs(self.env.actions) > 0.3) * self.env.actions), dim=-1)
    
    def _reward_terminal_time_taken(self):
        return (1 - (self.env.episode_length_buf/1500)) * (self.env.gs_buf * 1.0)
        
    def _reward_terminal_num_collisions(self):
        return (1 - (self.env.legged_env.num_collisions/(3000))) * (self.env.gs_buf * 1.0)

    def _reward_time(self):
        return torch.ones(self.env.num_envs, device=self.env.device)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_zero_velocity(self):
        return ((torch.abs(self.env.legged_env.base_lin_vel[:, 0]) < 0.2) | (torch.abs(self.env.legged_env.base_lin_vel[:, 1]) < 0.2)) * 1.0

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.legged_env.contact_forces[:, self.env.legged_env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_side_limits(self):
        cross = torch.zeros(self.env.robot_bounding_box[0].shape[0], device=self.env.robot_bounding_box[0].device)
        for i in range(4):
            cross[:] += ((torch.abs(self.env.robot_bounding_box[i][:, 1]) > 0.95) * 1.0)
        return (cross > 0) * 1.0

    def _reward_back_limits(self):
        cross = torch.zeros(self.env.robot_bounding_box[0].shape[0], device=self.env.robot_bounding_box[0].device)
        for i in range(4):
            cross[:] += ((self.env.robot_bounding_box[i][:, 0] < -0.35) * 1.0)
        return (cross > 0) * 1.0
    
    def _reward_torque_energy(self):
        return torch.square(self.env.legged_env.torques).sum(dim=-1)

    def _reward_distance(self):
        # abs(2.0 - 3.5)
        # return torch.abs(self.env.goal_positions - self.env.base_pos[:, 0])
        r = torch.zeros_like(self.env.base_pos[:, 0])
        r = (self.env.goal_positions - self.env.base_pos[:, 0])
        return r

        r = torch.zeros_like(self.env.base_pos[:, 0])
        r[self.env.base_pos[:, 0] < 0.0] = self.env.base_pos[self.env.base_pos[:, 0] < 0.0, 0]
        r[self.env.base_pos[:, 0] >= 0.0] = torch.exp(self.env.base_pos[self.env.base_pos[:, 0] >= 0.0, 0])/torch.exp(self.env.goal_positions[self.env.base_pos[:, 0] >= 0.0])
        # print(self.env.base_pos[0, 0], r[0])
        return r # self.env.base_pos[:, 0]

    def _reward_norm_distance(self):
        # print(self.env.goal_positions)
        normalized_location = torch.clamp((((self.env.goal_positions - self.env.base_pos[:, 0])/4.0)), min=0., max=1.)

        # print(self.env.base_pos[0, 0], (1 - (normalized_location ** 0.4))[0], normalized_location[0])
        # print(normalized_location[0], (1 - (normalized_location ** 0.4))[0])
        # print(1/torch.exp(torch.abs(self.env.base_pos[:, 0] - self.env.goal_positions))[0], 1 - (normalized_location ** 0.4)[0])
        return 1 - (normalized_location ** 0.4)

    