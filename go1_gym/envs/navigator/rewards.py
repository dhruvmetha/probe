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

    def _reward_torque_energy(self):
        return torch.sum(torch.square(self.env.legged_env.torques), dim=-1)

    def _reward_action_energy(self):
        return torch.sum(torch.square((torch.abs(self.env.actions) > 0.3) * self.env.actions), dim=-1)

        # return torch.sum(torch.square(self.env.actions), dim=-1)
        # return torch.sum(torch.square(self.env.actions), dim=-1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_zero_velocity(self):
        return ((torch.abs(self.env.legged_env.base_lin_vel[:, 0]) < 0.2) | (torch.abs(self.env.legged_env.base_lin_vel[:, 1]) < 0.2)) * 1.0

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.legged_env.contact_forces[:, self.env.legged_env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    # def _reward_side_limits(self):
    #     cross = torch.zeros(self.robot_bounding_box[0].shape[0], device=self.robot_bounding_box[0].device)
    #     # print(self.robot_bounding_box[0][0, 1], self.robot_bounding_box[1][0, 1], self.robot_bounding_box[2][0, 1], self.robot_bounding_box[3][0, 1])
    #     for i in range(4):
    #         cross[:] += (torch.abs(self.robot_bounding_box[i][:, 1]) > 0.95).long()
    #     # print(cross[0])
    #     return (cross > 0) * 1.0

    # def _reward_back_limits(self):
    #     cross = torch.zeros(self.robot_bounding_box[0].shape[0], device=self.robot_bounding_box[0].device)
    #     for i in range(4):
    #         cross[:] += (self.robot_bounding_box[i][:, 0] < -0.35).long()
    #     return (cross > 0) * 1.0
    

    