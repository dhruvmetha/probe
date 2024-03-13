from isaacgym import gymutil, gymapi
from isaacgym.torch_utils import *
import torch
import numpy as np
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
import time


from go1_gym.envs.base.base_task import BaseTask
from go1_gym.envs.navigator.navigator_config import Cfg

from go1_gym.envs.base.legged_robot import LeggedRobot
from go1_gym.envs.base.legged_robot_config import Cfg as LeggedCfg

from go1_gym.envs.world.new_world import WorldAsset


from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.utils.math_utils import quat_apply_yaw


# from scene_predictor.model import MiniTransformer
from scene_predictor.inference import ObstacleInference
from scene_predictor.inference import PoseInference
# from scene_predictor.velocity_estimator import VelocityEstimator
from scene_predictor.visualization import get_visualization

class Navigator(BaseTask):
    def __init__(self, cfg: Cfg, sim_device, headless, num_envs=None, eval_cfg:Cfg=None, physics_engine="SIM_PHYSX", initial_dynamics_dict=None, save_data=False, random_pose=False, use_localization_model=False, use_obstacle_model=True, inference_device='cuda:0', **kwargs):
        self.use_obstacle_model = use_obstacle_model
        self.use_velocity_estimator = False  # use_localization_model
        self.use_localization_model = use_localization_model
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.save_data = save_data
        self.random_pose = random_pose

        if num_envs is not None:
            cfg.env.num_envs = num_envs
        self.num_envs = cfg.env.num_envs

        self.train_test_split = cfg.env.train_test_split

        self.num_train_envs = max(1, int(self.num_envs * self.train_test_split)) 
        self.num_eval_envs = self.num_envs - self.num_train_envs

        if self.save_data:
            num_workers = 12
            self.q_s = [mp.JoinableQueue(maxsize=500) for _ in range(num_workers)]
            self.workers = [mp.Process(target=self.worker, args=(q, idx)) for idx, q in enumerate(self.q_s)]
            for worker in self.workers:
                worker.daemon = True
                worker.start()

        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        self.sim_params = sim_params
        self.initial_dynamics_dict = initial_dynamics_dict
        if eval_cfg is not None: self._parse_cfg(eval_cfg)
        self._parse_cfg(self.cfg)

        # base task has the create sim call
        self.envs = []

        self.legged_env = None
        self.legged_env_obs = None

        self.world_env = None
        self.world_env_obs = None

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg)

        self.inference_model_device = inference_device
        # if self.use_velocity_estimator:
        #     VEL_FOLDER = Path(f'./scene_predictor/results/velocity/old_model')
        #     localization_model_path=str(VEL_FOLDER/'checkpoints/model_164.pt')
        #     self.pe = VelocityEstimator(input_size=39, num_heads=1, num_layers=1, hidden_size=2048, embed_size=512)
        #     self.pe.load_model('/home/dhruv/projects_dhruv/gaited_walk_deploy/velocity_predictor/model_164.pt', device=self.inference_model_device)
        #     for i in range(5):
        #         with torch.inference_mode():
        #             _ = self.pe.predict(torch.zeros(1, 1500, 39, device=self.inference_model_device, dtype=torch.float32))

        # if self.use_localization_model:
        #     POSE_FOLDER = Path(f'./scene_predictor/results/transformer_750_2048/2023-05-20_00-36-45')
        #     localization_model_path=str(POSE_FOLDER/'checkpoints/model_96.pt')
        #     self.localization_inference = PoseInference(model_path=localization_model_path, sequence_length=750, device=self.inference_model_device)
        
        # if self.use_obstacle_model:
        #     # PRIV_INFO_FOLDER = Path(f'./scene_predictor/results/transformer_750_2048/2023-05-20_16-57-15')
        #     PRIV_INFO_FOLDER = Path(f'./scene_predictor/results_priv_info/transformer_750_2048/2023-05-23_02-09-44')
        #     obstacle_model_path=str(PRIV_INFO_FOLDER/'checkpoints/model_91.pt')
        #     self.obstacle_inference = ObstacleInference(model_path=obstacle_model_path, sequence_length=750, device=self.inference_model_device)

        self.__init_buffers()

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self.record_now = False
        self.record_eval_now = False

        self.world_env.reset()

        self._prepare_reward_function()


    def _pre_create_env(self):

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        self._get_env_origins(torch.arange(self.num_envs, device=self.device))

        # self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))

        self.legged_env = HistoryWrapper(VelocityTrackingEasyEnv(self.gym, self.sim, self.num_envs, LeggedCfg, self.sim_params, self.env_origins, random_pose=self.random_pose, device=self.device))
        self.legged_env.pre_create_actor()

        self.world_env = WorldAsset(self.gym, self.sim, self.num_envs, self.env_origins, device=self.device, train_ratio=self.train_test_split)
        self.world_env.pre_create_actor()
    
    def _create_envs(self):
        self._pre_create_env()
        chosen_asset = np.random.randint(0, 3, size=self.num_envs)
        for i in range(self.num_envs):
            env_lower = gymapi.Vec3(0., 0., 0.)
            env_upper = gymapi.Vec3(0., 0., 0.)
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i]

            self.legged_env.create_actor(i, env_handle, pos)
            self.world_env.create_actor(i, chosen_asset[i], env_handle, pos)

            self.envs.append(env_handle)

        self._post_create_env()
    
    def _post_create_env(self):
        self.legged_env.post_create_actor(self.envs)
        self.world_env.post_create_actor(self.envs)
        # self.world_env.reset()
        
    def __init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.legged_env.init_buffers(actor_root_state, dof_state_tensor, net_contact_forces, rigid_body_state)
        self.world_env.init_buffers(actor_root_state, dof_state_tensor, net_contact_forces, rigid_body_state)

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float32)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float32)

        self.vel_window_size = 2
        self.vel_window_size = max(2, self.vel_window_size)
        self.last_five_pos = torch.zeros(self.num_envs, self.vel_window_size, 2, device=self.device, dtype=torch.float32)
        self.last_five_timesteps = torch.zeros(self.num_envs, 5, device=self.device, dtype=torch.int32)

        body_height_cmd = 0.0
        step_frequency_cmd = 3.0
        gait = torch.tensor([0.5, 0, 0.0])
        footswing_height_cmd = 0.08
        pitch_cmd = 0.0
        roll_cmd = 0.0
        stance_width_cmd = 0.25

        self.commands = torch.zeros(self.num_envs, 10, device=self.device, dtype=torch.float32) + torch.tensor([body_height_cmd, step_frequency_cmd, *gait, 0.5, footswing_height_cmd, pitch_cmd, roll_cmd, stance_width_cmd], device=self.device)
        # gaits = {"pronking": [0, 0, 0],
        #         "trotting": [0.5, 0, 0],
        #         "bounding": [0, 0.5, 0],
        #         "pacing": [0, 0, 0.5]}
        
        # body_height_cmd = 0.0
        # step_frequency_cmd = 3.0
        # gait = torch.tensor(gaits["trotting"])
        # footswing_height_cmd = 0.1
        # pitch_cmd = 0.0
        # roll_cmd = 0.0
        # stance_width_cmd = 0.3

        # self.commands = torch.zeros(self.num_envs, 10, device=self.device, dtype=torch.float32) + torch.tensor([body_height_cmd, step_frequency_cmd, *gait, 0.5, footswing_height_cmd, pitch_cmd, roll_cmd, stance_width_cmd], device=self.device)

        self.gs_buf = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.bool)

        self.base_pos = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False, dtype=torch.float32)
        self.base_yaw = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False, dtype=torch.float32)

        self.dones = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.bool)

        self.success_envs = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.int)
        self.count_envs = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.int)

        self.full_seen_world_obs = torch.zeros(self.num_envs, 21, device=self.device, requires_grad=False, dtype=torch.float32)
        self.distance_travelled = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.float32)
        self.last_base_pos = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False, dtype=torch.float32)
        self.last_base_yaw = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False, dtype=torch.float32)

        # self.num_collisions = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.int)

        self.goal_positions = self.legged_env.goal_positions

        self.env_ids = torch.arange(self.num_envs, device=self.inference_model_device if self.use_localization_model or self.use_obstacle_model else self.device, requires_grad=False, dtype=torch.long).view(-1, 1)
        self.env_step = torch.zeros(self.num_envs, 1, device=self.inference_model_device if self.use_localization_model or self.use_obstacle_model else self.device, requires_grad=False, dtype=torch.long)

        if self.use_localization_model:
            self.pose_inp = torch.zeros(self.num_envs, 750, 27, device=self.inference_model_device, requires_grad=False, dtype=torch.float32)
        if self.use_obstacle_model:
            self.priv_info_inp = torch.zeros(self.num_envs, 750, 12+3+24+6, device=self.inference_model_device, requires_grad=False, dtype=torch.float32)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        # cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def create_sim(self):
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        self._create_ground_plane()
        self._create_envs()
        self.setup_video_camera()

    def get_low_level_obs(self):
        return self.legged_env.get_observations()
    
    def step(self, actions):

        # print(self.obs_buf[0], self.legged_env.get_observations()['obs'][0])

        self.render_gui()
        
        self.actions[:, 0:1] = torch.clamp(actions[:, 0:1], -0.4, 0.4)
        self.actions[:, 1:2] = torch.clamp(actions[:, 1:2], -0.4, 0.4)
        self.actions[:, 2:3] = torch.clamp(actions[:, 2:3], -0.4, 0.4)
        # print(self.legged_env_obs['obs_history'][0])

        new_actions = torch.cat([self.actions, self.commands], dim=1)
        
        for i in range(self.cfg.control.decimation):
            self.legged_env.set_commands(new_actions)
            # print(new_actions)
            legged_env_data = []
            with torch.no_grad():
                self.legged_env_obs, _, _, extras = self.legged_env.step(self.legged_env.policy(self.legged_env_obs))
                # print(self.legged_env_obs['obs_history'][0][-70+18:-68+18], extras['joint_pos'][0][:2], self.legged_env.default_dof_pos[0, :2])
                legged_env_data.append({
                    'joint_pos': extras['joint_pos'],
                    'joint_vel': extras['joint_vel'],
                    'torques': extras['torques'],
                    'body_pos': extras['body_pos'],
                })
        
        self.extras.update({
            'legged_env': extras
        })
        
        env_ids = self.post_physics_step()

        
        self.dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dones[env_ids] = True

        # if 0 in env_ids:
        #     print("##################################")

        # clip observations

        # clip privileged observations

        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.dones, self.extras

    def post_physics_step(self):

        # if self.record_now:
        #     self.gym.step_graphics(self.sim)
        #     self.gym.render_all_camera_sensors(self.sim)
        self.last_base_pos[:, :] = self.base_pos[:, :2].clone()
        self.last_base_yaw[:, :] = self.base_yaw[:, :].clone()
        
        self.base_pos[:, :] = self.legged_env.base_pos[:, :2] - self.env_origins[:, :2]
        self.base_quat = self.legged_env.base_quat[:, :].clone()
        self.base_yaw = torch.atan2(2.0*(self.base_quat[:, 0]*self.base_quat[:, 1] + self.base_quat[:, 3]*self.base_quat[:, 2]), 1. - 2.*(self.base_quat[:, 1]*self.base_quat[:, 1] + self.base_quat[:, 2]*self.base_quat[:, 2])).view(-1, 1)


        self.distance_travelled[:] += torch.norm(self.base_pos[:, :2] - self.last_base_pos[:, :2], dim=-1)
        # print(self.distance_travelled.shape)
        # self.num_collisions = self.legged_env
        self.episode_length_buf += 1

        # print('diff in pos', self.base_pos[0, :2] - last_base_pos[0, :2])
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.reset_idx(env_ids)

        self.compute_observations()
        # if 0 in env_ids:
        #     print('2', self.legged_env.base_pos[0, :2], self.base_pos[0, :2])
        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
        
        self.last_actions[:] = self.actions[:]

        
        self._render_headless()
        
        return env_ids

    def check_termination(self):
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        
        self.gs_buf = (torch.abs(self.base_pos[:, 0] - self.goal_positions)) < 0.1
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gs_buf
        
    def compute_reward(self):
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            if 'terminal' in name:
                continue
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            for i in range(len(self.reward_functions)):
                name = self.reward_names[i]
                if 'terminal' not in name:
                    continue
                # print(rew)
                rew = self.reward_functions[i]() * self.reward_scales[name]
                self.rew_buf += rew
                if torch.sum(rew) >= 0:
                    self.rew_buf_pos += rew
                elif torch.sum(rew) <= 0:
                    self.rew_buf_neg += rew
                self.episode_sums[name] += rew
        # if self.cfg.rewards.only_positive_rewards:
        #     self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
        #     self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
        self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        self.env_step += 1
        
        self.base_pos[:, :] = self.legged_env.base_pos[:, :2] - self.env_origins[:, :2]
        self.base_quat = self.legged_env.base_quat[:, :].clone()

        torques = self.legged_env.torques[:, :].clone() # .to(self.inference_model_device)

        # projected_gravity = self.legged_env.projected_gravity[:, :].clone().to(self.inference_model_device)
        
        # joint_pos = ((self.legged_env.dof_pos[:, :] - self.legged_env.default_dof_pos) * self.legged_env.obs_scales.dof_pos).clone().to(self.inference_model_device)
        
        # joint_vel = self.legged_env.dof_vel[:, :].clone().to(self.inference_model_device) * self.legged_env.obs_scales.dof_vel

        inp_obs = self.legged_env_obs['obs_history'][:, -70:].clone()
        projected_gravity = inp_obs[:, :3]
        joint_pos = inp_obs[:, 18:30]
        joint_vel = inp_obs[:, 30:42]
        torques = self.legged_env.torques[:, :].clone() * 0.08

        self.last_five_pos[:, :-1] = self.last_five_pos[:, 1:].clone()
        self.last_five_pos[:, -1, :] = self.base_pos[:, :2].clone()

        vel = (self.last_five_pos[:, -1] - self.last_five_pos[:, 0]) / (0.04 * (self.vel_window_size-1)) # shape (num_envs, 2)
      
        vel_x = torch.cos(self.base_yaw) * vel[:, :1] + torch.sin(self.base_yaw) * vel[:, 1:2]
        vel_y = -torch.sin(self.base_yaw) * vel[:, :1] + torch.cos(self.base_yaw) * vel[:, 1:2]

        vel = torch.cat([vel_x, vel_y], dim=-1)

        vel = vel.view(-1, 2)
        # print(vel[0], self.legged_env.base_lin_vel[0, :2], vel[0] - self.legged_env.base_lin_vel[0, :2])

        if self.use_localization_model:
            # print()
            self.pose_inp[self.env_ids, self.env_step, :] = torch.cat([projected_gravity, joint_pos, joint_vel], dim=-1).unsqueeze(1)
            obs = (self.localization_inference.predict(self.pose_inp[:, 1:, :])[self.env_ids, self.env_step-1, :]).squeeze(1).to(self.device)

            obs = torch.cat([obs, self.actions.clone()], dim=-1)


            obs_yaw = torch.atan2(2.0*(self.base_quat[:, 0]*self.base_quat[:, 1] + self.base_quat[:, 3]*self.base_quat[:, 2]), 1. - 2.*(self.base_quat[:, 1]*self.base_quat[:, 1] + self.base_quat[:, 2]*self.base_quat[:, 2])).view(-1, 1)

            # setup obs buf and scale it to normalize observations
            true_obs = torch.cat([((self.legged_env.base_pos[:, :1] - self.env_origins[:, :1])), (self.legged_env.base_pos[:, 1:2] - self.env_origins[:, 1:2]), obs_yaw, self.legged_env.base_lin_vel[:, :2], self.legged_env.base_ang_vel[:, 2:], self.actions.clone()], dim = -1)
            scaling_vec = torch.tensor([0.33, 1], device=self.device)

            # patch_set = get_visualization(0, true_obs[:, :6], None, obs[:, :6]*torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=obs.device), None, None, estimate_pose=True)
            # print(len(patch_set))
            # fig, ax = plt.subplots(nrows=1, ncols=1)
            # ax.set_xlim(-1.0, 4.0)
            # ax.set_ylim(-1.0, 1.0)
            # ax.add_patch(patch_set[0])
            # ax.add_patch(patch_set[1])
            # plt.show()
            
        else:
            # if not self.use_scene_model:
            # obs_yaw = torch.atan2(2.0*(self.base_quat[:, 0]*self.base_quat[:, 1] + self.base_quat[:, 3]*self.base_quat[:, 2]), 1. - 2.*(self.base_quat[:, 1]*self.base_quat[:, 1] + self.base_quat[:, 2]*self.base_quat[:, 2])).view(-1, 1)

            # print('obs yaw', obs_yaw[0]*1/3.14)
            # setup obs buf and scale it to normalize observations #
            # if np.random.uniform() > 0.05:
            
            # obs = torch.cat([((self.legged_env.base_pos[:, :1] - self.env_origins[:, :1])), (self.legged_env.base_pos[:, 1:2] - self.env_origins[:, 1:2]), self.base_yaw, vel, self.actions.clone()], dim = -1)
            # scaling_vec = torch.tensor([0.25, 1, 1/3.14]  + [1/0.4, 1/0.4] + [1/0.4, 1/0.4, 1/0.4], device=self.device)

            obs = torch.cat([((self.legged_env.base_pos[:, :1] - self.env_origins[:, :1])), (self.legged_env.base_pos[:, 1:2] - self.env_origins[:, 1:2]), self.base_yaw, self.legged_env.base_lin_vel[:, :2], self.actions.clone()], dim = -1)
            scaling_vec = torch.tensor([0.25, 1, 1/3.14] + [1/0.4, 1/0.4] + [1/0.4, 1/0.4, 1/0.4], device=self.device)

            # obs = torch.cat([joint_pos, torques, ((self.legged_env.base_pos[:, :1] - self.env_origins[:, :1])), (self.legged_env.base_pos[:, 1:2] - self.env_origins[:, 1:2]), self.base_yaw, self.actions.clone()], dim = -1)
            # scaling_vec = torch.tensor([1.]*12 + [0.08] * 12 + [0.25, 1, 1/3.14] + [1/0.4, 1/0.4, 1/0.4], device=self.device)

            # obs = torch.cat([((self.legged_env.base_pos[:, :1] - self.env_origins[:, :1])), (self.legged_env.base_pos[:, 1:2] - self.env_origins[:, 1:2]), self.base_yaw, vel, self.actions.clone()], dim = -1)

            # print(vel[0], self.legged_env.base_lin_vel[0, :2], vel[0] - self.legged_env.base_lin_vel[0, :2])
            # obs = torch.cat([((self.legged_env.base_pos[:, :1] - self.env_origins[:, :1])), (self.legged_env.base_pos[:, 1:2] - self.env_origins[:, 1:2]), self.base_yaw, self.actions.clone()], dim = -1)
            # scaling_vec = torch.tensor([0.25, 1, 1/3.14] + [1/0.4, 1/0.4, 1/0.4], device=self.device)

            # obs = torch.cat([joint_pos, torques, ((self.legged_env.base_pos[:, :1] - self.env_origins[:, :1])), (self.legged_env.base_pos[:, 1:2] - self.env_origins[:, 1:2]), self.base_yaw, self.actions.clone()], dim = -1)
            # scaling_vec = torch.tensor([1.]*12 + [0.08] * 12 + [0.25, 1, 1/3.14] + [1/0.4, 1/0.4, 1/0.4], device=self.device)
            # else:
            #     obs = torch.cat([((self.last_base_pos[:, :1])), (self.last_base_pos[:, 1:2]), self.last_base_yaw, self.legged_env.base_lin_vel[:, :2], self.actions.clone()], dim = -1)
            #     scaling_vec = torch.tensor([0.25, 1, 1/3.14]  + [1/0.4, 1/0.4] + [1/0.4, 1/0.4, 1/0.4], device=self.device)
            #########################################################

            # print()

            # ll_o
        
            # setup obs buf without vel and scale it to normalize observations #
            # if np.random.uniform() > 0.05:
            #     obs = torch.cat([((self.base_pos[:, :1] - self.env_origins[:, :1])), (self.base_pos[:, 1:2] - self.env_origins[:, 1:2]), obs_yaw, self.actions.clone()], dim = -1)
            #     scaling_vec = torch.tensor([0.25, 1, 1/3.14] + [1/0.4, 1/0.4, 1/0.4], device=self.device)
            # else:
            #     obs = torch.cat([((self.last_base_pos[:, :1])), (self.last_base_pos[:, 1:2]), obs_yaw, self.actions.clone()], dim = -1)
            #     scaling_vec = torch.tensor([0.25, 1, 1/3.14, 1/0.4, 1/0.4, 1/0.4], device=self.device)
            #########################################################

            # setup obs buf with legged env torques without vel and scale it to normalize observations #
            # obs = torch.cat([((self.base_pos[:, :1] - self.env_origins[:, :1])), (self.base_pos[:, 1:2] - self.env_origins[:, 1:2]), obs_yaw, self.legged_env.torques, self.actions.clone()], dim = -1)
            # scaling_vec = torch.tensor([0.25, 1, 1/3.14] + [0.08]*12 + [1/0.4, 1/0.4, 1/0.4], device=self.device)
            #########################################################

            # obs = torch.cat([((self.legged_env.base_pos[:, :1] - self.env_origins[:, :1])), (self.legged_env.base_pos[:, 1:2] - self.env_origins[:, 1:2]), obs_yaw, self.actions.clone()], dim = -1)
            # scaling_vec = torch.tensor([0.25, 1, 1/3.14, 1/0.4, 1/0.4, 1/0.4], device=self.device)
            
            # obs scaling or normalization
            obs *= scaling_vec
        
            # obs[:, :6] += torch.randn_like(obs[:, :6]) * 0.05 * scaling_vec[:6]

        if self.use_obstacle_model:
            # start = time.time()
            self.priv_info_inp[self.env_ids, self.env_step, :-6] = torch.cat([torques, projected_gravity, joint_pos, joint_vel], dim=-1).unsqueeze(1).to(self.inference_model_device)

            self.priv_info_inp[self.env_ids, self.env_step, -6:] = obs[:, :6].unsqueeze(1).to(self.inference_model_device)

            # start = time.time()
            priv_obs = self.obstacle_inference.predict(self.priv_info_inp[:, 1:, :].clone())[self.env_ids, self.env_step-1, :].squeeze(1)
            # print("obstacle inference time: ", time.time() - start)
            
            priv_obs[:, :7] = priv_obs[:, :7] * (torch.prod(priv_obs[:, 5:7], dim=-1) > 0.1).float().view(-1, 1)
            priv_obs[:, 7:14] = priv_obs[:, 7:14] * (torch.prod(priv_obs[:, 12:14], dim=-1) > 0.1).float().view(-1, 1)
            priv_obs[:, 14:21] = priv_obs[:, 14:21] * (torch.prod(priv_obs[:, 19:21], dim=-1) > 0.1).float().view(-1, 1)
            priv_obs = priv_obs.to(self.device)
            self.privileged_obs_buf[:] = priv_obs[:]
            # self.world_env_obs, self.full_seen_world_obs = self.world_env.get_block_obs()

            # if torch.sum(priv_obs[0]) > 0:
            #     frame = get_visualization(0, obs[:, :6]*torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=obs.device), self.world_env_obs, obs[:, :6]*torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=obs.device), priv_obs*torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=priv_obs.device), self.full_seen_world_obs, estimate_pose=False)
            #     print(len(frame))

            #     fig, axes = plt.subplots(2, 2, figsize=(48, 24))
            #     ax = axes.flatten()
            #     pred_robot, robot, robot_1, robot_2, robot_3 = frame[0], frame[1], frame[2], frame[3], frame[4]

            #     ax[0].add_patch(pred_robot)
            #     ax[0].add_patch(robot)
            #     ax[1].add_patch(robot_1)
            #     ax[2].add_patch(robot_2)
            #     ax[3].add_patch(robot_3)

            #     ax[0].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='all', aspect='auto')
            #     ax[1].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='truth', aspect='auto')
            #     ax[2].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='predicted', aspect='auto')
            #     ax[3].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='full seen world', aspect='auto')
                
            #     for i in range(2):
            #         j = i*6 + 5
            #         ax[0].add_patch(frame[j])
            #         ax[0].add_patch(frame[j+1])

            #         ax[1].add_patch(frame[j+3])
            #         ax[2].add_patch(frame[j+4])

            #         ax[3].add_patch(frame[j+5])
                    
            #     plt.show()
            

        else:
            
            # setup privileged obs buf and scale it to normalize observations
            self.world_env_obs, self.full_seen_world_obs = self.world_env.get_block_obs()
            self.privileged_obs_buf[:] = self.world_env_obs.clone()
            priv_obs = self.privileged_obs_buf.clone()[:, :18]

            # add scaled noise
            priv_obs *= torch.tensor([1, 1, 0.33, 1, 1/3.14, 1, 1/1.7, 1, 1] * 2, device=self.device)
            
        # add scaled noise
        
        self.obs_buf[:] = torch.cat([obs], dim=-1)

    def get_privileged_obs(self):
        return self.privileged_obs_buf

    def get_full_seen_world_obs(self):
        return self.full_seen_world_obs

    def reset(self):
        self.legged_env_obs = self.legged_env.reset()
        self.world_env_obs, self.full_seen_world_obs = self.world_env.reset()
        self.reset_idx(torch.arange(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long))
        # self.reset_video_camera()
        # self.step(torch.zeros_like(self.actions))
        self.base_pos[:, :] = self.legged_env.base_pos[:, :2] - self.env_origins[:, :2]
        # self.goal_positions[self.base_pos[:, 0] < 1.5] = 3.2
        # self.goal_positions[self.base_pos[:, 0] >= 1.5] = -0.2

        # print(self.base_pos[:, 0], self.goal_positions)

        return self.obs_buf

    def reset_idx(self, env_ids):

        if len(env_ids) == 0:
            return
        
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        # for env_id in train_env_ids:
            # print(env_id, self.world_env.inplay[env_id.item()])
            # self.world_env.world_types_success[self.world_env.inplay[env_id.item()].item()] += self.gs_buf[env_id.item()] * 1
            # self.world_env.world_types_count[self.world_env.inplay[env_id.item()].item()] += 1

        self.legged_env.reset_idx(env_ids)
        self.legged_env.compute_observations()

        
        self.world_env.reset_idx(env_ids)
        self.world_env_obs, self.full_seen_world_obs = self.world_env.get_block_obs()

        # self.legged_env.stand_still(env_ids)

        
        self.reset_video_camera(env_ids)

        self.success_envs[env_ids] += self.gs_buf[env_ids] * 1
        self.count_envs[env_ids] += 1
        # record all extras information here
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["train/episode"]['rew_' + key] = torch.mean(
                    self.episode_sums[key][train_env_ids])
                
                self.episode_sums[key][train_env_ids] = 0.
            # if torch.sum(self.count_envs[0:self.num_train_envs] > 0) == self.num_train_envs:
                # self.extras["train/episode"]['success_rate'] = self.success_envs[0:self.num_train_envs] / self.count_envs[0:self.num_train_envs]
                # print("success rate for all envs is ", torch.mean(self.success_envs[0:self.num_train_envs] / self.count_envs[0:self.num_train_envs]))
            
            # for k, v in d_env.world_types_count.items():
            #     if self.world_env.world_types_count[k] > 10:
            #         print("success rate for world type ", k, " is ", self.world_env.world_types_success[k] / self.world_env.world_types_count[k])
            #         self.world_env.world_types_count[k] = 0
            #         self.world_env.world_types_success[k] = 0

        eval_env_ids = env_ids[env_ids >= self.num_train_envs]
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}
            for key in self.episode_sums.keys():
                if key == 'intrinsic':
                    continue
                # save the evaluation rollout result if not already saved
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.episode_sums[key][eval_env_ids] = 0.
            # if torch.sum(self.count_envs[self.num_train_envs:] > 0) == self.num_eval_envs:
            #     self.extras["eval/episode"]['success_rate'] = self.success_envs[self.num_train_envs:] / self.count_envs[self.num_train_envs:]

        self.reset_buf[env_ids] = False
        self.gs_buf[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.
        self.env_step[env_ids] = 0
        self.base_yaw[env_ids] = 0.
        self.distance_travelled[env_ids] = 0.
        self.base_pos[env_ids, :] = 0.
        self.last_base_pos[env_ids, :] = 0.
        self.last_base_yaw[env_ids, :] = 0.
        self.last_five_pos[env_ids, :] = 0.

        if self.use_localization_model:
            self.pose_inp[env_ids, :, :] = 0.
        if self.use_obstacle_model:
            self.priv_info_inp[env_ids, :, :] = 0.
        # print(self.goal_positions[env_ids])

        # if count envs for every env is greater than 10, reset counts to 0 and success to 0
        text_arr = []
        if torch.sum(self.count_envs[:self.num_train_envs] > 0) == self.num_train_envs:
            for k, v in self.world_env.world_types_count.items():
                if self.world_env.world_types_count[k] > 0:
                    # if k == 0:
                    #     if (self.world_env.world_types_success[k] / self.world_env.world_types_count[k]).item() > 0.95:
                    #         self.world_env.world_sampling_dist[1] += 1.0
                    #         self.world_env.world_sampling_dist[0] = self.world_env.world_sampling_dist[0]/(self.world_env.world_sampling_dist[0] + self.world_env.world_sampling_dist[1])
                    #         self.world_env.world_sampling_dist[1] = 1.0 - self.world_env.world_sampling_dist[0]

                    text_arr.append(f'{k}: ' + str((self.world_env.world_types_success[k] / self.world_env.world_types_count[k]).item()))
                    
            text_arr.append('train: ' + str((torch.sum(self.success_envs[:self.num_train_envs]) / torch.sum(self.count_envs[:self.num_train_envs])).item()))

            self.count_envs[:self.num_train_envs] = 0
            self.success_envs[:self.num_train_envs] = 0

            for k, v in self.world_env.world_types_count.items():
                self.world_env.world_types_count[k] = 0
                self.world_env.world_types_success[k] = 0

                    
        if torch.sum(self.count_envs[self.num_train_envs:] > 0) == self.num_eval_envs:
            text_arr.append('eval: ' + str((torch.sum(self.success_envs[self.num_train_envs:]) / torch.sum(self.count_envs[self.num_train_envs:])).item()))
            self.count_envs[self.num_train_envs:] = 0
            self.success_envs[self.num_train_envs:] = 0

        if len(text_arr) > 0:
            print(', '.join(text_arr))

        
        return self.obs_buf, self.privileged_obs_buf

    def _parse_cfg(self, cfg:Cfg):
        self.dt = self.cfg.control.decimation * 4 * self.sim_params.dt # (2 * 4 * 0.005 = 0.04s) -> 100/4 = 25Hz
        self.obs_scales = self.cfg.obs_scales
        self.reward_scales = vars(self.cfg.reward_scales)
        
        self.max_episode_length = cfg.env.max_episode_length


    def _get_env_origins(self, env_ids):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        num_cols = np.floor(np.sqrt(len(env_ids)))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        xx, yy = xx.to(device=self.device), yy.to(device=self.device)
        spacing = self.cfg.env.env_spacing
        self.env_origins[env_ids, 0] = spacing * xx.flatten()[:len(env_ids)]
        self.env_origins[env_ids, 1] = spacing//2 * yy.flatten()[:len(env_ids)]
        self.env_origins[env_ids, 2] = 0.

    
    def _call_train_eval(self, func, env_ids):
        env_ids_train = env_ids[env_ids < self.num_train_envs]
        env_ids_eval = env_ids[env_ids >= self.num_train_envs]

        ret, ret_eval = None, None

        if len(env_ids_train) > 0:
            ret = func(env_ids_train, self.cfg)
        if len(env_ids_eval) > 0:
            ret_eval = func(env_ids_eval, self.eval_cfg)
            if ret is not None and ret_eval is not None: ret = torch.cat((ret, ret_eval), axis=-1)
        return ret

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        bx, by, bz = self.legged_env.base_pos[0, 0], self.legged_env.base_pos[0, 1], self.legged_env.base_pos[0, 2]
        self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                     gymapi.Vec3(bx, by, bz))
        
        # self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(0.1, 0.2, 3.0),
        #                              gymapi.Vec3(0.0, 0.0, 0.5))
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        return img.reshape([w, h // 4, 4])

    def _render_headless(self):
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            
            bx, by, bz = self.legged_env.base_pos[0, 0], self.legged_env.base_pos[0, 1], self.legged_env.base_pos[0, 2]
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx - 1.0, by, bz + 2.0),
                                         gymapi.Vec3(bx, by, bz))
            # self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(0.1, 0.2, 3.0),
            #                          gymapi.Vec3(0.0, 0.0, 0.5))
            self.video_frame = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera,
                                                         gymapi.IMAGE_COLOR)
            self.video_frame = self.video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))

            self.video_frames.append(self.video_frame)

        if self.record_eval_now and self.complete_video_frames_eval is not None and len(
                self.complete_video_frames_eval) == 0:
            if self.eval_cfg is not None:
                bx, by, bz = self.legged_env.base_pos[self.num_train_envs, 0], self.legged_env.base_pos[self.num_train_envs, 1], \
                             self.legged_env.base_pos[self.num_train_envs, 2]
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(bx -1.0, by, bz + 2.0),
                                             gymapi.Vec3(bx, by, bz))
                self.video_frame_eval = self.gym.get_camera_image(self.sim, self.envs[self.num_train_envs],
                                                                  self.rendering_camera_eval,
                                                                  gymapi.IMAGE_COLOR)
                self.video_frame_eval = self.video_frame_eval.reshape(
                    (self.camera_props.height, self.camera_props.width, 4))
                self.video_frames_eval.append(self.video_frame_eval)


    def setup_video_camera(self):
        # if recording video, set up camera
        if self.cfg.env.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 1280
            self.camera_props.height = 720
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
            if self.eval_cfg is not None:
                self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[self.num_train_envs],
                                                                           self.camera_props)
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(1.5, 1, 3.0),
                                             gymapi.Vec3(0, 0, 0))

        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []

    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True

    def start_recording_eval(self):
        self.complete_video_frames_eval = None
        self.record_eval_now = True

    def pause_recording(self):
        self.complete_video_frames = []
        self.video_frames = []
        self.record_now = False

    def pause_recording_eval(self):
        self.complete_video_frames_eval = []
        self.video_frames_eval = []
        self.record_eval_now = False

    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames

    def get_complete_frames_eval(self):
        if self.complete_video_frames_eval is None:
            return []
        return self.complete_video_frames_eval

    def reset_video_camera(self, env_ids):
        if self.cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []

        if self.cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
            if self.complete_video_frames_eval is None:
                self.complete_video_frames_eval = []
            else:
                self.complete_video_frames_eval = self.video_frames_eval[:]
            self.video_frames_eval = []

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from go1_gym.envs.navigator.rewards import CoRLRewards
        reward_containers = {"CoRLRewards": CoRLRewards}
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                if 'terminal' not in key: 
                    self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        
    