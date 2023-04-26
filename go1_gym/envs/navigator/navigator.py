from isaacgym import gymutil, gymapi
import torch
import numpy as np

from go1_gym.envs.base.base_task import BaseTask
from go1_gym.envs.navigator.navigator_config import Cfg

from go1_gym.envs.base.legged_robot import LeggedRobot
from go1_gym.envs.base.legged_robot_config import Cfg as LeggedCfg

from go1_gym.envs.world.world import WorldAsset


from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv



class Navigator(BaseTask):
    def __init__(self, cfg: Cfg, sim_device, headless, num_envs=None, eval_cfg:Cfg=None, physics_engine="SIM_PHYSX", initial_dynamics_dict=None):
        self.cfg = cfg
        self.eval_cfg = eval_cfg

        if num_envs is not None:
            cfg.env.num_envs = num_envs
        self.num_envs = cfg.env.num_envs

        self.train_test_split = cfg.env.train_test_split

        self.num_train_envs = max(1, int(self.num_envs * self.train_test_split))
        self.num_eval_envs = self.num_envs - self.num_train_envs

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

        self.__init_buffers()

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self.record_now = False
        self.record_eval_now = False

        self._prepare_reward_function()


    def _pre_create_env(self):

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        self._get_env_origins(torch.arange(self.num_envs, device=self.device))

        # self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))

        self.legged_env = HistoryWrapper(VelocityTrackingEasyEnv(self.gym, self.sim, self.num_envs, LeggedCfg, self.sim_params, self.env_origins, device=self.device))
        self.legged_env.pre_create_actor()

        self.world_env = WorldAsset(self.gym, self.sim, self.num_envs, self.env_origins, device=self.device, train_ratio=self.train_test_split)
        self.world_env.pre_create_actor()
    
    def _create_envs(self):
        self._pre_create_env()
        for i in range(self.num_envs):
            env_lower = gymapi.Vec3(0., 0., 0.)
            env_upper = gymapi.Vec3(0., 0., 0.)
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i]   

            self.legged_env.create_actor(i, env_handle, pos)
            self.world_env.create_actor(i, env_handle, pos)

            self.envs.append(env_handle)

        self._post_create_env()
    
    def _post_create_env(self):
        self.legged_env.post_create_actor(self.envs)
        self.world_env.post_create_actor(self.envs)
        
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
        self.commands = torch.zeros(self.num_envs, 10, device=self.device, dtype=torch.float32) + torch.tensor([0.0, 3.0, 0, 0, 0.5, 0.5, 0.08, 0.0, 0.0, 0.33], device=self.device)
        self.gs_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.base_pos = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
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
    
    def step(self, actions):
        
        self.render_gui()

        self.actions[:, :3] = torch.clamp(actions[:, :3], -0.65, 0.65)

        new_actions = torch.cat([self.actions, self.commands], dim=1)

        for i in range(self.cfg.control.decimation):
            self.legged_env.set_commands(new_actions)
            with torch.no_grad():
                self.legged_env_obs, _, _, _ = self.legged_env.step(self.legged_env.policy(self.legged_env_obs))
        
        self.post_physics_step()

        # clip observations

        # clip privileged observations
        

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):

        # if self.record_now:
        #     self.gym.step_graphics(self.sim)
        #     self.gym.render_all_camera_sensors(self.sim)

        self.base_pos[:, :] = self.legged_env.base_pos[:, :2] - self.env_origins[:, :2]

        self.episode_length_buf += 1

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
        
        self.last_actions[:] = self.actions[:]

        self._render_headless()


    def check_termination(self):
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length

        self.gs_buf = (self.base_pos[:, 0] - 3.2) > 0

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gs_buf
        
    def compute_reward(self):
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
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

        self.base_quat = self.legged_env.base_quat[:, :]
        obs_yaw = torch.atan2(2.0*(self.base_quat[:, 0]*self.base_quat[:, 1] + self.base_quat[:, 3]*self.base_quat[:, 2]), 1. - 2.*(self.base_quat[:, 1]*self.base_quat[:, 1] + self.base_quat[:, 2]*self.base_quat[:, 2])).view(-1, 1)

        
        # setup obs buf and scale it to normalize observations
        self.obs_buf[:] = torch.cat([self.legged_env.base_pos[:, :2] - self.env_origins[:, :2], 
                                    obs_yaw, 
                                    self.legged_env.base_lin_vel[:, :2],    
                                    self.legged_env.base_ang_vel[:, 2:], self.actions[:, :3]], dim = -1)
        # add scaled noise

        # setup privileged obs buf and scale it to normalize observations
        self.world_env_obs, self.full_seen_world_obs = self.world_env.get_block_obs()
        self.privileged_obs_buf[:] = self.world_env_obs.clone()
        # add scaled noise

    def reset(self):
        self.legged_env_obs = self.legged_env.reset()
        self.world_env_obs = self.world_env.reset()
    
    def reset_idx(self, env_ids):

        if len(env_ids) == 0:
            return
        self.legged_env.reset_idx(env_ids)
        self.world_env.reset_idx(env_ids)

        self.reset_buf[env_ids] = False
        self.gs_buf[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.
        
        self.reset_video_camera(env_ids)

        # record all extras information here
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["train/episode"]['rew_' + key] = torch.mean(
                    self.episode_sums[key][train_env_ids])
                self.episode_sums[key][train_env_ids] = 0.
        eval_env_ids = env_ids[env_ids >= self.num_train_envs]
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}
            for key in self.episode_sums.keys():
                # save the evaluation rollout result if not already saved
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.episode_sums[key][eval_env_ids] = 0.

    def _parse_cfg(self, cfg:Cfg):
        self.dt = self.cfg.control.decimation * 4 * self.sim_params.dt
        self.obs_scales = self.cfg.obs_scales
        self.reward_scales = vars(self.cfg.reward_scales)
        # self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        # cfg.command_ranges = vars(cfg.commands)
        # if cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
        #     cfg.terrain.curriculum = False
        max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(max_episode_length_s / self.dt)
        self.max_episode_length = cfg.env.max_episode_length

        # cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        # cfg.domain_rand.rand_interval = np.ceil(cfg.domain_rand.rand_interval_s / self.dt)
        # cfg.domain_rand.gravity_rand_interval = np.ceil(cfg.domain_rand.gravity_rand_interval_s / self.dt)
        # cfg.domain_rand.gravity_rand_duration = np.ceil(cfg.domain_rand.gravity_rand_interval * cfg.domain_rand.gravity_impulse_duration)



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
        self.env_origins[env_ids, 1] = spacing * yy.flatten()[:len(env_ids)]
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
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        return img.reshape([w, h // 4, 4])

    def _render_headless(self):
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            bx, by, bz = self.legged_env.base_pos[0, 0], self.legged_env.base_pos[0, 1], self.legged_env.base_pos[0, 2]
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                         gymapi.Vec3(bx, by, bz))
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
                                             gymapi.Vec3(bx, by - 1.0, bz + 1.0),
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
            self.camera_props.width = 360
            self.camera_props.height = 240
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
        