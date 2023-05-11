import isaacgym
assert isaacgym
import torch
import gym
from .navigator import Navigator
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
import numpy as np

class NavigationHistoryWrapper(gym.Wrapper):
    def __init__(self, env: Navigator):
        super().__init__(env)
        self.env = env

        self.num_train_envs = self.env.num_train_envs

        self.obs_history_length = self.env.cfg.env.num_observation_history

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs



        self.input_data = torch.zeros((env.num_envs, 750, 70), device=env.device)
        self.actions_data = torch.zeros((env.num_envs, 750, 3), device=env.device)
        self.torques_data = torch.zeros((env.num_envs, 750, 12))
        self.target_data = torch.zeros((env.num_envs, 750, 27), device=env.device)
        self.fsw_data = torch.zeros((env.num_envs, 750, 21), device=env.device)
        self.done_data = torch.zeros((env.num_envs, 750), dtype=torch.bool, device=env.device)
        self.env_idx = torch.arange(env.num_envs, dtype=torch.long, device=env.device).unsqueeze(-1)
        self.env_step = torch.zeros((env.num_envs, 1), dtype=torch.long, device=env.device).view(-1, 1)
        self.dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        self.num_workers = 24
        self.q_s = [mp.JoinableQueue(maxsize=500) for _ in range(self.num_workers)]
        self.workers = [mp.Process(target=self.worker, args=(q, idx)) for idx, q in enumerate(self.q_s)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def collect_data(self, env):
        # input data

        input_obs = self.legged_env.obs_history[:, -70:].clone()
        target_obs = torch.cat([self.obs_history[:, -30:-24], self.obs_history[:, -21:]], dim=-1).clone()
        fsw_obs = self.env.get_full_seen_world_obs().clone()

        # input_obs = env.legged_env.get_observations()['obs'].clone()
        
        # # find rotation
        # obs_yaw = torch.atan2(2.0*(env.legged_env.base_quat[:, 0]*env.legged_env.base_quat[:, 1] + env.legged_env.base_quat[:, 3]*env.legged_env.base_quat[:, 2]), 1. - 2.*(env.legged_env.base_quat[:, 1]*env.legged_env.base_quat[:, 1] + env.legged_env.base_quat[:, 2]*env.legged_env.base_quat[:, 2])).view(-1, 1)

        # # creating target obs
        # obs = torch.cat([(env.legged_env.base_pos[:, :1] - env.env_origins[:, :1]), (env.legged_env.base_pos[:, 1:2] - env.env_origins[:, 1:2]), obs_yaw , env.legged_env.base_lin_vel[:, :2], env.legged_env.base_ang_vel[:, 2:], ], dim = -1)
        # priv_obs = env.get_privileged_obs().clone()
        # target_obs = torch.cat([obs, priv_obs], dim=-1)

        # fsw_obs = env.get_full_seen_world_obs().clone()

        return input_obs, target_obs, fsw_obs


    def step(self, action):

        input_obs, target_obs, fsw_obs = self.collect_data(self.env)
        actions_ = action.clone().unsqueeze(1)
        dones_data = self.dones.clone()

        self.input_data[self.env_idx, self.env_step, :] = input_obs.unsqueeze(1)
        self.target_data[self.env_idx, self.env_step, :] = target_obs.unsqueeze(1)
        self.fsw_data[self.env_idx, self.env_step, :] = fsw_obs.unsqueeze(1)
        self.actions_data[self.env_idx, self.env_step, :] = actions_
        self.done_data[self.env_idx, self.env_step] = ~(dones_data.view(-1, 1))

        # privileged information and observation history are stored in info
        obs, privileged_obs, rew, done, info = self.env.step(action)
        self.torques_data[self.env_idx, self.env_step, :] = torch.tensor(info["legged_env"]["torques"]).unsqueeze(1)

        self.env_step += 1

        # privileged_obs = info["privileged_obs"]
        # self.dones[:] = done
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        
        # ret = {'obs': obs.clone(), 'privileged_obs': privileged_obs.clone(), 'obs_history': self.obs_history.clone()}

        env_ids = done.nonzero(as_tuple=False).flatten()

        if len(env_ids) > 0:
            q_id = np.random.randint(0, self.num_workers)
            self.q_s[q_id].put({
                'input': self.input_data[env_ids].clone().cpu(),
                'actions': self.actions_data[env_ids].clone().cpu(),
                'target': self.target_data[env_ids].clone().cpu(),
                'torques': self.torques_data[env_ids].clone().cpu(),
                'fsw': self.fsw_data[env_ids].clone().cpu(),
                'done': self.done_data[env_ids].clone().cpu(),
                })
            self.env_step[env_ids] = 0
            self.input_data[env_ids, :, :] = 0.0
            self.actions_data[env_ids, :, :] = 0.0
            self.target_data[env_ids, :, :] = 0.0
            self.fsw_data[env_ids, :, :] = 0.0
            self.done_data[env_ids, :] = False

            self.obs_history[env_ids, :] = 0
            # self.env.obs_buf[env_ids, :] = 0
            # obs[env_ids, :] = 0

        done_env_horizon = (self.env_step >= 750).nonzero()[:, 0].view(-1)
        if done_env_horizon.size(0) > 0:
            print('time out dones', done_env_horizon)
            q_id = np.random.randint(0, self.num_workers)
            self.q_s[q_id].put({
                'input': self.input_data[done_env_horizon].clone().cpu(),
                'actions': self.actions_data[done_env_horizon].clone().cpu(),
                'target': self.target_data[done_env_horizon].clone().cpu(),
                'torques': self.torques_data[done_env_horizon].clone().cpu(),
                'fsw': self.fsw_data[done_env_horizon].clone().cpu(),
                'done': self.done_data[done_env_horizon].clone().cpu(),
                })
            self.env_step[done_env_horizon] = 0
            self.input_data[done_env_horizon, :, :] = 0.0
            self.actions_data[done_env_horizon, :, :] = 0.0
            self.target_data[done_env_horizon, :, :] = 0.0
            self.fsw_data[done_env_horizon, :, :] = 0.0
            self.done_data[done_env_horizon, :] = False

        ret = {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}
        
        return ret, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations() # all zeros initially
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1) # all zeros initially
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}  

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = self.env.reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        # this never gets called
        super().reset()
        # ret, _, _, _ = self.step(torch.zeros_like(self.env.actions))
        self.obs_history[:, :] = 0
        # ret = self.env.get_observations()
        # privileged_obs = self.env.get_privileged_observations()
        # return ret # {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}


    def worker(self, q, q_idx):
        data_path = Path(f'/common/users/dm1487/legged_manipulation/rollout_data/random_seed_7/{q_idx}')
        data_path.mkdir(parents=True, exist_ok=True)
        
        while True:
            item = q.get()
            if item is None:
                break
            # save the data
            for k, v in item.items():
                if isinstance(item[k], torch.Tensor):
                    item[k] = item[k].numpy()
            np.savez_compressed(data_path/f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz', **item)
            q.task_done()