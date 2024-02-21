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
    def __init__(self, env: Navigator, save_data=False, save_folder='random_seed_test_1'):
        super().__init__(env)
        self.env = env

        self.num_train_envs = self.env.num_train_envs
        self.save_data = save_data
        self.save_folder = save_folder

        self.obs_history_length = self.env.cfg.env.num_observation_history

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs


        self.input_data = torch.zeros((env.num_envs, 1500, 39), device=env.device)
        self.ll_actions_data = torch.zeros((env.num_envs, 1500, 12), device=env.device)
        self.actions_data = torch.zeros((env.num_envs, 1500, 3), device=env.device)
        self.torques_data = torch.zeros((env.num_envs, 1500, 12))
        self.target_data = torch.zeros((env.num_envs, 1500, 6), device=env.device)
        self.target_env_data = torch.zeros((env.num_envs, 1500, 27), device=env.device)
        self.fsw_data = torch.zeros((env.num_envs, 1500, 27), device=env.device)
        self.done_data = torch.zeros((env.num_envs, 1500), dtype=torch.bool, device=env.device)
        self.env_idx = torch.arange(env.num_envs, dtype=torch.long, device=env.device).unsqueeze(-1)
        self.env_step = torch.zeros((env.num_envs, 1), dtype=torch.long, device=env.device).view(-1, 1)
        self.dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        self.diff_base_pos = []

        self.standing_still_ctr = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) + 50

        if self.save_data:
            self.num_workers = 24
            self.q_s = [mp.JoinableQueue(maxsize=500) for _ in range(self.num_workers)]
            self.workers = [mp.Process(target=self.worker, args=(q, idx)) for idx, q in enumerate(self.q_s)]
            for worker in self.workers:
                worker.daemon = True
                worker.start()

        # self.checking_id = 0
        # self.print_now = True

    def collect_data(self, env):
        # input data
        
        # input_obs = self.legged_env.obs_history[:, -70:].clone() # 3 + 15 + 12 + 12 + 12 + 12 + 4 = 71
        target_env_obs = env.get_privileged_obs().clone()
        # target_obs = torch.cat([self.obs_history[:, -30:-24], self.obs_history[:, -21:]], dim=-1).clone()
        # target_obs = torch.cat([self.obs_history[:, -9:-3], priv_obs], dim=-1).clone()

        input_obs = self.legged_env.obs_history[:, -70:].clone()
        projected_gravity = input_obs[:, :3].clone()
        dof_pos = input_obs[:, 18:30]
        dof_vel = input_obs[:, 30:42]
       
        torques_applied = self.legged_env.torques.clone()
        ll_actions = input_obs[:, 42:54] # self.legged_env.actions.clone()

        obs_buf = env.obs_buf.clone()
        base_pos = obs_buf[:, -8:-6].clone()
        base_ang = obs_buf[:, -6:-5].clone()
        base_lin_vel = env.legged_env.base_lin_vel[:, :2].clone() * 1/0.4
        base_ang_vel = env.legged_env.base_ang_vel[:, 2:3].clone() * 1/0.4
        # actions = env.actions

        input_obs = torch.cat([projected_gravity, dof_pos, dof_vel, torques_applied], dim=-1) # 3 + 12 + 12 + 12 = 39
        target_obs = torch.cat([base_pos, base_ang, base_lin_vel, base_ang_vel], dim=-1) # 6
        fsw_obs = self.env.get_full_seen_world_obs().clone() # 21
        
        return input_obs, target_obs, target_env_obs, fsw_obs, ll_actions
        # return dof_pos, dof_vel, torques_applied, actions, base_pos, base_lin_vel, base_ang_vel, priv_obs, fsw_obs


    def step(self, action):
        if self.save_data:
            input_obs, target_obs, target_env_obs, fsw_obs, ll_actions = self.collect_data(self.env)
            actions_ = action.clone().unsqueeze(1)
            dones_data = self.dones.clone()

            self.input_data[self.env_idx, self.env_step, :] = input_obs.unsqueeze(1)
            self.target_data[self.env_idx, self.env_step, :] = target_obs.unsqueeze(1)
            self.target_env_data[self.env_idx, self.env_step, :] = target_env_obs.unsqueeze(1)
            self.fsw_data[self.env_idx, self.env_step, :] = fsw_obs.unsqueeze(1)
            self.ll_actions_data[self.env_idx, self.env_step, :] = ll_actions.unsqueeze(1)
            self.actions_data[self.env_idx, self.env_step, :] = actions_.to(self.env.device)
            self.done_data[self.env_idx, self.env_step] = ~(dones_data.view(-1, 1))

        # privileged information and observation history are stored in info
        # last_base_pos = self.env.base_pos[0, :2].clone()
        obs, privileged_obs, rew, done, info = self.env.step(action)
        # self.diff_base_pos.append(self.env.base_pos[0, :2] - last_base_pos[0, :2])

        if self.save_data:
            # self.torques_data[self.env_idx, self.env_step, :] = torch.tensor(info["legged_env"]["torques"]).unsqueeze(1)
            self.env_step += 1

        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        
        if self.save_data:
            env_ids = done.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                # print(self.legged_env.torques[env_ids[0], :])
                # self.checking_id = env_ids[0]
                # self.print_now = True
                q_id = np.random.randint(0, self.num_workers)
                self.q_s[q_id].put({
                    'input': self.input_data[env_ids].clone().cpu(), # dof_pos, dof_vel, torques_applied
                    'll_actions': self.ll_actions_data[env_ids].clone().cpu(), # ll_actions
                    'actions': self.actions_data[env_ids].clone().cpu(), # actions
                    'target': self.target_data[env_ids].clone().cpu(), # base_pos, base_ang, base_lin_vel, base_ang_vel
                    'target_env': self.target_env_data[env_ids].clone().cpu(), 
                    'fsw': self.fsw_data[env_ids].clone().cpu(),
                    'done': self.done_data[env_ids].clone().cpu(),
                    })
                self.env_step[env_ids] = 0
                self.input_data[env_ids, :, :] = 0.0
                self.ll_actions_data[env_ids, :, :] = 0.0
                self.actions_data[env_ids, :, :] = 0.0
                self.target_data[env_ids, :, :] = 0.0
                self.target_env_data[env_ids, :, :] = 0.0
                self.fsw_data[env_ids, :, :] = 0.0
                self.done_data[env_ids, :] = False

                self.obs_history[env_ids, :] = 0
                # self.env.obs_buf[env_ids, :] = 0
                # obs[env_ids, :] = 0

            done_env_horizon = (self.env_step >= 1500).nonzero()[:, 0].view(-1)
            if done_env_horizon.size(0) > 0:
                print('time out dones', done_env_horizon)
                q_id = np.random.randint(0, self.num_workers) 
                self.q_s[q_id].put({
                    'input': self.input_data[done_env_horizon].clone().cpu(),
                    'll_actions': self.ll_actions_data[done_env_horizon].clone().cpu(),
                    'actions': self.actions_data[done_env_horizon].clone().cpu(),
                    'target': self.target_data[done_env_horizon].clone().cpu(),
                    'target_env': self.target_env_data[done_env_horizon].clone().cpu(),
                    'fsw': self.fsw_data[done_env_horizon].clone().cpu(),
                    'done': self.done_data[done_env_horizon].clone().cpu(),
                    })
                self.env_step[done_env_horizon] = 0
                self.input_data[done_env_horizon, :, :] = 0.0
                self.ll_actions_data[done_env_horizon, :, :] = 0.0
                self.actions_data[done_env_horizon, :, :] = 0.0
                self.target_data[done_env_horizon, :, :] = 0.0
                self.target_env_data[done_env_horizon, :, :] = 0.0
                self.fsw_data[done_env_horizon, :, :] = 0.0
                self.done_data[done_env_horizon, :] = False

        ret = {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}
        # print('history wrapper', done)
        return ret, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations() # all zeros initially
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1) # all zeros initially
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}  

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = self.env.reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        # self.env.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        # this never gets called
        print('########################reseting########################')
        super().reset()
        # ret, _, _, _ = self.step(torch.zeros_like(self.env.actions))
        self.obs_history[:, :] = 0
        self.env_step[:] = 0
        self.input_data[:, :, :] = 0.0
        self.ll_actions_data[:, :, :] = 0.0
        self.actions_data[:, :, :] = 0.0
        self.target_data[:, :, :] = 0.0
        self.target_env_data[:, :, :] = 0.0
        self.fsw_data[:, :, :] = 0.0
        self.done_data[:, :] = False
        
        # ret = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        return { "privileged_obs": privileged_obs, "obs_history": self.obs_history}


    def worker(self, q, q_idx):
        data_path = Path(f'/common/users/dm1487/legged_manipulation_data_store/{self.save_folder}/{q_idx}')
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