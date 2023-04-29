from go1_gym.envs.navigator.navigator_config import Cfg
from go1_gym.envs.navigator.navigator import Navigator
from go1_gym.envs.navigator.history_wrapper import NavigationHistoryWrapper
from high_level_control import Runner
import torch
import numpy as np  
import os
import glob
from pathlib import Path
from ml_logger import logger
from go1_gym import MINI_GYM_ROOT_DIR
from high_level_control.actor_critic import ActorCritic
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime

SAVE_FILE_NAME = 'new_round_1'

def worker(q, q_idx):
    data_path = Path(f'/common/users/dm1487/legged_manipulation_data/rollout_data/{SAVE_FILE_NAME}/{q_idx}')
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

def load_policy(env):
    actor_critic = ActorCritic(env.num_obs, env.num_privileged_obs, env.num_obs_history, env.num_actions)
    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    actor_critic.eval()
    policy = actor_critic.act_inference
    return policy

def collect_data(env):
    # input data
    input_obs = env.legged_env.get_observations()['obs'].clone()
    
    # find rotation
    obs_yaw = torch.atan2(2.0*(env.legged_env.base_quat[:, 0]*env.legged_env.base_quat[:, 1] + env.legged_env.base_quat[:, 3]*env.legged_env.base_quat[:, 2]), 1. - 2.*(env.legged_env.base_quat[:, 1]*env.legged_env.base_quat[:, 1] + env.legged_env.base_quat[:, 2]*env.legged_env.base_quat[:, 2])).view(-1, 1)

    # creating target obs
    obs = torch.cat([(env.legged_env.base_pos[:, :1] - env.env_origins[:, :1]), (env.legged_env.base_pos[:, 1:2] - env.env_origins[:, 1:2]), obs_yaw , env.legged_env.base_lin_vel[:, :2], env.legged_env.base_ang_vel[:, 2:], ], dim = -1)
    priv_obs = env.get_privileged_obs().clone()
    target_obs = torch.cat([obs, priv_obs], dim=-1)

    fsw_obs = env.get_full_seen_world_obs().clone()

    return input_obs, target_obs, fsw_obs
    

if __name__ == "__main__":


    num_workers = 4
    q_s = [mp.JoinableQueue(maxsize=500) for _ in range(num_workers)]
    workers = [mp.Process(target=worker, args=(q, idx)) for idx, q in enumerate(q_s)]
    for worker in workers:
        worker.daemon = True
        worker.start()
    
    recent_runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/high_level_policy/*/*/*"), key=os.path.getmtime)
    model_path = recent_runs[-1]
    model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-04-29/navigator_train/062711.386694'
    logger.configure(Path(model_path).resolve())

    # params = logger.load_pkl('parameters.pkl')
    num_envs = 5
    Cfg.env.num_envs = num_envs

    env = Navigator(Cfg, sim_device='cuda:0', headless=False)
    env = NavigationHistoryWrapper(env)
    
    obs = env.reset()
    policy = load_policy(env)
    
    num_eval_steps = 1000000

    # create buffers for data collection
    input_obs, target_obs, fsw_obs = collect_data(env)
    input_data = torch.zeros((env.num_envs, 750, input_obs.shape[-1]), device=env.device)
    actions_data = torch.zeros((env.num_envs, 750, 3), device=env.device)
    target_data = torch.zeros((env.num_envs, 750, target_obs.shape[-1]), device=env.device)
    fsw_data = torch.zeros((env.num_envs, 750, fsw_obs.shape[-1]), device=env.device)
    done_data = torch.zeros((env.num_envs, 750), dtype=torch.bool, device=env.device)

    actions = torch.zeros_like(env.actions)
    dones = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)
    env_idx = torch.arange(env.num_envs, dtype=torch.long, device=env.device).unsqueeze(-1)
    env_step = torch.zeros((env.num_envs, 1), dtype=torch.long, device=env.device).view(-1, 1)
    pbar = tqdm(total=1000000)
    for i in range(num_eval_steps):
        with torch.inference_mode():
            
            input_obs, target_obs, fsw_obs = collect_data(env)
            actions_ = actions.clone().unsqueeze(1)
            dones_data = dones.clone()

            input_data[env_idx, env_step, :] = input_obs.unsqueeze(1)
            target_data[env_idx, env_step, :] = target_obs.unsqueeze(1)
            fsw_data[env_idx, env_step, :] = fsw_obs.unsqueeze(1)
            actions_data[env_idx, env_step, :] = actions_
            done_data[env_idx, env_step] = ~(dones_data.view(-1, 1))

            env_step += 1
            
            done_envs = (dones*1.0).nonzero().view(-1)
            if done_envs.size(0) > 0:
                print('here')
                q_id = np.random.randint(0, num_workers)
                q_s[q_id].put({
                    'input': input_data[done_envs].clone().cpu(),
                    'actions': actions_data[done_envs].clone().cpu(),
                    'target': target_data[done_envs].clone().cpu(),
                    'fsw': fsw_data[done_envs].clone().cpu(),
                    'done': done_data[done_envs].clone().cpu(),
                    })
                env_step[done_envs] = 0
                input_data[done_envs, :, :] = 0.0
                actions_data[done_envs, :, :] = 0.0
                target_data[done_envs, :, :] = 0.0
                fsw_data[done_envs, :, :] = 0.0
                done_data[done_envs, :] = False
                pbar.update(done_envs.size(0))

            done_env_horizon = (env_step == 625).nonzero()[:, 0].view(-1)
            if done_env_horizon.size(0) > 0:
                q_id = np.random.randint(0, num_workers)
                q_s[q_id].put({
                    'input': input_data[done_env_horizon].clone().cpu(),
                    'actions': actions_data[done_env_horizon].clone().cpu(),
                    'target': target_data[done_env_horizon].clone().cpu(),
                    'fsw': fsw_data[done_env_horizon].clone().cpu(),
                    'done': done_data[done_env_horizon].clone().cpu(),
                    })
                env_step[done_env_horizon] = 0
                input_data[done_env_horizon, :, :] = 0.0
                actions_data[done_env_horizon, :, :] = 0.0
                target_data[done_env_horizon, :, :] = 0.0
                fsw_data[done_env_horizon, :, :] = 0.0
                done_data[done_env_horizon, :] = False
                pbar.update(done_env_horizon.size(0))
                # env.reset(done_env_horizon)

            actions = policy(obs['obs_history'], obs['privileged_obs'])
            obs, rewards, dones, info = env.step(actions)
            print(dones[0])
            # collect data from high level observations, process and low level observations
            # input_obs, target_obs, fsw_obs = collect_data(env)

    for q in q_s:
        q.join()

    for q in q_s:
        q.put(None)

    for worker in workers:
        worker.join()