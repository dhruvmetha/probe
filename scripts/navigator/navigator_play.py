from isaacgym import gymutil, gymapi
from go1_gym.envs.navigator.navigator_config import Cfg
from go1_gym.envs.navigator.navigator import Navigator
from go1_gym.envs.navigator.history_wrapper import NavigationHistoryWrapper
from high_level_control import Runner
from tqdm import tqdm
from matplotlib import patches as pch
from matplotlib import pyplot as plt
import torch
import numpy as np  
import os
import glob
import random
# import cv2
from pathlib import Path
from ml_logger import logger
from go1_gym import MINI_GYM_ROOT_DIR
from high_level_control.actor_critic import ActorCritic

def load_policy(env):
    actor_critic = ActorCritic(env.num_obs, env.num_privileged_obs, env.num_obs_history, env.num_actions)
    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    actor_critic.eval()
    policy = actor_critic.act_inference
    return policy

if __name__ == "__main__":
    import pickle as pkl
    import matplotlib.pyplot as plt
    recent_runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/high_level_policy/2024-02-17/navigator_train/*"), key=os.path.getmtime)
    
    # model_path = recent_runs[-1]
    # model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-08-25/navigator_train/114827.185849'
    # model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-08-25/navigator_train/201005.149004'
    # model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-08-25/navigator_train/201005.149004'
    # model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-08-27/navigator_train/123552.439413'
    # model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-09-01/navigator_train/160000.149004'
    # model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-09-01/navigator_train/201005.149004'
    
    # model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2024-02-15/navigator_train/050540.754504' # 050321.571692, 050540.754504
    # recent_runs = ['/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2024-02-18/navigator_train/035410.215673'] # 050321.571692, 050540.754504
    recent_runs = [
        '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2024-02-15/navigator_train/050540.754504', 
        '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2024-02-15/navigator_train/050321.571692', 
        '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2024-02-17/navigator_train/012555.571562', 
        '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2024-02-18/navigator_train/035410.215673',
        '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2024-02-19/navigator_train/055837.269443'
    ]
    # recent_runs = ['/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2024-02-19/navigator_train/064525.312710']


    # with open(model_path + "/parameters.pkl", 'rb') as file:
    #     pkl_cfg = pkl.load(file)
#     print(pkl_cfg.keys())
    #     cfg = pkl_cfg["Cfg"]
    #     print(cfg.keys())

    #     for key, value in cfg.items():
    #         if hasattr(Cfg, key):
    #             for key2, value2 in cfg[key].items():
    #                 setattr(getattr(Cfg, key), key2, value2)
    SEED = 68
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    
    Cfg.env.num_envs = 2048
    Cfg.env.max_episode_length = 1499
    Cfg.env.num_observation_history = 750
    Cfg.env.num_observations = 8
    save_data = True
    headless = True
    obs_name = '2_obs'
    
    env = Navigator(Cfg, sim_device='cuda:0', headless=headless, random_pose=False, use_localization_model=False, use_obstacle_model=False, inference_device='cuda:0')
    env = NavigationHistoryWrapper(env, save_data=save_data, save_folder=f'iros24_play_feb19/{obs_name}/data_store_set_{SEED}')
    obs = env.reset()
    for model_path in recent_runs[:]:
        print(model_path)
        logger.configure(Path(model_path).resolve())
        policy = load_policy(env)
        total_dones = 5000
        dones_ctr = 0
        progress_bar = tqdm(total=total_dones)
        
        for _ in range(Cfg.env.max_episode_length * 10):
            if dones_ctr >= total_dones:
                progress_bar.close()
                break
            patches = []
            with torch.no_grad():
                actions = policy(obs['obs_history'], obs['privileged_obs'])
                obs, _, done, _ = env.step(actions)
                env_ids = done.nonzero(as_tuple=True)[0]
                if len(env_ids) > 0:
                    obs['obs_history'][env_ids] = 0.
                    dones_ctr += len(env_ids)
                    progress_bar.update(len(env_ids))

        obs = env.reset()

    if save_data:
        for worker in env.workers:
            worker.join()
        