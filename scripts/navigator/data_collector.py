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

import multiprocessing as mp

def worker(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        # do something with the item
        print(f"Process {mp.current_process().name} processed item {item}")
        queue.task_done()


def load_policy(env):
    actor_critic = ActorCritic(env.num_obs, env.num_privileged_obs, env.num_obs_history, env.num_actions)
    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    actor_critic.eval()
    policy = actor_critic.act_inference
    return policy

if __name__ == "__main__":


    queue_1 = mp.JoinableQueue()
    queue_2 = mp.JoinableQueue()

    p1 = mp.Process(target=worker, args=(queue_1,))
    p1.daemon = True  # process will exit when the main process exits
    p1.start()

    p2 = mp.Process(target=worker, args=(queue_2,))
    p2.daemon = True
    p2.start()


    
    recent_runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/high_level_policy/*/*/*"), key=os.path.getmtime)
    model_path = recent_runs[-1]
    logger.configure(Path(model_path).resolve())

    # params = logger.load_pkl('parameters.pkl')
    num_envs = 5
    Cfg.env.num_envs = num_envs

    env = Navigator(Cfg, sim_device='cuda:0', headless=True)
    env = NavigationHistoryWrapper(env)
    
    obs = env.reset()
    policy = load_policy(env)
    
    num_eval_steps = 5000

    low_level_obs = env.legged_env.get_observations()['obs'].clone()

    # create buffers for data collection
    obs_data = torch.zeros(num_envs, obs['obs'].shape[1], device=env.device)
    obs_data = torch.zeros(num_envs, obs['obs'].shape[1], device=env.device)


    for i in range(num_eval_steps):
        with torch.inference_mode():
            actions = policy(obs['obs_history'], obs['privileged_obs'])
            obs, rewards, dones, info = env.step(actions)
            # collect data from high level observations, process and low level observations
            obs['obs']

            if len(dones.nonzero(as_tuple=False).flatten()) > 0:
                # randomly pick a queue to send data to
                queue = np.random.choice([queue_1, queue_2])

                # send data to queue
                

            
            
