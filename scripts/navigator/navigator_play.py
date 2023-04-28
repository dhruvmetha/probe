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

def load_policy(env):
    actor_critic = ActorCritic(env.num_obs, env.num_privileged_obs, env.num_obs_history, env.num_actions)
    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    actor_critic.eval()
    policy = actor_critic.act_inference
    return policy

if __name__ == "__main__":
    
    recent_runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/high_level_policy/*/*/*"), key=os.path.getmtime)
    model_path = recent_runs[-1]
    logger.configure(Path(model_path).resolve())

    # params = logger.load_pkl('parameters.pkl')
    Cfg.env.num_envs = 5

    env = Navigator(Cfg, sim_device='cuda:0', headless=True)
    env = NavigationHistoryWrapper(env)
    
    obs = env.reset()
    policy = load_policy(env)
    
    num_eval_steps = 5000
    
    env.start_recording()
    record_torques = []
    for i in range(num_eval_steps):
        with torch.inference_mode():
            actions = policy(obs['obs_history'], obs['privileged_obs'])
            obs, _, _, _ = env.step(actions)
            record_torques.append(env.legged_env.torques[0])
        
        frames = env.get_complete_frames()
        if len(frames) > 0:
            print('LOGGING_VIDEO')
            env.pause_recording()
            logger.save_video(frames, f"eval_videos/{i:05d}.mp4", fps=1 / (env.dt))
            logger.save_pkl(record_torques[-len(frames):], path=f'eval_torques/torques_{i}.pkl')
            record_torques = []
            env.start_recording()