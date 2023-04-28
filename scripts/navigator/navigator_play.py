if __name__ == "__main__":
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


    recent_runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/high_level_policy/*/*/*"), key=os.path.getmtime)
    model_path = recent_runs[-1]
    logger.configure(Path(model_path).resolve())

    # params = logger.load_pkl('parameters.pkl')
    Cfg.env.num_envs = 5

    env = Navigator(Cfg, sim_device='cuda:0', headless=False)
    env = NavigationHistoryWrapper(env)
    
    actor_critic = ActorCritic(env.num_obs, env.num_privileged_obs, env.num_actions)
    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    policy = actor_critic.act_inference
    
    obs = env.reset()

    num_eval_steps = 1000
    for i in range(num_eval_steps):
        obs, _, _, _ = env.step(policy(obs['obs'], obs['privileged_obs']))

    # runner = Runner(env, device=f"cuda:{gpu_id}")
    # runner.learn(num_learning_iterations=100000,init_at_random_ep_len=False, eval_freq=100)