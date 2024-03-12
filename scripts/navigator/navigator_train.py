if __name__ == "__main__":
    from go1_gym.envs.navigator.navigator_config import Cfg
    from go1_gym.envs.navigator.navigator import Navigator
    from go1_gym.envs.navigator.history_wrapper import NavigationHistoryWrapper
    from high_level_control import Runner
    import torch
    import random
    import numpy as np  
    from pathlib import Path
    from ml_logger import logger
    from go1_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem

    logger.configure(logger.utcnow(f'high_level_policy/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )

    SEED = 45
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    Cfg.env.num_envs = 2048
    Cfg.env.max_episode_length = 1499
    Cfg.env.num_observation_history = 750
    Cfg.env.num_observations = 8
    obs = "3_obs"
    save_data = False
    headless = True

    env = Navigator(Cfg, sim_device='cuda:0', headless=headless, random_pose=False, use_localization_model=False, use_obstacle_model=False, inference_device='cuda:0')
    env = NavigationHistoryWrapper(env, save_data=save_data, save_folder=f'iros24/{obs}/data_store_set_{SEED}')

    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}")
    runner.learn(num_learning_iterations=1000, init_at_random_ep_len=False, eval_freq=100)