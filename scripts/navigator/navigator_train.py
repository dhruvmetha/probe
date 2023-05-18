if __name__ == "__main__":
    from go1_gym.envs.navigator.navigator_config import Cfg
    from go1_gym.envs.navigator.navigator import Navigator
    from go1_gym.envs.navigator.history_wrapper import NavigationHistoryWrapper
    from high_level_control import Runner
    import torch
    import numpy as np  
    from pathlib import Path
    from ml_logger import logger
    from go1_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem

    logger.configure(logger.utcnow(f'high_level_policy/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )

    Cfg.env.num_envs = 512
    env = Navigator(Cfg, sim_device='cuda:0', headless=True)
    env = NavigationHistoryWrapper(env, save_data=True, save_folder='random_pos_seed_test_3')

    # env.reset()

    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}")
    runner.learn(num_learning_iterations=1000, init_at_random_ep_len=False, eval_freq=100)

    # x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.65, 0, -0.1 # np.random.uniform(-0.65, 0.65), np.random.uniform(-0.65, 0.65), np.random.uniform(-0.65, 0.65)
    # body_height_cmd = 0.0
    # step_frequency_cmd = 3.0
    # gait = torch.tensor([0, 0, 0.5])
    # footswing_height_cmd = 0.08
    # pitch_cmd = 0.0
    # roll_cmd = 0.0
    # stance_width_cmd = 0.33

    # commands = torch.tensor([x_vel_cmd, y_vel_cmd, yaw_vel_cmd, body_height_cmd, step_frequency_cmd, *gait, 0.5, footswing_height_cmd, pitch_cmd, roll_cmd, stance_width_cmd], device='cuda:0').repeat(env.num_envs, 1)

    # print(commands.shape)

    # env.start_recording()

    # for i in range(1000):
    #     # print(i)
    #     obs, priv_obs, _, _, _ = env.step(commands)
    #     # print(priv_obs[0])

    #     frames = env.get_complete_frames()
    #     print(len(frames))
    #     if len(frames) > 0:
    #         env.pause_recording()


    # if i % 50 == 0:
    #     x_vel_cmd, y_vel_cmd, yaw_vel_cmd = np.random.uniform(-0.65, 0.65), np.random.uniform(-0.65, 0.65), np.random.uniform(-0.65, 0.65)
    #     commands = torch.tensor([x_vel_cmd, y_vel_cmd, yaw_vel_cmd, body_height_cmd, step_frequency_cmd, *gait, 0.5, footswing_height_cmd, pitch_cmd, roll_cmd, stance_width_cmd], device='cuda:0').repeat(env.num_envs, 1)