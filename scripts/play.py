import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    label = "gait-conditioned-agility/pretrain-v0/train"

    env, policy = load_env(label, headless=headless)

    num_eval_steps = 1000
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.65, 0.0, 0.0
    body_height_cmd = 0.15
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["pacing"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros(num_eval_steps)
    measured_y_vels = np.zeros(num_eval_steps)
    measured_yaw_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * 0.65
    target_y_vels = np.ones(num_eval_steps) * 0.65
    target_yaw_vels = np.ones(num_eval_steps) * 0.65
    joint_positions = np.zeros((num_eval_steps, 12))


    obs = env.reset()

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)

        # if i == 200:
        #     x_vel_cmd = 0.65
        #     y_vel_cmd = 0.0
        #     yaw_vel_cmd = 0.0
        # if i == 400:
        #     x_vel_cmd = 0.0
        #     y_vel_cmd = 0.65
        #     yaw_vel_cmd = 0.0
        # if i == 600:
        #     x_vel_cmd = 0.0
        #     y_vel_cmd = 0.0
        #     yaw_vel_cmd = 0.65
        # if i == 800:
        #     x_vel_cmd = 0.0
        #     y_vel_cmd = 0.0
        #     yaw_vel_cmd = 0.0
        
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        measured_y_vels[i] = env.base_lin_vel[0, 1]
        measured_yaw_vels[i] = env.base_ang_vel[0, 2]
        print(measured_x_vels[i], measured_yaw_vels[i])

        joint_positions[i] = env.dof_pos[0, :].cpu()

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(48, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")
    
    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_y_vels, color='black', linestyle="--", label="Measured")
    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_y_vels, color='black', linestyle="--", label="Desired")
    axs[1].legend()
    axs[1].set_title("Sideways Velocity")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")

    axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_yaw_vels, color='black', linestyle="--", label="Measured")
    axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_yaw_vels, color='black', linestyle="--", label="Desired")
    axs[2].legend()
    axs[2].set_title("Yaw Velocity")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Velocity (rad/s)")

    axs[3].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    axs[3].set_title("Joint Positions")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
