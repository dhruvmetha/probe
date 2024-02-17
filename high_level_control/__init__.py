import time
from collections import deque
import copy
import os

import torch
from ml_logger import logger
from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .one_step_model import OneStepModel, OneStep
from .rollout_storage import RolloutStorage

from tqdm import tqdm


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


# class DataCaches:
#     def __init__(self, curriculum_bins):
#         from go1_gym_learn.ppo.metrics_caches import SlotCache, DistCache

#         self.slot_cache = SlotCache(curriculum_bins)
#         self.dist_cache = DistCache()


# caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 25  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 200  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 60

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-09-14/navigator_train/045600.557505'  # updated from load_run and chkpt
    resume_curriculum = False


class Runner:

    def __init__(self, env, device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions
                                      ).to(self.device)

        # hl_one_step_models = [OneStepModel(27, self.env.num_actions, 27).to(self.device) for i in range(10)]
        hl_one_step_models = None
        # hl_one_step_models = [OneStep(27, self.env.num_actions, 8).to(self.device) for i in range(10)]
        # ll_one_step_models = [OneStepModel(70, self.env.num_actions, 70).to(self.device) for i in range(10)]
        # one_step_models = None
        # ll_one_step_models = None

        if RunnerArgs.resume:
            # load pretrained weights from resume_path
            # from ml_logger import ML_Logger
            # loader = ML_Logger(root="http://escher.csail.mit.edu:8080",
            #                    prefix=RunnerArgs.resume_path)
            # weights = torch.load("/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-05-22/navigator_train/043506.747887/checkpoints/ac_weights_last.pt")
            # weights = torch.load("/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-05-23/navigator_train/052500.669242/checkpoints/ac_weights_last.pt")
            # weights = torch.load("/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-05-25/navigator_train/211446.471194/checkpoints/ac_weights_last.pt")
            # weights = torch.load("/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/runs/high_level_policy/2023-06-05/navigator_train/220208.270068/checkpoints/ac_weights_last.pt")
            weights = torch.load(f"{RunnerArgs.resume_path}/checkpoints/ac_weights_last.pt")
            actor_critic.load_state_dict(state_dict=weights)

            # if hasattr(self.env, "curricula") and RunnerArgs.resume_curriculum:
            #     # load curriculum state
            #     distributions = loader.load_pkl("curriculum/distribution.pkl")
            #     distribution_last = distributions[-1]["distribution"]
            #     gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
            #     for gait_id, gait_name in enumerate(self.env.category_names):
            #         self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
            #         print(gait_name)

        self.alg = PPO(actor_critic, one_step_models=hl_one_step_models, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions], [self.env.legged_env.num_obs])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        from ml_logger import logger
        # initialize writer
        assert logger.prefix, "you will overwrite the entire instrument server"

        logger.start('start', 'epoch', 'episode', 'run', 'step')

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        last_critic_obs, last_critic_privileged_obs = torch.zeros_like(obs_history), torch.zeros_like(privileged_obs)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        ep_intrinsic_reward = torch.zeros(self.env.num_train_envs, dtype=torch.float, device=self.device)

        self.env.start_recording()

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in tqdm(range(self.current_learning_iteration, tot_iter)):
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    start = time.time()
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs], obs_history[:num_train_envs])
                    # self.alg.add_low_level_obs(self.env.get_low_level_obs()['obs'][:num_train_envs])

                    # if eval_expert:
                    #     actions_eval = self.alg.actor_critic.act_teacher(obs_history[num_train_envs:],
                    #                                                      privileged_obs[num_train_envs:])
                    # else:
                    #     actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:])

                    actions_eval = self.alg.actor_critic.act(obs_history[num_train_envs:], privileged_obs[num_train_envs:])
                    

                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))
                    obs_dict, rewards, dones, infos = ret

                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)

                    self.alg.process_env_step(obs[:num_train_envs], rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):
                            logger.store_metrics(**infos['train/episode'])

                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):
                            logger.store_metrics(**infos['eval/episode'])

                    if 'curriculum' in infos:

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                    last_critic_obs[:num_train_envs, :], last_critic_privileged_obs[:num_train_envs, :] = obs_history[:num_train_envs, :], privileged_obs[:num_train_envs][:, :]
                    
                    done_env_ids = dones.nonzero(as_tuple=False).flatten()
                    if len(done_env_ids) > 0:
                        # ep_intrinsic_reward[done_env_ids] = 0.
                        obs_history[done_env_ids] = 0.
                        # print(obs_history[done_env_ids[0], -16:])

                
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

                # if it % curriculum_dump_freq == 0:
                #     logger.save_pkl({"iteration": it,
                #                      **caches.slot_cache.get_summary(),
                #                      **caches.dist_cache.get_summary()},
                #                     path=f"curriculum/info.pkl", append=True)

                #     if 'curriculum/distribution' in infos:
                #         logger.save_pkl({"iteration": it,
                #                          "distribution": distribution},
                #                          path=f"curriculum/distribution.pkl", append=True)

            # Learning step
            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = self.alg.update()

            logger.store_metrics(
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
            )

            if RunnerArgs.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:
                logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
                logger.job_running()

            if it % RunnerArgs.save_interval == 0:
                with logger.Sync():
                    logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    path = './tmp/legged_data'

                    os.makedirs(path, exist_ok=True)

                    # adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                    # adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                    # traced_script_adaptation_module = torch.jit.script(adaptation_module)
                    # traced_script_adaptation_module.save(adaptation_module_path)
                    # logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)

                    # body_path = f'{path}/body_latest.jit'
                    # body_model = copy.deepcopy(torch.nn.Sequential(self.alg.actor_critic.shared_memory, self.alg.actor_critic.actor)).to('cpu')
                    # # body_model = copy.deepcopy(self.alg.actor_critic.actor).to('cpu')
                    # traced_script_body_module = torch.jit.script(body_model)
                    # traced_script_body_module.save(body_path)

                    # logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

            self.current_learning_iteration += num_learning_iterations

        with logger.Sync():
            logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
            logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

            path = './tmp/legged_data'

            os.makedirs(path, exist_ok=True)

            # adaptation_module_path = f'{path}/adaptation_module_latest.jit'
            # adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
            # traced_script_adaptation_module = torch.jit.script(adaptation_module)
            # traced_script_adaptation_module.save(adaptation_module_path)

            body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.shared_memory, self.alg.actor_critic.actor).to('cpu')
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)

            # logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
            logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)


    def log_video(self, it):
        # if it - self.last_recording_it >= RunnerArgs.save_video_interval:
        #     self.env.start_recording()
        #     if self.env.num_eval_envs > 0:
        #         self.env.start_recording_eval()
        #     print("START RECORDING")
        #     self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)

            self.env.start_recording()
            print("START RECORDING")

        # if self.env.num_eval_envs > 0:
        #     frames = self.env.get_complete_frames_eval()
        #     if len(frames) > 0:
        #         self.env.pause_recording_eval()
        #         print("LOGGING EVAL VIDEO")
        #         logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)

    # def get_inference_policy(self, device=None):
    #     self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
    #     if device is not None:
    #         self.alg.actor_critic.to(device)
    #     return self.alg.actor_critic.act_inference

    # def get_expert_policy(self, device=None):
    #     self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
    #     if device is not None:
    #         self.alg.actor_critic.to(device)
    #     return self.alg.actor_critic.act_expert
