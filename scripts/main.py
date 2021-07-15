import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm

import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
sys.path.append('./')
sys.path.append('..')
from utils.parameters import *
from storage.buffer import QLearningBufferExpert, QLearningBuffer
from helping_hands_rl_envs import env_factory
from utils.logger import Logger
from utils.schedules import LinearSchedule
from utils.torch_utils import rand_perlin_2d
from utils.env_wrapper import EnvWrapper

from agents.dqn_agent import DQNAgent
from networks.cnn import CNN

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_step(agent, replay_buffer, logger):
    batch = replay_buffer.sample(batch_size)
    loss, td_error = agent.update(batch)
    logger.trainingBookkeeping(loss, td_error.mean().item())
    logger.num_training_steps += 1
    if logger.num_training_steps % target_update_freq == 0:
        agent.updateTarget()

def saveModelAndInfo(logger, agent):
    logger.saveModel(logger.num_steps, env, agent)
    logger.saveLearningCurve(1000)
    logger.saveLossCurve(100)
    logger.saveTdErrorCurve(100)
    logger.saveRewards()
    logger.saveLosses()
    logger.saveTdErrors()
    logger.saveStepLeftCurve(1000)

def train():
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)

    # setup agent
    agent = DQNAgent()
    net = CNN().to(device)
    agent.initNetwork(net)

    if load_model_pre:
        agent.loadModel(load_model_pre)
    agent.train()

    # logging
    simulator_str = copy.copy(simulator)
    if simulator == 'pybullet':
        simulator_str += ('_' + robot)
    log_dir = os.path.join(log_pre, '{}'.format(alg))
    if note:
        log_dir += '_'
        log_dir += note

    logger = Logger(log_dir, env, 'train', num_processes, max_episode, log_sub)
    hyper_parameters['model_shape'] = agent.getModelStr()
    logger.saveParameters(hyper_parameters)

    if buffer_type == 'expert':
        replay_buffer = QLearningBufferExpert(buffer_size)
    else:
        replay_buffer = QLearningBuffer(buffer_size)
    exploration = LinearSchedule(schedule_timesteps=explore, initial_p=init_eps, final_p=final_eps)

    states, obs = envs.reset()

    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), envs, agent, replay_buffer)

    # pre train
    if load_buffer is not None and not load_sub:
        logger.loadBuffer(replay_buffer, load_buffer, load_n)
    if pre_train_step > 0:
        pbar = tqdm(total=pre_train_step)
        while len(logger.losses) < pre_train_step:
            t0 = time.time()
            train_step(agent, replay_buffer, logger)
            if logger.num_training_steps % 1000 == 0:
                logger.saveLossCurve(100)
                logger.saveTdErrorCurve(100)
            if not no_bar:
                pbar.set_description('loss: {:.3f}, time: {:.2f}'.format(float(logger.getCurrentLoss()), time.time()-t0))
                pbar.update(len(logger.losses)-pbar.n)

            if (time.time() - start_time) / 3600 > time_limit:
                logger.saveCheckPoint(args, envs, agent, replay_buffer)
                exit(0)
        pbar.close()
        logger.saveModel(0, 'pretrain', agent)
        # agent.sl = sl

    if not no_bar:
        pbar = tqdm(total=max_episode)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    while logger.num_episodes < max_episode:
        if fixed_eps:
            if logger.num_episodes < planner_episode:
                eps = 1
            else:
                eps = final_eps
        else:
            eps = exploration.value(logger.num_episodes)

        is_expert = 0
        actions_star_idx, actions_star = agent.getEGreedyActions(obs, eps)

        envs.stepAsync(actions_star, auto_reset=False)

        if len(replay_buffer) >= training_offset:
            for training_iter in range(training_iters):
                train_step(agent, replay_buffer, logger)

        states_, obs_, rewards, dones = envs.stepWait()
        steps_lefts = envs.getStepLeft()

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_obs_ = envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                states_[idx] = reset_states_[j]
                obs_[idx] = reset_obs_[j]


        for i in range(num_processes):
            replay_buffer.add(
                ExpertTransition(states[i], obs[i], actions_star_idx[i], rewards[i], states_[i],
                                 obs_[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
            )
        logger.stepBookkeeping(rewards.numpy(), steps_lefts.numpy(), dones.numpy())

        states = copy.copy(states_)
        obs = copy.copy(obs_)

        if (time.time() - start_time)/3600 > time_limit:
            break

        if not no_bar:
            timer_final = time.time()
            description = 'Steps:{}; Reward:{:.03f}; Explore:{:.02f}; Loss:{:.03f}; Time:{:.03f}'.format(
                logger.num_steps, logger.getCurrentAvgReward(1000), eps, float(logger.getCurrentLoss()),
                timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_episodes-pbar.n)
        logger.num_steps += num_processes

        if logger.num_steps % (num_processes * save_freq) == 0:
            saveModelAndInfo(logger, agent)

    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(args, envs, agent, replay_buffer)
    envs.close()

if __name__ == '__main__':
    train()