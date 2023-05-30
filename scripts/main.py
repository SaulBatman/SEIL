import os
import sys
import time
import copy
import collections
from tqdm import tqdm

sys.path.append('./')
sys.path.append('..')
from utils.parameters import *
from storage.buffer import QLearningBufferExpert, QLearningBuffer
from storage.aug_buffer import QLearningBufferAug
from utils.logger import Logger
from utils.schedules import LinearSchedule
from utils.env_wrapper import EnvWrapper

from utils.create_agent import createAgent
import threading

from utils.torch_utils import ExpertTransition
from utils.transition_sim import NpyBuffer, transitionSimulateSim
import matplotlib.pyplot as plt



def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def train_step(agent, replay_buffer, logger, p_beta_schedule):
    if buffer_type[:3] == 'per':
        beta = p_beta_schedule.value(logger.num_training_steps)
        batch, weights, batch_idxes = replay_buffer.sample(batch_size, beta)
        loss, td_error = agent.update(batch)
        new_priorities = np.abs(td_error.cpu()) + np.stack([t.expert for t in batch]) * per_expert_eps + per_eps
        replay_buffer.update_priorities(batch_idxes, new_priorities)
        logger.expertSampleBookkeeping(
            np.stack(list(zip(*batch))[-1]).sum() / batch_size)
    else:
        batch = replay_buffer.sample(batch_size)
        loss, td_error = agent.update(batch)

    logger.trainingBookkeeping(loss, td_error.mean().item())
    logger.num_training_steps += 1
    if logger.num_training_steps % target_update_freq == 0:
        agent.updateTarget()

def preTrainCURLStep(agent, replay_buffer, logger):
    if buffer_type[:3] == 'per':
        batch, weights, batch_idxes = replay_buffer.sample(batch_size, per_beta)
    else:
        batch = replay_buffer.sample(batch_size)
    loss = agent.updateCURLOnly(batch)
    logger.trainingBookkeeping(loss, 0)

def saveModelAndInfo(logger, agent):
    if save_multi_freq > 0:
        if logger.num_training_steps % save_multi_freq == 0:
            logger.saveMultiModel(logger.num_training_steps, env, agent)
    logger.saveModel(logger.num_steps, env, agent)
    logger.saveLearningCurve(20)
    logger.saveLossCurve(100)
    # logger.saveTdErrorCurve(100)
    logger.saveStepLeftCurve(100)
    logger.saveExpertSampleCurve(100)
    logger.saveEvalCurve()
    logger.saveRewards()
    logger.saveLosses()
    logger.saveTdErrors()
    logger.saveEvalRewards()

def evaluate(envs, agent, logger):
    states, obs = envs.reset()
    evaled = 0
    temp_reward = [[] for _ in range(num_eval_processes)]
    eval_rewards = []
    eval_success = []
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes, position=1, leave=False)
    while evaled < num_eval_episodes:
        actions_star_idx, actions_star = agent.getGreedyActions(states, obs)
        states_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)
        rewards = rewards.numpy()
        dones = dones.numpy()
        states = copy.copy(states_)
        obs = copy.copy(obs_)
        for i, r in enumerate(rewards.reshape(-1)):
            temp_reward[i].append(r)
        evaled += int(np.sum(dones))
        for i, d in enumerate(dones.astype(bool)):
            if d:
                R = 0
                for r in reversed(temp_reward[i]):
                    R = r + gamma * R
                eval_rewards.append(R)
                eval_success.append(np.sum(temp_reward[i]))
                temp_reward[i] = []
        if not no_bar:
            eval_bar.update(evaled - eval_bar.n)
    logger.eval_rewards.append(np.mean(eval_rewards[:num_eval_episodes]))
    logger.eval_success.append(np.mean(eval_success[:num_eval_episodes]))
    if not no_bar:
        eval_bar.clear()
        eval_bar.close()

def countParameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train():
    eval_thread = None
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    print('creating envs')
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    if simulate_n > 0:
        planner_envs = EnvWrapper(1, simulator, env, env_config, planner_config)
    else:
        planner_envs = envs
    # setup agent
    agent = createAgent()
    eval_agent = createAgent(test=True)
    # .train() is required for equivariant network
    agent.train()
    eval_agent.train()
    if load_model_pre:
        agent.loadModel(load_model_pre)

    # logging
    simulator_str = copy.copy(simulator)
    if simulator == 'pybullet':
        simulator_str += ('_' + robot)
    log_dir = os.path.join(log_pre, '{}_{}'.format(alg, model))
    if note:
        log_dir += '_'
        log_dir += note

    logger = Logger(log_dir, env, 'train', num_processes, max_train_step, gamma, log_sub)
    hyper_parameters['model_shape'] = agent.getModelStr()
    logger.saveParameters(hyper_parameters)

    if buffer_type == 'normal':
        replay_buffer = QLearningBuffer(buffer_size)
    elif buffer_type == 'aug':
        replay_buffer = QLearningBufferAug(buffer_size, aug_n=buffer_aug_n)
    else:
        raise NotImplementedError
    p_beta_schedule = LinearSchedule(schedule_timesteps=max_train_step, initial_p=per_beta, final_p=1.0)

    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), envs, agent, replay_buffer)

    if not load_sub:
        if load_buffer is not None:
            if load_buffer.split('.')[-1] == 'npy' and not ts_from_cloud:
                logger.loadNpyBuffer(replay_buffer, load_buffer, load_n)
            elif load_buffer.split('.')[-1] == 'npy' and ts_from_cloud:
                data = NpyBuffer(env_config, env, load_buffer, replay_buffer, resample=True, sim_n=simulate_n, sigma=sigma, data_balancing=data_balancing, sim_type=sim_type, no_bar=no_bar, load_n=load_n)
                data.addData()
            else:
                logger.loadBuffer(replay_buffer, load_buffer, load_n)

        elif planner_episode > 0:
            if simulate_n > 0:
                planner_num_process = 1
            else:
                planner_num_process = num_processes
            j = 0
            states, obs = planner_envs.reset()
            s = 0
            if not no_bar:
                planner_bar = tqdm(total=planner_episode, leave=True)
            local_transitions = [[] for _ in range(planner_num_process)]

            simulate_buffer = [[] for _ in range(planner_num_process)]
            extra_aug_buffer = [[] for _ in range(planner_num_process)]
            while j < planner_episode:
                plan_actions = planner_envs.getNextAction()
                planner_actions_star_idx, planner_actions_star = agent.getActionFromPlan(plan_actions)
                states_, obs_, rewards, dones = planner_envs.step(planner_actions_star, auto_reset=True)
                for i in range(planner_num_process):
                    transition = ExpertTransition(states[i].numpy(), obs[i].numpy().astype(np.float32), planner_actions_star_idx[i].numpy(),
                                                rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), dones[i].numpy(),
                                                np.array(100), np.array(1))
                    local_transitions[i].append(transition)
                    if len(local_transitions[i]) >=3 and ("bc" in alg) and (simulate_n>0):

                        f1 = planner_envs.canSimulate()
                        if not local_transitions[i][-2].state:
                            if sim_type == "breadth":
                                for _ in range(simulate_n):
                                    flag=0
                                    planner_envs.resetSimPose()
                                    # sigma = 0.2
                                    new_transition, flag = transitionSimulateSim(local_transitions[i], agent, planner_envs, sigma, i, planner_num_process)
                                    if flag == 1:
                                        simulate_buffer[i].append(new_transition)
                                    else:
                                        extra_aug_buffer[i].append(transition)
                            
                            elif sim_type == "depth":
                                planner_envs.resetSimPose()
                                for _ in range(simulate_n):
                                    flag=0
                                    
                                    # sigma = 0.2
                                    new_transition, flag = transitionSimulateSim(local_transitions[i], agent, planner_envs, sigma, i, planner_num_process)
                                    if flag == 1:
                                        simulate_buffer[i].append(new_transition)
                                    else:
                                        extra_aug_buffer[i].append(transition)

                            elif sim_type == "hybrid":
                                for _ in range(simulate_n):
                                    planner_envs.resetSimPose()
                                    for _ in range(simulate_n):
                                        flag=0
                                        # sigma = 0.2
                                        new_transition, flag = transitionSimulateSim(local_transitions[i], agent, planner_envs, sigma, i, planner_num_process)
                                        if flag == 1:
                                            simulate_buffer[i].append(new_transition)
                                        else:
                                            extra_aug_buffer[i].append(transition)
                        else:
                            extra_aug_buffer[i].append(transition)


                states = copy.copy(states_)
                obs = copy.copy(obs_)

                for i in range(planner_num_process):
                    if dones[i] and rewards[i]:
                        try:
                            logger.saveInitExpertImg(local_transitions[i][0].obs, 'expert.png')
                            logger.saveInitExpertImg(simulate_buffer[i][0].obs, 'expert_ts.png')
                        except:
                            print("No available obs")

                        if simulate_n > 0 and len(simulate_buffer[i]) > 0:
                            local_transitions[i]+=simulate_buffer[i]

                        for idx in range(len(local_transitions[i])):
                            replay_buffer.add(local_transitions[i][idx])

                        if data_balancing == "True" and simulate_n > 0:
                            for t in extra_aug_buffer[i]:
                                replay_buffer.addOnlyAug(t, simulate_n)
                        
                        local_transitions[i] = []
                        simulate_buffer[i] = []
                        extra_aug_buffer[i] = []
                        j += 1
                        s += 1
                        if not no_bar:
                            planner_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s) / j if j != 0 else 0))
                            planner_bar.update(1)
                        if j == planner_episode:
                            break
                    elif dones[i]:
                        local_transitions[i] = []
            if not no_bar:
                planner_bar.close()


    if not no_bar:
        pbar = tqdm(total=max_train_step, position=0, leave=True)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    while logger.num_training_steps < max_train_step:
        train_step(agent, replay_buffer, logger, p_beta_schedule)

        if (time.time() - start_time)/3600 > time_limit:
            break

        if not no_bar:
            timer_final = time.time()
            description = 'Eval Reward:{:.03f}; Loss:{:.03f}; Time:{:.03f}'.format(
                logger.eval_success[-1] if len(logger.eval_success) > 0 else 0, float(logger.getCurrentLoss()),
                timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_training_steps-pbar.n)

        if logger.num_training_steps > 0 and eval_freq > 0 and logger.num_training_steps % eval_freq == 0:
            if eval_thread is not None:
                eval_thread.join()
            eval_agent.copyNetworksFrom(agent)
            eval_thread = threading.Thread(target=evaluate, args=(envs, eval_agent, logger))
            eval_thread.start()

        if logger.num_training_steps % save_freq == 0:
            saveModelAndInfo(logger, agent)
        if save_multi_freq > 0:
            saveModelAndInfo(logger, agent)

    if eval_thread is not None:
        eval_thread.join()
    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(args, envs, agent, replay_buffer)
    if logger.num_training_steps >= max_train_step:
        logger.saveResult()
    envs.close()
    print('training finished')
    if not no_bar:
        pbar.close()

if __name__ == '__main__':
    train()