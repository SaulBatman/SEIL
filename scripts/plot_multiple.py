import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import pickle

def getRewardsSingle(rewards, window=1000):
    moving_avg = []
    i = window
    while i-window < len(rewards):
        moving_avg.append(np.average(rewards[i-window:i]))
        i += window

    moving_avg = np.array(moving_avg)
    return moving_avg

def plotLearningCurveAvg(rewards, window=1000, label='reward', color='b', shadow=True, ax=plt, legend=True, linestyle='-'):
    min_len = np.min(list(map(lambda x: len(x), rewards)))
    rewards = list(map(lambda x: x[:min_len], rewards))
    avg_rewards = np.mean(rewards, axis=0)
    # avg_rewards = np.concatenate(([0], avg_rewards))
    # std_rewards = np.std(rewards, axis=0)
    std_rewards = stats.sem(rewards, axis=0)
    # std_rewards = np.concatenate(([0], std_rewards))
    xs = np.arange(window, window * (avg_rewards.shape[0]+1), window)
    if shadow:
        ax.fill_between(xs, avg_rewards-std_rewards, avg_rewards+std_rewards, alpha=0.2, color=color)
    l = ax.plot(xs, avg_rewards, label=label, color=color, linestyle=linestyle, alpha=0.7)
    if legend:
        ax.legend(loc=4)
    return l

def plotEvalCurveAvg(rewards, freq=1000, label='reward', color='b', shadow=True, ax=plt, legend=True, linestyle='-'):
    min_len = np.min(list(map(lambda x: len(x), rewards)))
    rewards = list(map(lambda x: x[:min_len], rewards))
    avg_rewards = np.mean(rewards, axis=0)
    # avg_rewards = np.concatenate(([0], avg_rewards))
    # std_rewards = np.std(rewards, axis=0)
    std_rewards = stats.sem(rewards, axis=0)
    # std_rewards = np.concatenate(([0], std_rewards))
    xs = np.arange(freq, freq * (avg_rewards.shape[0]+1), freq)
    if shadow:
        ax.fill_between(xs, avg_rewards-std_rewards, avg_rewards+std_rewards, alpha=0.2, color=color)
    l = ax.plot(xs, avg_rewards, label=label, color=color, linestyle=linestyle, alpha=0.7)
    if legend:
        ax.legend(loc=4)
    return l

def plotEvalCurve(base, step=50000, use_default_cm=False, freq=1000):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'equi+bufferaug': 'b',
            'equi': 'b',
            'cnn+bufferaug': 'g',
            'cnn': 'g',
            'cnn+rad': 'r',
            'cnn+drq': 'purple',
            'cnn+curl': 'orange',
            'curl': 'orange',

            'equi_both': 'b',
            'equi_actor': 'r',
            'equi_critic': 'purple',
            'cnn_both': 'g',

            'equi_rotaugall': 'b',
            'cnn_rotaugall': 'g',
            'rad_rotaugall': 'r',
            'drq_rotaugall': 'purple',
            'ferm_rotaugall': 'orange',

            'sacfd_equi': 'b',
            'sacfd_cnn': 'g',
            'sacfd_rad_cn': 'r',
            'sacfd_drq_cn': 'purple',
            'sacfd_rad': 'r',
            'sacfd_drq': 'purple',
            'sacfd_ferm': 'orange',

            'sac_equi': 'b',
            'sac_cnn': 'g',
            'sac_rad_crop': 'r',
            'sac_drq_shift': 'purple',
            'sac_curl': 'orange',

            'dqn_equi': 'b',
            'dqn_cnn': 'g',
            'dqn_rad_crop': 'r',
            'dqn_drq_shift': 'purple',
            'dqn_curl': 'orange',

            'C8': 'b',
            'C4': 'g',
            'C2': 'r',
        }

    linestyle_map = {
    }
    name_map = {
        'equi+bufferaug': 'Equivariant',
        'equi': 'Equivariant',
        'cnn+bufferaug': 'CNN',
        'cnn': 'CNN',
        'cnn+rad': 'RAD',
        'cnn+drq': 'DrQ',
        'cnn+curl': 'FERM',
        'curl': 'CURL',

        'equi_both': 'Equi Actor + Equi Critic',
        'equi_actor': 'Equi Actor + CNN Critic',
        'equi_critic': 'CNN Actor + Equi Critic',
        'cnn_both': 'CNN Actor + CNN Critic',

        'equi_rotaugall': 'Equi SACAux',
        'cnn_rotaugall': 'CNN SACAux',
        'rad_rotaugall': 'RAD Crop SACAux',
        'drq_rotaugall': 'DrQ Shift SACAux',
        'ferm_rotaugall': 'FERM SACAux',

        'sacfd_equi': 'Equi SACAux',
        'sacfd_cnn': 'CNN SACAux',
        'sacfd_rad_cn': 'RAD SO(2) SACAux',
        'sacfd_drq_cn': 'DrQ SO(2) SACAux',
        'sacfd_rad': 'RAD Crop SACAux',
        'sacfd_drq': 'DrQ Shift SACAux',
        'sacfd_ferm': 'FERM SACAux',

        'sac_equi': 'Equi SAC',
        'sac_cnn': 'CNN SAC',
        'sac_rad_crop': 'RAD Crop SAC',
        'sac_drq_shift': 'DrQ Shift SAC',
        'sac_curl': 'FERM',

        'dqn_equi': 'Equi DQN',
        'dqn_cnn': 'CNN DQN',
        'dqn_rad_crop': 'RAD Crop DQN',
        'dqn_drq_shift': 'DrQ Shift DQN',
        'dqn_curl': 'CURL DQN',
    }

    sequence = {
        'equi+bufferaug': '0',
        'equi': '0',
        'cnn+bufferaug': '1',
        'cnn': '1',
        'cnn+rad': '2',
        'cnn+drq': '3',
        'cnn+curl': '4',
        'curl': '4',

        'equi_both': '0',
        'equi_actor': '1',
        'equi_critic': '2',
        'cnn_both': '3',

        'equi_rotaugall': '0',
        'cnn_rotaugall': '1',
        'rad_rotaugall': '2',
        'drq_rotaugall': '3',
        'ferm_rotaugall': '4',

        'sacfd_equi': '0',
        'sacfd_cnn': '1',
        'sacfd_rad_cn': '2',
        'sacfd_drq_cn': '3',
        'sacfd_rad': '2',
        'sacfd_drq': '3',
        'sacfd_ferm': '4',

        'sac_equi': '0',
        'sac_cnn': '1',
        'sac_rad_crop': '2',
        'sac_drq_shift': '3',
        'sac_curl': '4',

        'dqn_equi': '0',
        'dqn_cnn': '1',
        'dqn_rad_crop': '2',
        'dqn_drq_shift': '3',
        'dqn_curl': '4',

        'C8': '0',
        'C4': '1',
        'C2': '2',
    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/eval_success.npy'))
                # data = pickle.load(open(os.path.join(base, method, run, 'log_data.pkl'), 'rb'))
                # rewards = data['eval_eps_rewards']
                # r = [np.mean(x) for x in rewards[:-1]]
                rs.append(r[:step//freq])
            except Exception as e:
                print(e)
                continue

        plotEvalCurveAvg(rs, freq, label=name_map[method] if method in name_map else method,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of training steps')
    # if base.find('bbp') > -1:
    plt.ylabel('eval success rate')
    # plt.xlim((-100, step+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))
    # plt.ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'eval.png'), bbox_inches='tight',pad_inches = 0)

def plotEvalBarChart(base, step=50000, use_default_cm=False, freq=1000):
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
        }

    linestyle_map = {
    }
    name_map = {
    }

    sequence = {
    }

    names = []
    ys = []
    errs = []
    bar_colors = []
    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/eval_success.npy'))
                # data = pickle.load(open(os.path.join(base, method, run, 'log_data.pkl'), 'rb'))
                # rewards = data['eval_eps_rewards']
                # r = [np.mean(x) for x in rewards[:-1]]
                rs.append(np.max(r[:step//freq]))
            except Exception as e:
                print(e)
                continue

        # plotEvalCurveAvg(rs, freq, label=name_map[method] if method in name_map else method,
        #                  color=color_map[method] if method in color_map else colors[i],
        #                  linestyle=linestyle_map[method] if method in linestyle_map else '-')
        # plt.bar(i, np.mean(rs), yerr=stats.sem(rs), color=color_map[method] if method in color_map else colors[i], tick_label=name_map[method] if method in name_map else method)
        names.append(name_map[method] if method in name_map else method)
        ys.append(np.mean(rs))
        errs.append(stats.sem(rs))
        bar_colors.append(color_map[method] if method in color_map else colors[i])
        i += 1

    for j in range(i):
        print('{}: {:.1f}+{:.1f}'.format(names[j], ys[j]*100, errs[j]*100))

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(2*i, 5))
    plt.bar(np.arange(i), ys, yerr=errs, color=bar_colors, width=0.3, ecolor='black', capsize=5)
    plt.xticks(np.arange(i), names)
    plt.ylabel('eval success rate')
    plt.yticks(np.arange(0., 1.05, 0.1))
    plt.xlim((-0.5, i-0.5))
    plt.grid(axis='x')
    # plt.tight_layout()
    plt.savefig(os.path.join(base, 'eval_bar.png'), bbox_inches='tight',pad_inches = 0)
    plt.close()

def plotStepRewardCurve(base, step=50000, use_default_cm=False, freq=1000, file_name='step_reward'):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'dpos=0.05, drot=0.25pi': 'b',
            'dpos=0.05, drot=0.125pi': 'g',
            'dpos=0.03, drot=0.125pi': 'r',
            'dpos=0.1, drot=0.25pi': 'purple',

            'ban0': 'g',
            'ban2': 'r',
            'ban4': 'b',
            'ban8': 'purple',
            'ban16': 'orange',

            'C4': 'g',
            'C8': 'r',
            'D4': 'b',
            'D8': 'purple',

            '0': 'r',
            '10': 'g',
            '20': 'b',
            '40': 'purple',

            'sac+ban4': 'b',
            'sac+rot rad': 'g',
            'sac+rot rad+ban4': 'r',
            'sac+ban0': 'purple',

            'sac+aux+ban0': 'g',
            'sac+aux+ban4': 'r',

            'equi sac': 'b',
            'ferm': 'g'
        }

    linestyle_map = {
    }
    name_map = {
        'ban0': 'buffer aug 0',
        'ban2': 'buffer aug 2',
        'ban4': 'buffer aug 4',
        'ban8': 'buffer aug 8',
        'ban16': 'buffer aug 16',

        'sac+ban4': 'SAC + buffer aug',
        'sac+rot rad': 'SAC + rot RAD',
        'sac+rot rad+ban4': 'SAC + rot RAD + buffer aug',
        'sac+ban0': 'SAC',

        'sac+aux+ban4': 'SAC + aux loss + buffer aug',
        'sac+aux+ban0': 'SAC + aux loss',

        'sac': 'SAC',
        'sacfd': 'SACfD',

        'sac+crop rad': 'SAC + crop RAD',

        'equi sac': 'Equivariant SAC',
        'ferm': 'FERM'
    }

    sequence = {
        'ban0': '0',
        'ban2': '1',
        'ban4': '2',
        'ban8': '3',
        'ban16': '4',

        'sac+ban0': '0',
        'sac+ban4': '1',
        'sac+aux+ban0': '2',
        'sac+aux+ban4': '3',
    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                step_reward = np.load(os.path.join(base, method, run, 'info/{}.npy'.format(file_name)))
                r = []
                for k in range(1, step+1, freq):
                    window_rewards = step_reward[(k <= step_reward[:, 0]) * (step_reward[:, 0] < k + freq)][:, 1]
                    if window_rewards.shape[0] > 0:
                        r.append(window_rewards.mean())
                    else:
                        break
                    # r.append(step_reward[(i <= step_reward[:, 0]) * (step_reward[:, 0] < i + freq)][:, 1].mean())
                rs.append(r)
            except Exception as e:
                print(e)
                continue

        plotEvalCurveAvg(rs, freq, label=name_map[method] if method in name_map else method,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of training steps')
    # if base.find('bbp') > -1:
    plt.ylabel('discounted reward')
    # plt.xlim((-100, step+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))
    # plt.ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'step_reward.png'), bbox_inches='tight',pad_inches = 0)

def plotStepSRCurve(base, step=50000, use_default_cm=False, freq=1000, file_name='step_success_rate'):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
        }

    linestyle_map = {
    }
    name_map = {
    }

    sequence = {
    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                step_reward = np.load(os.path.join(base, method, run, 'info/{}.npy'.format(file_name)))
                r = []
                for k in range(1, step+1, freq):
                    window_rewards = step_reward[(k <= step_reward[:, 0]) * (step_reward[:, 0] < k + freq)][:, 1]
                    if window_rewards.shape[0] > 0:
                        r.append(window_rewards.mean())
                    else:
                        break
                    # r.append(step_reward[(i <= step_reward[:, 0]) * (step_reward[:, 0] < i + freq)][:, 1].mean())
                rs.append(r)
            except Exception as e:
                print(e)
                continue

        plotEvalCurveAvg(rs, freq, label=name_map[method] if method in name_map else method,
                         color=color_map[method] if method in color_map else colors[i],
                         linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of training steps')
    # if base.find('bbp') > -1:
    plt.ylabel('success rate')
    # plt.xlim((-100, step+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))
    # plt.ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'step_reward.png'), bbox_inches='tight',pad_inches = 0)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def plotLearningCurve(base, ep=50000, use_default_cm=False, window=1000):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'equi+bufferaug': 'b',
            'cnn+bufferaug': 'g',
            'cnn+rad': 'r',
            'cnn+drq': 'purple',
            'cnn+curl': 'orange',
        }

    linestyle_map = {

    }
    name_map = {
        'equi+bufferaug': 'Equivariant',
        'cnn+bufferaug': 'CNN',
        'cnn+rad': 'RAD',
        'cnn+drq': 'DrQ',
        'cnn+curl': 'FERM',
    }

    sequence = {
        'equi+equi': '0',
        'cnn+cnn': '1',
        'cnn+cnn+aug': '2',
        'equi_fcn_asr': '3',
        'tp': '4',

        'equi_fcn': '0',
        'fcn_si': '1',
        'fcn_si_aug': '2',
        'fcn': '3',

        'equi+deictic': '2',
        'cnn+deictic': '3',

        'q1_equi+q2_equi': '0',
        'q1_equi+q2_cnn': '1',
        'q1_cnn+q2_equi': '2',
        'q1_cnn+q2_cnn': '3',

        'q1_equi+q2_deictic': '0.5',
        'q1_cnn+q2_deictic': '4',

        'equi_fcn_': '1',

        '5l_equi_equi': '0',
        '5l_equi_deictic': '1',
        '5l_equi_cnn': '2',
        '5l_cnn_equi': '3',
        '5l_cnn_deictic': '4',
        '5l_cnn_cnn': '5',

    }

    # house1-4
    # plt.plot([0, 100000], [0.974, 0.974], label='expert', color='pink')
    # plt.axvline(x=10000, color='black', linestyle='--')

    # house1-5
    # plt.plot([0, 50000], [0.974, 0.974], label='expert', color='pink')
    # 0.004 pos noise
    # plt.plot([0, 50000], [0.859, 0.859], label='expert', color='pink')

    # house1-6 0.941

    # house2
    # plt.plot([0, 50000], [0.979, 0.979], label='expert', color='pink')
    # plt.axvline(x=20000, color='black', linestyle='--')

    # house3
    # plt.plot([0, 50000], [0.983, 0.983], label='expert', color='pink')
    # plt.plot([0, 50000], [0.911, 0.911], label='expert', color='pink')
    # 0.996
    # 0.911 - 0.01

    # house4
    # plt.plot([0, 50000], [0.948, 0.948], label='expert', color='pink')
    # plt.plot([0, 50000], [0.862, 0.862], label='expert', color='pink')
    # 0.875 - 0.006
    # 0.862 - 0.007 *
    # stack
    # plt.plot([0, 100000], [0.989, 0.989], label='expert', color='pink')
    # plt.axvline(x=10000, color='black', linestyle='--')

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/rewards.npy'))
                if method.find('BC') >= 0 or method.find('tp') >= 0:
                    rs.append(r[-window:].mean())
                else:
                    rs.append(getRewardsSingle(r[:ep], window=window))
            except Exception as e:
                print(e)
                continue

        if method.find('BC') >= 0 or method.find('tp') >= 0:
            avg_rewards = np.mean(rs, axis=0)
            std_rewards = stats.sem(rs, axis=0)

            plt.plot([0, ep], [avg_rewards, avg_rewards],
                     label=name_map[method] if method in name_map else method,
                     color=color_map[method] if method in color_map else colors[i])
            plt.fill_between([0, ep], avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, color=color_map[method] if method in color_map else colors[i])
        else:
            plotLearningCurveAvg(rs, window, label=name_map[method] if method in name_map else method,
                                 color=color_map[method] if method in color_map else colors[i],
                                 linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of episodes')
    # if base.find('bbp') > -1:
    plt.ylabel('discounted reward')

    # plt.xlim((-100, ep+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'plot.png'), bbox_inches='tight',pad_inches = 0)

def plotSuccessRate(base, ep=50000, use_default_cm=False, window=1000):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'equi+bufferaug': 'b',
            'cnn+bufferaug': 'g',
            'cnn+rad': 'r',
            'cnn+drq': 'purple',
            'cnn+curl': 'orange',
        }

    linestyle_map = {
    }
    name_map = {
    }

    sequence = {
    }

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/success_rate.npy'))
                if method.find('BC') >= 0 or method.find('tp') >= 0:
                    rs.append(r[-window:].mean())
                else:
                    rs.append(getRewardsSingle(r[:ep], window=window))
            except Exception as e:
                print(e)
                continue

        if method.find('BC') >= 0 or method.find('tp') >= 0:
            avg_rewards = np.mean(rs, axis=0)
            std_rewards = stats.sem(rs, axis=0)

            plt.plot([0, ep], [avg_rewards, avg_rewards],
                     label=name_map[method] if method in name_map else method,
                     color=color_map[method] if method in color_map else colors[i])
            plt.fill_between([0, ep], avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, color=color_map[method] if method in color_map else colors[i])
        else:
            plotLearningCurveAvg(rs, window, label=name_map[method] if method in name_map else method,
                                 color=color_map[method] if method in color_map else colors[i],
                                 linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of episodes')
    # if base.find('bbp') > -1:
    plt.ylabel('success rate')

    # plt.xlim((-100, ep+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'sr.png'), bbox_inches='tight',pad_inches = 0)

def showPerformance(base):
    methods = sorted(filter(lambda x: x[0] != '.', get_immediate_subdirectories(base)))
    for method in methods:
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/rewards.npy'))
                rs.append(r[-1000:].mean())
            except Exception as e:
                print(e)
        print('{}: {:.3f}'.format(method, np.mean(rs)))


def plotTDErrors():
    plt.style.use('ggplot')
    colors = "bgrycmkw"
    method_map = {
        'ADET': 'm',
        'ADET+Q*': 'g',
        'DAGGER': 'k',
        'DQN': 'c',
        'DQN+guided': 'y',
        'DQN+Q*': 'b',
        'DQN+Q*+guided': 'r',
        "DQfD": 'chocolate',
        "DQfD+Q*": 'grey'
    }
    i = 0

    base = '/media/dian/hdd/unet/perlin'
    for method in sorted(get_immediate_subdirectories(base)):
        rs = []
        if method[0] == '.' or method == 'DAGGER' or method == 'DQN':
            continue
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/td_errors.npy'))
                rs.append(getRewardsSingle(r[:120000], window=1000))
            except Exception as e:
                continue
        if method in method_map:
            plotLearningCurveAvg(rs, 1000, label=method, color=method_map[method])
        else:
            plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        # plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        i += 1

    plt.legend(loc=0)
    plt.xlabel('number of training steps')
    plt.ylabel('TD error')
    plt.yscale('log')
    # plt.ylim((0.8, 0.93))
    plt.show()

def plotLoss(base, step):
    plt.style.use('ggplot')
    colors = "bgrycmkw"
    method_map = {
        'ADET': 'm',
        'ADET+Q*': 'g',
        'DAGGER': 'k',
        'DQN': 'c',
        'DQN+guided': 'y',
        'DQN+Q*': 'b',
        'DQN+Q*+guided': 'r',
        "DQfD": 'chocolate',
        "DQfD+Q*": 'grey'
    }
    i = 0

    for method in sorted(get_immediate_subdirectories(base)):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/losses.npy'))[:, 1]
                rs.append(getRewardsSingle(r[:step], window=1000))
            except Exception as e:
                continue
        if method in method_map:
            plotLearningCurveAvg(rs, 1000, label=method, color=method_map[method])
        else:
            plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        # plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        i += 1

    plt.legend(loc=0)
    plt.xlabel('number of training steps')
    plt.ylabel('loss')
    plt.yscale('log')
    # plt.ylim((0.8, 0.93))
    plt.tight_layout()
    plt.savefig(os.path.join(base, 'plot.png'), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    base = '/media/dian/hdd/mrun_results/ibc/0417_do_planner20_b128'
    # plotLearningCurve(base, 1000, window=20)
    # plotSuccessRate(base, 3000, window=100)
    plotEvalCurve(base, 20000, freq=1000)
    plotEvalBarChart(base, 20000, freq=1000)
    # showPerformance(base)
    # plotLoss(base, 30000)

    # plotStepSRCurve(base, 10000, freq=500, file_name='step_success_rate')
    # plotStepRewardCurve(base, 10000, freq=200)

