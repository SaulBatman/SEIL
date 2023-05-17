from utils.parameters import *
from networks.cnn import Actor

from agents.bc_continuous import BehaviorCloningContinuous
from agents.ibc import ImplicitBehaviorCloning
from networks.equivariant_sac_net import EquivariantPolicy, EquivariantPolicyDihedral
from networks.cnn import CNNEBM


def createAgent(test=False):
    print('initializing agent')
    if view_type == 'camera_fix_rgbd':
        obs_channel = 7
    else:
        obs_channel = 2
    if load_sub is not None or load_model_pre is not None or test:
        initialize = False
    else:
        initialize = True
    n_p = 2
    if not random_orientation:
        n_theta = 1
    else:
        n_theta = 3

    # setup agent
    if alg in ['bc_con']:
        agent = BehaviorCloningContinuous(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                                          n_a=len(action_sequence))

        if model == 'equi':
            policy = EquivariantPolicy((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
        elif model == 'equi_d':
            policy = EquivariantPolicyDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
        elif model == 'cnn':
            policy = Actor(len(action_sequence)).to(device)

        else:
            raise NotImplementedError
        agent.initNetwork(policy)
        
    elif alg in ['bc_implicit']:
        agent = ImplicitBehaviorCloning(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                                          n_a=len(action_sequence), ibc_ts=ibc_ts, ibc_is=ibc_is)
        if model == 'cnn_ssm':
            policy = CNNEBM(len(action_sequence), reducer='spatial_softmax').to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(policy)

    else:
        raise NotImplementedError
    agent.aug = aug
    agent.aug_type = aug_type
    print('initialized agent')
    return agent