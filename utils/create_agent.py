from utils.parameters import *
from networks.cnn import Actor

from agents.bc_continuous import BehaviorCloningContinuous

from networks.equivariant_sac_net import EquivariantPolicy, EquivariantPolicyDihedral
from networks.equivariant_sac_net import EquivariantPolicySO2, EquivariantPolicyO2

from agents.ibc import ImplicitBehaviorCloning
from networks.cnn import CNNEBM
from networks.equivariant_sac_net import EquivariantEBMDihedral, EquivariantEBMDihedralSpatialSoftmax

from agents.bc_fac import BCFac
from agents.ibc_fac import ImplicitBehaviorCloningFactored
from networks.equivariant_sac_net import EquivariantEBMDihedralFac, EquivariantEBMDihedralFacSepEnc

from agents.ibc_fac_all import ImplicitBehaviorCloningFactoredAll
from networks.equivariant_sac_net import EquivariantEBMDihedralFacAll

from networks.cnn import CNNMSE

from networks.equivariant_sac_net import EquivariantPolicyDihedralSpatialSoftmax, EquivariantPolicyDihedralSpatialSoftmax1

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
        elif model == 'equi_d_ssm':
            policy = EquivariantPolicyDihedralSpatialSoftmax((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
        elif model == 'equi_d_ssm_1':
            policy = EquivariantPolicyDihedralSpatialSoftmax1((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
        elif model == 'equi_enc_2':
            policy = EquivariantPolicy((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, enc_id=2).to(device)
        elif model == 'cnn':
            policy = Actor(len(action_sequence)).to(device)
        elif model == 'cnn_maxpool':
            policy = CNNMSE(len(action_sequence), reducer='maxpool').to(device)
        elif model == 'cnn_pro_maxpool':
            policy = CNNMSE(len(action_sequence), reducer='progressive_maxpool').to(device)
        elif model == 'cnn_ssm':
            policy = CNNMSE(len(action_sequence), reducer='spatial_softmax').to(device)
        elif model == 'equi_so2':
            policy = EquivariantPolicySO2((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
        elif model == 'equi_o2':
            policy = EquivariantPolicyO2((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
        elif model == 'equi_d_k3':
            policy = EquivariantPolicyDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, kernel_size=3).to(device)
        elif model == 'equi_d_k5':
            policy = EquivariantPolicyDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, kernel_size=5).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(policy)

    elif alg in ['bc_implicit']:
        agent = ImplicitBehaviorCloning(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                                          n_a=len(action_sequence), ibc_ts=ibc_ts, ibc_is=ibc_is)
        if model == 'cnn_maxpool':
            policy = CNNEBM(len(action_sequence), reducer='maxpool').to(device)
        elif model == 'cnn_ssm':
            policy = CNNEBM(len(action_sequence), reducer='spatial_softmax').to(device)
        elif model == 'equi_d':
            policy = EquivariantEBMDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
        elif model == 'equi_d_ssm':
            policy = EquivariantEBMDihedralSpatialSoftmax((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(policy)

    elif alg in ['bc_implicit_fac']:
        agent = ImplicitBehaviorCloningFactored(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                                                n_a=len(action_sequence), ibc_ts=ibc_ts, ibc_is=ibc_is)
        if model == 'equi_d':
            policy = EquivariantEBMDihedralFac((obs_channel, crop_size, crop_size), len(action_sequence),
                                               n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
        elif model == 'equi_d_sep':
            policy = EquivariantEBMDihedralFacSepEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                     n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(policy)

    elif alg in ['bc_implicit_fac_all']:
        agent = ImplicitBehaviorCloningFactoredAll(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                                                   n_a=len(action_sequence), ibc_ts=ibc_ts, ibc_is=ibc_is)
        if model == 'equi_d':
            policy = EquivariantEBMDihedralFacAll((obs_channel, crop_size, crop_size), len(action_sequence),
                                                  n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(policy)

    else:
        raise NotImplementedError
    agent.aug = aug
    agent.aug_type = aug_type
    print('initialized agent')
    return agent