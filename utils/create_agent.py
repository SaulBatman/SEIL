from utils.parameters import *
from agents.dqn_agent_fac import DQNAgentFac
from agents.dqn_agent_com import DQNAgentCom
from networks.cnn import CNNFac, CNNCom
from networks.equivariant import EquivariantCNNFac, EquivariantCNNFac2, EquivariantCNNFac3, EquivariantCNNCom, EquivariantCNNCom2

def createAgent():
    if load_sub is not None or load_model_pre is not None:
        initialize = False
    else:
        initialize = True
    if env in ['close_loop_block_picking']:
        n_p = 2
    elif env in ['close_loop_block_reaching']:
        n_p = 1
    else:
        raise NotImplementedError
    if not random_orientation:
        n_theta = 1
    else:
        n_theta = 3

    # setup agent
    if alg == 'dqn_fac':
        agent = DQNAgentFac(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p, n_theta=n_theta)
        if model == 'cnn':
            net = CNNFac(n_p=n_p, n_theta=n_theta).to(device)
        elif model == 'equi_1':
            net = EquivariantCNNFac(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
        elif model == 'equi_2':
            net = EquivariantCNNFac2(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
        elif model == 'equi_3':
            net = EquivariantCNNFac3(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(net)
    elif alg == 'dqn_com':
        agent = DQNAgentCom(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p, n_theta=n_theta)
        if model == 'cnn':
            net = CNNCom(n_p=n_p, n_theta=n_theta).to(device)
        elif model == 'equi_1':
            net = EquivariantCNNCom(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
        elif model == 'equi_2':
            net = EquivariantCNNCom2(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)

        else:
            raise NotImplementedError
        agent.initNetwork(net, initialize=initialize)
    else:
        raise NotImplementedError

    return agent