import os
import sys
import time
import copy
import collections
from tqdm import tqdm

from utils.create_agent import createAgent
import threading

from utils.torch_utils import ExpertTransition
from utils.debug import visualizeTransitionTS
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import transformations
import scipy
from scipy.ndimage import rotate
from utils.debug import visualizeTraj

def transitionSimulateSim(local_transition, agent, envs, sigma, i, planner_num_process):
    
    #num_processes=1 # only support single process now

    flag = 1
    sim_startpoint = -3
    sim_obs0 = local_transition[sim_startpoint].obs
    sim_states0 = local_transition[sim_startpoint].state
    sim_actions0_star_idx = local_transition[sim_startpoint].action
    sim_states1, sim_obs1 = local_transition[sim_startpoint+1].state, local_transition[sim_startpoint+1].obs
    sim_actions1_star_idx = local_transition[sim_startpoint+1].action
    sim_steps_lefts = local_transition[sim_startpoint+1].step_left
    sim_states2, sim_obs2 = local_transition[sim_startpoint+2].state, local_transition[sim_startpoint+2].obs
    sim_rewards2, sim_dones2 = local_transition[sim_startpoint+2].reward, local_transition[sim_startpoint+2].done
    if sim_dones2:
        flag = 0
        return None, flag
    sim_actions1_star_idx_inv, sim_actions1_star_inv = agent.getInvBCActions(sim_actions0_star_idx, sim_actions1_star_idx, sigma, "gaussian")
    temp = np.zeros([planner_num_process, agent.n_a])
    temp[i, :] = sim_actions1_star_inv
    sim_states_new, sim_obs_new, _, _, sim_flag = envs.simulate(torch.from_numpy(temp))

    sim_actions_new_star_idx,  sim_actions_new_star= agent.getSimBCActions(sim_actions1_star_idx_inv, torch.tensor(sim_actions1_star_idx[0]))
    
    sim_obs = [sim_obs0, sim_obs1, sim_obs2, sim_obs_new]
    scaled_sim_action, unscales_sim_action = agent.decodeSingleActions(*[torch.tensor(sim_actions1_star_idx)[i] for i in range(5)])
    actions = [unscales_sim_action, sim_actions_new_star[0]]
    # fig = visualizeTransitionTS(sim_obs, actions)
    # fig.clf()
    # sim_obs = [sim_obs1, sim_obs2]
    # actions = [sim_actions1_star_idx]
    # fig = visualizeTransition(agent, sim_obs, actions)

    is_expert = 1
    transition = ExpertTransition(sim_states_new[i].numpy(), sim_obs_new[i].numpy(), sim_actions_new_star_idx[0].numpy(),
                                sim_rewards2, sim_states2, sim_obs2, sim_dones2,
                                sim_steps_lefts, np.array(is_expert))
    # if obs_type == 'pixel':
    #     transition = normalizeTransition(transition)
    if sim_flag == False:
        flag = 0
    return transition, flag




class NpyBuffer():
    def __init__(self, config, agent, path, buffer, resample=True, sim_n=4, sigma=0.4, data_balancing=True, sim_type='breadth', no_bar = "False", load_n=9999):
        # self.view_type = 'render_center'

        self.load = np.load(path, allow_pickle=True)
        self.agent = agent
        self.resample = resample  # all obs come from cloud reprojection
        self.no_bar = no_bar
        self.view_type = config['view_type']
        self.buffer = buffer
        self.load_n = load_n
        # self.step_idx = 0
        # self.epi_idx = 0

        self.data_balancing = data_balancing
        self.sim_type = sim_type
        self.sim_n = sim_n
        self.sigma = sigma

        self.current_pos = None
        self.simulate_pos = None
        self.cloud = None
        self.is_holding = None
        self.t = None

        
        self.desk_center = (-0.527, -0.005)
        self.z_min = -0.080
        
        ws_x = 0.4
        ws_y = 0.4
       
        self.desk_workspace = np.asarray([[self.desk_center[0] - ws_x / 2, self.desk_center[0] + ws_x / 2],
                                          [self.desk_center[1] - ws_y/2, self.desk_center[1] + ws_y/2],
                                          [self.z_min, self.z_min+0.2]])
        self.view_scale = config['view_scale'] 
        self.heightmap_size = config['obs_size']


        self.obs_size_m = ws_x * self.view_scale
        self.simulate_z_threshold = self.desk_workspace[2][0] + 0.07
        

    def getTransition(self):
        step_total = len(self.load[self.epi_idx])

        if self.step_idx < step_total:
            output = self.load[self.epi_idx][self.step_idx]
        else:
            self.epi_idx



        self.epi_idx += 1

    def interpolate(self, depth):
        """
        Fill nans in depth image
        """
        # a boolean array of (width, height) which False where there are missing values and True where there are valid (non-missing) values
        mask = np.logical_not(np.isnan(depth))
        # array of (number of points, 2) containing the x,y coordinates of the valid values only
        xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

        # the valid values in the first, second, third color channel,  as 1D arrays (in the same order as their coordinates in xym)
        data0 = np.ravel(depth[:, :][mask])

        # three separate interpolators for the separate color channels
        interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)

        # interpolate the whole image, one color channel at a time
        result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

        return result0

    def getProjectImg(self, gripper_pos=(-0.5, 0, 0.1)):
        """
        return orthographic projection depth img from self.cloud
        obs_size_m: img coverage size in meters
        heightmap_size: img pixel size
        gripper_pos: the pos of the camera
        return depth image
        """
        cloud = np.copy(self.cloud)
        cloud = cloud[(cloud[:, 2] < max(gripper_pos[2], self.z_min + 0.05))]
        view_matrix = transformations.euler_matrix(0, np.pi, 0).dot(np.eye(4))
        # view_matrix = np.eye(4)
        view_matrix[:3, 3] = [gripper_pos[0], -gripper_pos[1], gripper_pos[2]]
        view_matrix = transformations.euler_matrix(0, 0, -np.pi/2).dot(view_matrix)
        augment = np.ones((1, cloud.shape[0]))
        pts = np.concatenate((cloud.T, augment), axis=0)
        projection_matrix = np.array([
            [1 / (self.obs_size_m / 2), 0, 0, 0],
            [0, 1 / (self.obs_size_m / 2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        tran_world_pix = np.matmul(projection_matrix, view_matrix)
        pts = np.matmul(tran_world_pix, pts)
        # pts[1] = -pts[1]
        pts[0] = (pts[0] + 1) * self.heightmap_size / 2
        pts[1] = (pts[1] + 1) * self.heightmap_size / 2

        pts[0] = np.round_(pts[0])
        pts[1] = np.round_(pts[1])
        mask = (pts[0] >= 0) * (pts[0] < self.heightmap_size) * (pts[1] > 0) * (pts[1] < self.heightmap_size)
        pts = pts[:, mask]
        # dense pixel index
        mix_xy = (pts[1].astype(int) * self.heightmap_size + pts[0].astype(int))
        # lexsort point cloud first on dense pixel index, then on z value
        ind = np.lexsort(np.stack((pts[2], mix_xy)))
        # bin count the points that belongs to each pixel
        bincount = np.bincount(mix_xy)
        # cumulative sum of the bin count. the result indicates the cumulative sum of number of points for all previous pixels
        cumsum = np.cumsum(bincount)
        # rolling the cumsum gives the ind of the first point that belongs to each pixel.
        # because of the lexsort, the first point has the smallest z value
        cumsum = np.roll(cumsum, 1)
        cumsum[0] = bincount[0]
        cumsum[cumsum == np.roll(cumsum, -1)] = 0
        # pad for unobserved pixels
        cumsum = np.concatenate((cumsum, -1 * np.ones(self.heightmap_size * self.heightmap_size - cumsum.shape[0]))).astype(int)

        depth = pts[2][ind][cumsum]
        depth[cumsum == 0] = np.nan
        depth = depth.reshape(self.heightmap_size, self.heightmap_size)
        # fill nans
        depth = self.interpolate(depth)
        # mask = np.isnan(depth)
        # depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])
        # imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        # imputer_depth = imputer.fit_transform(depth)
        # if imputer_depth.shape != depth.shape:
        #     mask = np.isnan(depth)
        #     depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])
        # else:
        #     depth = imputer_depth
        return depth

    def getHeightmapReconstruct(self, gripper_pos=None, obs_size_m=None):
        # get orthographic projection image
        if obs_size_m is None:
            obs_size_m = self.obs_size_m
        # get img from camera
        obss = []
        for i in range(1):
            obss.append(self.getProjectImg(gripper_pos))
        obs = np.median(obss, axis=0)
        obs = scipy.ndimage.median_filter(obs, 2)
        return obs

    def getObs(self, gripper_width, is_holding, current_pos):
        self.heightmap = self.getHeightmapReconstruct(gripper_pos=current_pos[:3])
        self.heightmap = np.clip(self.heightmap, -0.5, 0.5)
        gripper_img = self.getGripperImg(gripper_state=gripper_width, gripper_rz=current_pos[3]) 
        heightmap = np.copy(self.heightmap)
        heightmap[gripper_img.astype(bool)] = 0
        heightmap = heightmap.reshape([1, self.heightmap_size, self.heightmap_size])
        img = heightmap
        # img = np.stack((self.heightmap, gripper_img))
        is_holding = is_holding
        return is_holding, img

    def getGripperImg(self, gripper_state=None, gripper_rz=None):

        im = np.zeros((self.heightmap_size, self.heightmap_size))
        len = 0.098/self.obs_size_m * self.heightmap_size
        d = int(len * gripper_state) 
        l = int(0.02/self.obs_size_m * self.heightmap_size/2)
        w = int(0.015/self.obs_size_m * self.heightmap_size/2)
        im[self.heightmap_size//2-d//2-w:self.heightmap_size//2-d//2+w, self.heightmap_size//2-l:self.heightmap_size//2+l] = 1
        im[self.heightmap_size//2+d//2-w:self.heightmap_size//2+d//2+w, self.heightmap_size//2-l:self.heightmap_size//2+l] = 1
        im = rotate(im, np.rad2deg(gripper_rz), reshape=False, order=0)
        return im

    # def _getHeightmap(self, obs_size_m, heightmap_size):
    #     if self.view_type == 'render_center':
    #         if self.points.shape[0] == 0:
    #             self.points = self.points[self.points[:, 2] <= max(self.gripper_pos[2]-0.01, 0.05)]
    #         points = np.copy(self.points)
    #         points = points[points[:, 2] <= max(self.gripper_pos[2]-0.01, 0.05)]
    #         # self.points = self.points[(self.workspace[0, 0] <= self.points[:, 0]) * (self.points[:, 0] <= self.workspace[0, 1])]
    #         # self.points = self.points[(self.workspace[1, 0] <= self.points[:, 1]) * (self.points[:, 1] <= self.workspace[1, 1])]

    #         render_cam_target_pos = [self.gripper_pos[0], self.gripper_pos[1], 0]
    #         # render_cam_up_vector = [-1, 0, 0]
    #         T = transformations.euler_matrix(0, 0, self.gripper_pos[3])
    #         render_cam_up_vector = T.dot(np.array([-1, 0, 0, 1]))[:3]


    #         render_cam_pos1 = [self.gripper_pos[0], self.gripper_pos[1], self.gripper_pos[2]]
    #         # t0 = time.time()
    #         depth = self.projectDepth(points, heightmap_size, render_cam_pos1, render_cam_up_vector,
    #                                 render_cam_target_pos, obs_size_m)
    #         # depth = sk_transform.rotate(depth, np.rad2deg(gripper_rz))
    #         return depth
    #     else:
    #         raise NotImplementedError

    def canSimulate(self):
        return not self.is_holding and self.simulate_pos[2] > self.simulate_z_threshold

    def resetSimPose(self):
        self.simulate_pos = self.current_pos

    def simulate(self, action):
        flag = True
        p, dx, dy, dz, r = action[0]
        dtheta = r
        # pos = list(self.robot._getEndEffectorPosition())
        # gripper_rz = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())[2]
        pos = self.simulate_pos[:3]
        gripper_rz = self.simulate_pos[3]
        pos[0] += dx
        pos[1] += dy
        pos[2] += dz
        temp1, temp2, temp3 = pos[0].copy(), pos[1].copy(), pos[2].copy()
        pos[0] = np.clip(pos[0], self.desk_workspace[0, 0], self.desk_workspace[0, 1])
        pos[1] = np.clip(pos[1], self.desk_workspace[1, 0], self.desk_workspace[1, 1])
        pos[2] = np.clip(pos[2], self.simulate_z_threshold, self.desk_workspace[2, 1])
        if (temp1!=pos[0]) or (temp2!=pos[1]) or (temp3!=pos[2]):
            flag=False
        gripper_rz += dtheta
        self.simulate_pos = pos
        # obs = self.renderer.getTopDownDepth(self.obs_size_m, self.heightmap_size, pos, 0)
        # obs = self.getProjectImg(gripper_pos=self.simulate_pos[:3])
        gripper_img = self.getGripperImg(p, gripper_rz)
        is_holding, obs = self.getObs(self.t[4], self.is_holding, self.current_pos)
        if self.view_type.find('height') > -1:
            obs[0][gripper_img == 1] = self.simulate_pos[2]
        else:
            obs[0][gripper_img == 1] = 0
        # gripper_img = gripper_img.reshape([1, self.heightmap_size, self.heightmap_size])
        # obs[gripper_img==1] = 0
        obs = obs.reshape([1, self.heightmap_size, self.heightmap_size])
        return False, obs, None, None, flag
        # return self.is_holding, None, obs, flag

    def addData(self):

        if not self.no_bar:
            N = len(self.load) if len(self.load) < self.load_n else self.load_n
            load_bar = tqdm(total=len(self.load), leave=True)
        tra_idx = 0
        for traj in self.load:
            tra_idx += 1
            self.local_transitions = []
            self.simulate_buffer = []
            self.extra_aug_buffer = [] # extra rotation aug for failed simulated transition
            i=0
            for t in traj:
                i+=1
                self.t = t
                self.cloud = t[9]
                self.current_pos = t[10]
                self.is_holding = t[0]
                if self.resample:
                    # obs = self.getProjectImg(gripper_pos=t[10][:3])
                    if len(self.local_transitions) > 0:
                        p = self.local_transitions[-1].action[0]
                    else:
                        p=1
                    is_holding, obs = self.getObs(p, self.is_holding, self.current_pos)
                    transition = ExpertTransition(is_holding, obs.astype(np.float32), 
                                                  t[2].astype(np.float32), t[3].astype(np.float32), 
                                                  np.array(False).astype(np.float32), t[5][1].astype(np.float32), 
                                                  t[6].astype(np.float32), t[7].astype(np.float32), 
                                                  t[8].astype(np.float32))
                else:
                    transition = ExpertTransition(t[0].astype(np.float32), t[1][1].astype(np.float32), 
                                                  t[2].astype(np.float32), t[3].astype(np.float32), 
                                                  t[4].astype(np.float32), t[5][1].astype(np.float32), 
                                                  t[6].astype(np.float32), t[7].astype(np.float32), 
                                                  t[8].astype(np.float32))
                
                self.local_transitions.append(transition)
                if (self.sim_n>0):
                    # f1 = self.canSimulate()
                    if i > 2 and i <= len(traj):
                        if not self.local_transitions[-2].state:
                            if self.sim_type == 'breadth':
                                for _ in range(self.sim_n):
                                    flag=0
                                    self.resetSimPose()
                                    new_transition, flag = self.transitionSimulateReal()
                                    if flag == 1:
                                        self.simulate_buffer.append(new_transition)
                                    else:
                                        self.extra_aug_buffer.append(new_transition)
                            
                            elif self.sim_type == 'depth':
                                for _ in range(self.sim_n):
                                    flag=0
                                    new_transition, flag = self.transitionSimulateReal()
                                    if flag == 1:
                                        self.simulate_buffer.append(new_transition)
                                    else:
                                        self.extra_aug_buffer.append(new_transition)

                            elif self.sim_type == "hybrid":
                                for _ in range(self.sim_n):
                                    self.resetSimPose()
                                    for _ in range(self.sim_n):
                                        flag=0
                                        # sigma = 0.2
                                        new_transition, flag = transitionSimulateSim()
                                        if flag == 1:
                                            self.simulate_buffer.append(new_transition)
                                        else:
                                            self.extra_aug_buffer.append(new_transition)

                    else:
                        self.extra_aug_buffer.append(transition)

            
            

            if self.sim_n > 0 and len(self.simulate_buffer) > 0:
                self.local_transitions+=self.simulate_buffer

            for t in self.local_transitions:
                self.buffer.add(t)

            if self.data_balancing == "True" and self.sim_n > 0:
                for t in self.extra_aug_buffer:
                    self.buffer.addOnlyAug(t, self.sim_n)
            # visualizeTraj(self.agent, self.local_transitions, vis_num=10)
            self.local_transitions = []
            self.simulate_buffer = []
            self.extra_aug_buffer = []
            # j += 1
            # s += 1

            if tra_idx == self.load_n:
                break
            if not self.no_bar:
                
                load_bar.set_description(f'{tra_idx}/{N}')
                load_bar.update(1)


    def transitionSimulateReal(self):
        # only support single process now
        flag = 1 # stop sign
        if len(self.local_transitions) >=3:
            sim_startpoint = -3
            sim_obs0 = self.local_transitions[sim_startpoint].obs
            sim_states0 = self.local_transitions[sim_startpoint].state
            sim_actions0_star_idx = self.local_transitions[sim_startpoint].action
            sim_states1, sim_obs1 = self.local_transitions[sim_startpoint+1].state, self.local_transitions[sim_startpoint+1].obs
            sim_actions1_star_idx = self.local_transitions[sim_startpoint+1].action
            sim_steps_lefts = self.local_transitions[sim_startpoint+1].step_left
            sim_states2, sim_obs2 = self.local_transitions[sim_startpoint+2].state, self.local_transitions[sim_startpoint+2].obs
            sim_rewards2, sim_dones2 = self.local_transitions[sim_startpoint+2].reward, self.local_transitions[sim_startpoint+2].done
        else:
            flag = 0
            return None, flag
        if sim_dones2 or flag == 0: # if episode ends, stop TS
            flag = 0
            return None, flag
        sim_actions1_star_idx_inv, sim_actions1_star_inv = self.agent.getInvBCActions(sim_actions0_star_idx, sim_actions1_star_idx, self.sigma, "gaussian")
        temp = np.zeros([1, self.agent.n_a])
        temp[0, :] = sim_actions1_star_inv
        sim_states_new, sim_obs_new, _, _, sim_flag = self.simulate(torch.from_numpy(temp))

        sim_actions_new_star_idx,  sim_actions_new_star= self.agent.getSimBCActions(sim_actions1_star_idx_inv, torch.tensor(sim_actions1_star_idx[0]))
        
        sim_obs0= [sim_obs0, sim_obs1, sim_obs2, sim_obs_new]
        scaled_sim_action, unscales_sim_action = self.agent.decodeSingleActions(*[torch.tensor(sim_actions1_star_idx)[i] for i in range(5)])
        actions = [unscales_sim_action, sim_actions_new_star[0]]
        # fig = visualizeTransitionTS(sim_obs, actions)
        # fig.clf()
        # sim_obs = [sim_obs1, sim_obs2]
        # actions = [sim_actions1_star_idx]
        # fig = visualizeTransition(agent, sim_obs, actions)

        is_expert = 1
        transition = ExpertTransition(np.array(sim_states_new), sim_obs_new.astype(np.float32), sim_actions_new_star_idx[0].numpy(),
                                    sim_rewards2, sim_states2, sim_obs2, sim_dones2,
                                    sim_steps_lefts, np.array(is_expert))
        # if obs_type == 'pixel':
        #     transition = normalizeTransition(transition)
        if sim_flag == False:
            flag = 0
        return transition, flag

if __name__ == "__main__":
    data = NpyBuffer()