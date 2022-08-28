import time

from src.ur5 import UR5
from src.img_proxy import ImgProxy
from src.cloud_proxy import CloudProxy
# from src.planner import Planner
from src.two_bin_grasp_planner import TwoBinGraspPlanner
import skimage
import scipy
import ros_numpy
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sk_transform
from scipy.ndimage import median_filter
from src.utils import transformation
from sklearn.impute import SimpleImputer
import rospy
from scipy.ndimage import rotate

class Env:
    def __init__(self, ws_center=(-0.5, -0.05, -0.04), ws_x=0.2, ws_y=0.2, obs_size=(90, 90),
                 action_sequence='pxyr'):
        rospy.init_node('ur5_env')
        self.ws_center = ws_center
        self.ws_x = ws_x
        self.ws_y = ws_y
        self.z_min = 0.04
        self.workspace = np.asarray([[self.ws_center[0] - ws_x / 2, self.ws_center[0] + ws_x / 2],
                                     [self.ws_center[1] - ws_y/2, self.ws_center[1] + ws_y/2],
                                     [self.z_min, self.z_min+0.2]])
        # observation size in pixels
        self.obs_size = obs_size
        # observation size in meters
        self.obs_size_m = 0.3
        self.heightmap_resolution = self.obs_size_m / obs_size[0]
        # pxyzr, the sequence of input action in step function
        self.action_sequence = action_sequence

        self.ur5 = UR5()
        self.cloud_proxy = None
        self.old_heightmap = np.zeros((self.obs_size[0], self.obs_size[1]))
        self.heightmap = np.zeros((self.obs_size[0], self.obs_size[1]))

        self.heightmap_size = obs_size[0]

        self.ee_offset = 0.095

        self.current_episode_steps = 0
        self.max_steps = 50

        # the cumulative rotation of gripper in the current episode
        self.rotated = 0

        self.planner = None

        self.simulate_z_threshold = self.z_min + 0.1
        self.simulate_pos = None
        self.simulate_rot = None

    def getWorkSpace(self, bin_id=None):
        return self.workspace

    def _decodeAction(self, action):
        """
        decode input action base on self.action_sequence
        Args:
          action: action tensor

        Returns: motion_primative, x, y, z, rot

        """
        primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a),
                                                          ['p', 'x', 'y', 'z', 'r'])
        motion_primative = action[primative_idx] if primative_idx != -1 else 0
        x = action[x_idx]
        y = action[y_idx]
        z = action[z_idx]
        rz, ry, rx = 0, np.pi, 0
        if self.action_sequence.count('r') <= 1:
            rz = action[rot_idx] if rot_idx != -1 else 0
            ry = 0
            rx = 0
        elif self.action_sequence.count('r') == 2:
            rz = action[rot_idx]
            ry = action[rot_idx + 1]
            rx = 0
        elif self.action_sequence.count('r') == 3:
            rz = action[rot_idx]
            ry = action[rot_idx + 1]
            rx = action[rot_idx + 2]

        rot = (rx, ry, rz)

        return motion_primative, x, y, z, rot

    def _preProcessObs(self, obs):
        obs = scipy.ndimage.median_filter(obs, 2)
        return obs

    def getHeightmapReconstruct(self, gripper_pos=None, obs_size_m=None):
        # get orthographic projection image
        if gripper_pos is None:
            gripper_pos, _ = self.ur5.getEEPose()
        if obs_size_m is None:
            obs_size_m = self.obs_size_m
        # get img from camera
        obss = []
        for i in range(1):
            obss.append(self.cloud_proxy.getProjectImg(obs_size_m, self.obs_size[0], gripper_pos))
        obs = np.median(obss, axis=0)
        obs = self._preProcessObs(obs)
        return obs

    def getObs(self):
        self.heightmap = self.getHeightmapReconstruct()
        self.heightmap = np.clip(self.heightmap, -0.5, 0.5)
        gripper_img = self.getGripperImg()
        heightmap = np.copy(self.heightmap)
        heightmap[gripper_img.astype(bool)] = 0
        heightmap = heightmap.reshape([1, self.heightmap_size, self.heightmap_size])
        img = heightmap
        # img = np.stack((self.heightmap, gripper_img))
        is_holding = self.ur5.holding_state
        return is_holding, img

    def getGripperImg(self, gripper_state=None, gripper_rz=None):
        if gripper_state is None:
            gripper_state = self.ur5.getGripperState()
        if gripper_rz is None:
            gripper_rz = self.ur5._getEndEffectorRotation()[2]

        im = np.zeros((self.heightmap_size, self.heightmap_size))
        len = 0.098/self.obs_size_m * self.heightmap_size
        d = int(len * gripper_state)
        l = int(0.02/self.obs_size_m * self.heightmap_size/2)
        w = int(0.015/self.obs_size_m * self.heightmap_size/2)
        im[self.heightmap_size//2-d//2-w:self.heightmap_size//2-d//2+w, self.heightmap_size//2-l:self.heightmap_size//2+l] = 1
        im[self.heightmap_size//2+d//2-w:self.heightmap_size//2+d//2+w, self.heightmap_size//2-l:self.heightmap_size//2+l] = 1
        im = rotate(im, np.rad2deg(gripper_rz), reshape=False, order=0)
        return im

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        p, x, y, z, rot = self._decodeAction(action)
        # if self.rotated + rot[2] > np.pi or self.rotated + rot[2] < -np.pi:
        #     rot = (0, 0, 0)
        # else:
        #     self.rotated += rot[2]
        if self.ur5.joint_values[-1] > np.deg2rad(300) and rot[2] < 0:
            rot = (0, 0, 0)
        elif self.ur5.joint_values[-1] < -np.deg2rad(300) and rot[2] > 0:
            rot = (0, 0, 0)
        current_pos = self.ur5._getEndEffectorPosition()
        current_rot = self.ur5._getEndEffectorRotation()

        pos = np.array(current_pos) + np.array([x, y, z])
        rot = np.array(current_rot) + np.array(rot)
        # constrain action within the ws
        workspace = self.getWorkSpace()
        pos[0] = np.clip(pos[0], workspace[0, 0], workspace[0, 1])
        pos[1] = np.clip(pos[1], workspace[1, 0], workspace[1, 1])
        pos[2] = np.clip(pos[2], workspace[2, 0], workspace[2, 1])
        if pos[2] < workspace[2, 0] + 0.05 and z < 0:
            v = 0.7
        else:
            v = 0.7
        self.ur5.protective_stop_flag = False
        self.ur5.collision_flag = False
        self.ur5.controlGripper(p)
        self.ur5.current_target = pos[0], pos[1], pos[2], 0, 0, rot[2]
        rospy.sleep(0.3)
        # t0 = time.time()
        # self.ur5.gripper.waitUntilNotMoving()
        # t1 = 0.8 - (time.time() - t0)
        # if t1 > 0:
        #     rospy.sleep(t1)
        self.simulate_pos = pos
        self.simulate_rot = rot
        self.cloud_proxy.clearPointCloud()

        is_holding, heightmap = self.getObs()
        self.ur5.gripper.waitUntilNotMoving(max_it=20, sleep_time=0.1)
        self.ur5.holding_state = self.ur5.gripper.isHolding()
        is_holding = self.ur5.holding_state
        obs = is_holding, heightmap
        # check if grasp success
        done = self.checkTermination()
        reward = 1.0 if done else 0.0
        self.current_episode_steps += 1
        if not done:
            done = self.current_episode_steps >= self.max_steps
            if self.current_episode_steps >= self.max_steps:
                self.current_episode_steps = 0

        # penalty for raising safety flag
        if self.ur5.protective_stop_flag:
            reward -= 0.1

        # terminate episode after collision
        # if self.ur5.safety_flag and not self.ur5.holding_state:
        #     done = True

        # terminate episode after failing to recover from protective stop
        if self.ur5.fail_to_recover:
            done = True
            input('Failed to recover from protective stop. Please manually recover the arm')
            self.ur5.fail_to_recover = False

        return obs, reward, done

    def simulate(self, action):
        p, dx, dy, dz, r = self._decodeAction(action)
        dtheta = r[2]
        # pos = list(self.robot._getEndEffectorPosition())
        # gripper_rz = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())[2]
        pos = self.simulate_pos
        gripper_rz = self.simulate_rot[2]
        pos[0] += dx
        pos[1] += dy
        pos[2] += dz
        workspace = self.getWorkSpace()
        pos[0] = np.clip(pos[0], workspace[0, 0], workspace[0, 1])
        pos[1] = np.clip(pos[1], workspace[1, 0], workspace[1, 1])
        pos[2] = np.clip(pos[2], self.simulate_z_threshold, workspace[2, 1])
        self.simulate_pos = pos
        self.simulate_rot = [0, 0, gripper_rz]

        heightmap = self.getHeightmapReconstruct(pos)
        gripper_img = self.getGripperImg(p, gripper_rz + dtheta)
        heightmap[gripper_img.astype(bool)] = 0
        heightmap = heightmap.reshape([1, self.heightmap_size, self.heightmap_size])
        img = heightmap
        # gripper_img = gripper_img.reshape([1, self.heightmap_size, self.heightmap_size])
        # img = np.stack((obs, gripper_img))
        return (0, img), 0, 0

    def resetSimPose(self):
        current_pos = self.ur5._getEndEffectorPosition()
        current_rot = self.ur5._getEndEffectorRotation()
        self.simulate_pos = np.array(current_pos)
        self.simulate_rot = np.array(current_rot)

    def canSimulate(self):
        # pos = list(self.robot._getEndEffectorPosition())
        return not self.ur5.holding_state and self.simulate_pos[2] > self.simulate_z_threshold

    def checkTermination(self):
        raise NotImplementedError

    def getGripperClosed(self):
        return self.ur5.gripper.isClosed()

    def plotObs(self, cam_resolution):
        self.cam_resolution = cam_resolution
        obs = self.getHeightmapReconstruct()
        plt.imshow(obs[0, 0])
        plt.colorbar()
        plt.show()

    def getNextAction(self):
        return self.planner.getNextAction()
