from pyPS4Controller.controller import Controller
import time
from datetime import datetime

from src.ur5 import UR5
from src.img_proxy import ImgProxy
from src.cloud_proxy_desk import CloudProxyDesk
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
from src.env import Env
from threading import Thread
import collections
class MyController(Controller):

    def __init__(self, **kwargs):
        Controller.__init__(self, **kwargs)
        self.p_scale = 0.1
        self.xyz_scale = 0.01
        self.theta_scale = 0.1
        # self.p = self.on_R1_press() if self.on_R1_press() else self.on_R2_press() if self.on_R2_press() else 0
        # self.x = self.on_L3_left() if self.on_L3_left() else self.on_L3_right() if self.on_L3_right() else 0
        # self.y = self.on_L3_up() if self.on_L3_up() else self.on_L3_down() if self.on_L3_down() else 0
        # self.z = self.on_L1_press() if self.on_L1_press() else self.on_L2_press() if self.on_L2_press() else 0
        self.p = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.theta = 0
        self.delta_action = [0, 0, 0, 0, 0] # pxyzr
        self.reward = 0
        self.reset = 0
        self.okay = 0 # save step
        self.save = 0 # save episode
        self.grasp = 0 # grasp and save pre-grasp step

        # self.monitor_thread = Thread(target=self.monitor_delta_action)
        # self.monitor_thread.start()
        # self.action_dict={}

    def on_L1_press(self):
        self.z = 1

    def on_L1_release(self):
        self.z = 0

    def on_L2_press(self, value):
        self.z = -1

    def on_L2_release(self):
        self.z = 0

    def on_R1_press(self):
        self.p = 1

    def on_R1_release(self):
        self.p = 0

    def on_R2_press(self, value):
        self.p = -1

    def on_R2_release(self):
        self.p = 0

    def on_square_press(self):
        self.theta = 1

    def on_square_release(self):
        self.theta = 0

    def on_circle_press(self):
        self.theta = -1

    def on_circle_release(self):
        self.theta = 0

    def on_L3_up(self, value):
        # value = abs(value)
        # self.y = round(value/32767, 2)
        self.x = -1

    def on_L3_down(self, value):
        # value = abs(value)
        # self.y = -round(value/32767, 2)
        self.x = 1

    def on_L3_left(self, value):
        # value = abs(value)
        # self.x = -round(value/32767, 2)
        self.y = -1

    def on_L3_right(self, value):
        # value = abs(value)
        # self.x = round(value/32767, 2)
        self.y = 1

    def on_L3_y_at_rest(self):
        self.x = 0

    def on_L3_x_at_rest(self):
        self.y = 0

    def on_triangle_press(self):
        self.reset = 1
        self.reward = 1

    def on_triangle_release(self):
        self.reset = 0
        self.reward = 0

    def on_x_press(self):
        self.reset = 1
        self.reward = 0

    def on_x_release(self):
        self.reset = 0
        self.reward = 0

    def on_up_arrow_press(self):
        self.grasp = 0
        self.okay = 1

    def on_down_arrow_press(self):
        self.grasp = 1
        self.okay = 1

    def on_up_down_arrow_release(self):
        self.okay = 0

    def on_playstation_button_press(self):
        self.okay = 1

    def on_playstation_button_release(self):
        self.okay = 0

    def on_share_press(self):
        self.save = 1

    def on_share_release(self):
        self.save = 0

    def monitor_delta_action(self):
        while True:
            self.delta_action = [self.p * self.p_scale,
                                 self.x * self.xyz_scale,
                                 self.y * self.xyz_scale,
                                 self.z * self.xyz_scale,
                                 self.theta * self.theta_scale]
            print(f"Time: {time.time()}, [p, x, y, z, theta]: {self.delta_action}")
            time.sleep(0.5)

    def getAction(self):
        return [self.p * self.p_scale,
                 self.x * self.xyz_scale,
                 self.y * self.xyz_scale,
                 self.z * self.xyz_scale,
                 self.theta * self.theta_scale]



class DeskEnv(Env):
    def __init__(self, ws_x=0.4, ws_y=0.4, obs_size=(128, 128), action_sequence='pxyzr'):
        super().__init__(ws_x=ws_x, ws_y=ws_y, obs_size=obs_size, action_sequence=action_sequence)
        self.desk_center = (-0.527, -0.005)
        self.z_min = -0.080
        # workspace for two bins
        self.desk_workspace = np.asarray([[self.desk_center[0] - ws_x / 2, self.desk_center[0] + ws_x / 2],
                                          [self.desk_center[1] - ws_y/2, self.desk_center[1] + ws_y/2],
                                          [self.z_min, self.z_min+0.2]])


        self.cloud_proxy = CloudProxyDesk(self.desk_center, self.z_min)
        self.max_steps = 50
        self.current_episode_steps = -1

        self.controller = MyController(interface="/dev/input/js0", connecting_using_ds4drv=False)

        # you can start listening before controller is paired, as long as you pair it within the timeout window
        self.ps4_thread = Thread(target=self.controller.listen, kwargs={'timeout': 60})
        self.ps4_thread.start()
        self.desk_id = 3
        self.obs_size_m = 0.45

    def getWorkSpace(self):
        return self.desk_workspace

    def isDeskEmpty(self):
        desk_center = self.getWorkSpace()
        img = self.cloud_proxy.getProjectImg(0.39, 128, (desk_center[0], desk_center[1], self.z_min+0.1))
        img = self.z_min+0.1-img
        return (img > self.z_min+0.025).sum() < 20

    def reset(self):
        self.ur5.moveToHome(3)
        self.ur5.gripper.openGripper()
        self.ur5.gripper.waitUntilNotMoving()
        self.cloud_proxy.clearPointCloud()

        self.cloud_proxy.clearPointCloud()
        obs = self.getObs()
        self.rotated = 0
        self.current_episode_steps = 0
        self.resetSimPose()
        self.controller.reward=0
        return obs

    def getStepSpace(self):
        x, y, z = self.ur5._getEndEffectorPosition()
        rx, ry, rz = self.ur5._getEndEffectorRotation()
        step_space = np.asarray([[x - self.dpos, x + self.dpos],
                                  [y - self.dpos, y + self.dpos],
                                  [z - self.dpos, z + self.dpos],
                                 [rz - self.drot, rz + self.drot]])
        return step_space


    def is_action_valid(self, action, gripper_state_):
        # to determine whether any actions is big enough to consider
        # to prevent camera keep working while no actions are taken
        # action: pxyzr (r=rz), gripper_state_: next gripper state
        action = np.array(action)
        return (action[1:4] > 0.0005).any() or (action[4] > 0.002) or abs(gripper_state_-action[0] > 0.01)

    def encodeAction(self, action):
        action = np.array(action)
        x, y, z = action[1:4]/self.dpos
        return np.array([action[0], x, y, z, action[4]])

    def reset_arm(self):
        self.ur5.moveToHome(3)
        self.ur5.gripper.openGripper()
        rospy.sleep(0.6)
        self.cloud_proxy.clearPointCloud()

        is_holding, heightmap = self.getObs()
        pointCloud = self.cloud_proxy.cloud
        state = self.ur5.getGripperState()
        self.ur5.holding_state = self.ur5.gripper.isHolding()
        is_holding = self.ur5.holding_state
        obs = is_holding, heightmap
        return state, obs, pointCloud


    def checkTermination(self):
        if self.controller.reward:
            return 1
        else:
            return 0
        # x = input("Press o if reward else press x")
        # while 1:
        #     if x == "x":
        #         break
        #         return 0
        #     elif x == "o":
        #         break
        #         return 1
        #     else:
        #         print("Please press o for reward, x for non-reward")
        #         x = input("Press o if reward else press x")










if __name__ == "__main__":
    # controller = MyController(interface="/dev/input/js0", connecting_using_ds4drv=False)
    # # you can start listening before controller is paired, as long as you pair it within the timeout window
    # controller.listen(timeout=60)
    now = datetime.now()
    current_time =str(now).replace(' ','-')
