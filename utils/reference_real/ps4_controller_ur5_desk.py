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
from desk_env import MyController

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')




def visualizeBC(obs, obs_, scaled_action, obs_size_m=0.3):

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(obs[1][0])
    axes[0].set_title("obs")
    axes[0].arrow(x=64, y=64, dx=scaled_action[2] / obs_size_m * 128, dy=scaled_action[1] / obs_size_m * 128, width=.8)
    axes[0].text(0, 0, u"\u2191", rotation=scaled_action[4] * 180 / np.pi)
    axes[0].text(10, 0, f": {np.round(scaled_action[4] * 180 / np.pi, 2)}")
    axes[0].text(90, 0, f"z: {np.round(scaled_action[3], 3)}")
    axes[1].imshow(obs_[1][0])
    axes[1].set_title("obs_")


    plt.show()


class PS4ExpertRecorderDesk(Env):
    def __init__(self, ws_x=0.4, ws_y=0.4, obs_size=(128, 128), action_sequence='pxyzr', mode='move', save_type='npy', save_path='./transition_data.npy', dpos='0.02', drot_n=4, obs_size_m=0.3):
        super().__init__(ws_x=ws_x, ws_y=ws_y, obs_size=obs_size, action_sequence=action_sequence)
        self.desk_center = (-0.527, -0.005)
        self.z_min = -0.080
        # workspace for two bins
        self.desk_workspace = np.asarray([[self.desk_center[0] - ws_x / 2, self.desk_center[0] + ws_x / 2],
                                          [self.desk_center[1] - ws_y/2, self.desk_center[1] + ws_y/2],
                                          [self.z_min, self.z_min+0.2]])


        self.cloud_proxy = CloudProxyDesk(self.desk_center, self.z_min)

        self.controller = MyController(interface="/dev/input/js0", connecting_using_ds4drv=False)
        # you can start listening before controller is paired, as long as you pair it within the timeout window
        self.ps4_thread = Thread(target=self.controller.listen, kwargs={'timeout': 60})
        self.ps4_thread.start()

        # now support "move" and "record" mode
        # record mode is limited by dpos, as move mode is not
        # "record": press playstation button to forward to next step
        self.mode = mode
        print(f"In '{self.mode}' mode")
        self.manipulate_thread = Thread(target=self.manipulate, kwargs={'mode': self.mode})
        self.manipulate_thread.start()
        print("manipulating")

        # for record_discrete mode
        self.transition_data = []
        self.save_type = save_type
        self.save_path = save_path

        # define action space
        self.dpos = dpos
        self.drot = np.pi/drot_n

        self.obs_size_m = obs_size_m

        self.desk_id = 3

        self.enable_grasp = True # only True when recording tasks that involves pick and place
    def getWorkSpace(self):
        return self.desk_workspace

    def isDeskEmpty(self):
        desk_center = self.getWorkSpace()
        img = self.cloud_proxy.getProjectImg(0.39, 128, (desk_center[0], desk_center[1], self.z_min+0.1))
        img = self.z_min+0.1-img
        return (img > self.z_min+0.025).sum() < 20

    def moveAllObjToOtherBin(self):
        while not self.isBinEmpty(self.bin_id):
            obs = np.copy(self.heightmap)
            gripper_pos, _ = self.ur5.getEEPose()
            obs = -obs
            obs -= -(gripper_pos[2] - self.getWorkSpace()[2][0])
            obs[obs > self.getWorkSpace()[2][0] + 0.1] = 0
            # get all pixels with positive height value
            pixels = np.argwhere(obs > 0.03)
            if pixels.shape[0] == 0:
                pixels = np.argwhere(obs > 0)
            # random select a pixel as the target pos
            pixel = pixels[np.random.randint(pixels.shape[0])]
            # transform pixel into real xy values
            x = (pixel[0] - 128 // 2) * self.heightmap_resolution + gripper_pos[0]
            y = (pixel[1] - 128 // 2) * self.heightmap_resolution + gripper_pos[1]
            x = np.clip(x, self.getWorkSpace()[0][0], self.getWorkSpace()[0][1])
            y = np.clip(y, self.getWorkSpace()[1][0], self.getWorkSpace()[1][1])
            pixel_z = obs[pixel[0], pixel[1]]
            z = max(self.getWorkSpace()[2][0] + 0.01, pixel_z - 0.04)
            rz = (np.random.random() - 0.5) * np.pi
            self.ur5.holding_state = False
            self.ur5.pick(x, y, z, (0, 0, rz), self.bin_id)
            if self.ur5.holding_state:
                release_x = self.getWorkSpace(1 - self.bin_id)[0].mean() + (np.random.random() - 0.5) * 0.1
                release_y = self.getWorkSpace(1 - self.bin_id)[1].mean() + (np.random.random() - 0.5) * 0.1
                self.ur5.moveToP(release_x, release_y, self.z_min + 0.2, 0, 0, 0)
                self.ur5.gripper.openGripper()
                self.ur5.moveToHome(self.bin_id)
            self.cloud_proxy.clearPointCloud()
            self.getObs()

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
        ee_translation = self.ur5.getEEPose()
        return state, obs, pointCloud, ee_translation

    def manipulate(self, mode="move"):
        _, obs, pointCloud, ee_translation = self.reset_arm()
        trajectory = []
        counter = 0
        done = 0
        record_flag = 0
        refresh_flag = 1
        # holding_flag = False
        saveFile = 0
        reward = 0
        reset = 0
        while True:
            x, y, z = self.ur5._getEndEffectorPosition()
            rx, ry, rz = self.ur5._getEndEffectorRotation()
            self.ur5.holding_state = self.ur5.gripper.isHolding()

            delta_pos = self.controller.getAction()
            state = self.ur5.getGripperState()

            x = x + delta_pos[1]
            y = y + delta_pos[2]
            z = z + delta_pos[3]
            rz = rz + delta_pos[4]
            if self.enable_grasp:
                p_target = 0 if self.controller.grasp else 1
            else:
                p_target = np.clip(state + delta_pos[0], 0, 1)

            workspace = self.getWorkSpace()
            x = np.clip(x, workspace[0, 0], workspace[0, 1])
            y = np.clip(y, workspace[1, 0], workspace[1, 1])
            z = np.clip(z, workspace[2, 0], workspace[2, 1])
            if mode == "record" and not record_flag:
                # limit action in each step
                if refresh_flag:
                    ori_x, ori_y, ori_z = self.ur5._getEndEffectorPosition()
                    ori_rx, ori_ry, ori_rz = self.ur5._getEndEffectorRotation()
                    stepspace = self.getStepSpace()
                    refresh_flag = 0
                x = np.clip(x, stepspace[0, 0], stepspace[0, 1])
                y = np.clip(y, stepspace[1, 0], stepspace[1, 1])
                z = np.clip(z, stepspace[2, 0], stepspace[2, 1])
                rz = np.clip(rz, stepspace[3, 0], stepspace[3, 1])
                if self.controller.okay:
                    record_flag = 1
                    refresh_flag = 1


            if z < workspace[2, 0] + 0.05 and z < 0:
                v = 0.7
            else:
                v = 0.7
            self.ur5.protective_stop_flag = False
            self.ur5.collision_flag = False
            self.ur5.current_target = x, y, z, 0, 0, rz
            self.ur5.controlGripper(p_target)
            # print(f"cur_pos: {[x, y, z]}, current_rot: {rx, ry, rz};\n",
            #       f"delta_pos: {delta_pos}, target: {self.ur5.current_target}")
            rospy.sleep(0.01)


            if mode == 'record': # record a transition once got an image
                # get expert planner


                reward = 0 if not self.controller.reward else 1
                reset = 0 if not self.controller.reset else 1
                saveFile = 0 if not self.controller.save else 1
                # if self.is_action_valid(planner_actions_star_idx, state_) or self.controller.okay:
                state_ = self.ur5.getGripperState()
                if record_flag == 1:
                    x_, y_, z_ = self.ur5._getEndEffectorPosition()
                    rx_, ry_, rz_ = self.ur5._getEndEffectorRotation()


                    if not self.enable_grasp:
                        self.ur5.gripper.waitUntilNotMoving(max_it=20, sleep_time=0.1)
                    self.ur5.holding_state = self.ur5.gripper.isHolding()
                    # IMPORTANT(for enable_grasp=False): you need to keep holding R2 to keep holding_state true, otherwise false even if it's holding sth.
                    is_holding = self.ur5.holding_state
                    p_action = 0 if is_holding == 1 else state_


                    # if self.enable_grasp and state_ < 0.8:
                    #     # before grasing, close the gripper a little and press record
                    #     # in order to gather the expert p action 0 while holding state=0
                    #     p_action = 0
                    if self.enable_grasp:
                        p_action = 0 if self.controller.grasp else 1
                        # grasp_flag =0 if not self.controller.grasp else 1
                        # if grasp_flag:


                    action = [p_action, x_ - ori_x, y_ - ori_y, z_ - ori_z, rz_ - ori_rz]
                    planner_actions_star_idx = self.encodeAction(action)

                    print(f"scaled action: {action}; \nis_holding: {is_holding}")
                    start_time = time.time()
                    self.cloud_proxy.clearPointCloud()

                    obs_ = self.getObs()
                    print(f"getObs finished in {time.time()-start_time}")
                    # self.ur5.holding_state = self.ur5.gripper.isHolding()


                    pointCloud_ = self.cloud_proxy.cloud
                    #---visualiztion
                    # plt.imshow(obs[1][0])
                    # plt.imshow(obs_[1][0])
                    # plt.show()
                    visualizeBC(obs, obs_, action, self.obs_size_m)
                    #-------------
                    # transition = ExpertTransition(np.array(state), obs, np.array(planner_actions_star_idx),
                    #                               np.array(reward), np.array(state_), np.array(obs_), np.array(reset),
                    #                               np.array(100), np.array(1))
                    # add pointcloud, endeffector position, grasping flag on original transition
                    transition = [np.array(is_holding), obs, np.array(planner_actions_star_idx),
                                  np.array(reward), np.array(state_), obs_, np.array(reset),
                                  np.array(100), np.array(1), pointCloud, np.array([ori_x, ori_y, ori_z, ori_rz]), self.controller.grasp]

                    obs = obs_
                    state = state_
                    pointCloud = pointCloud_
                    trajectory.append(transition)
                    record_flag = 0

            if reward:
                counter +=1
                print(f"saving No.{counter} episode into buffer")
                self.transition_data.append(trajectory[1:])
                trajectory = []
                print("resetting")
                state, obs, point_cloud, ee_translation = self.reset_arm()
                refresh_flag = 1
                # holding_flag = False
                print(f"now you have {len(self.transition_data)} episodes")
            elif reset:
                trajectory = []
                print("abandon this episode and resetting")
                state, obs, point_cloud, ee_translation = self.reset_arm()
                refresh_flag = 1
                # holding_flag = False
                print(f"now you have {len(self.transition_data)} episodes")
            elif saveFile:
                # self.transition_data.append(trajectory)
                trajectory = []
                if self.save_type == "buffer":
                    NotImplementedError
                elif self.save_type == "npy":
                    np.save(self.save_path, self.transition_data, allow_pickle=True)
                else:
                    NotImplementedError
                print("resetting and exiting")
                state, obs, point_cloud, ee_translation = self.reset_arm()
                refresh_flag = 1
                # holding_flag = False
                print(f"saving {len(self.transition_data)} episodes ({self.save_type}) into dir: {self.save_path}")
                break

        self.ur5.moveToHome(self.desk_id)
        self.ur5.gripper.openGripper()








if __name__ == "__main__":
    # controller = MyController(interface="/dev/input/js0", connecting_using_ds4drv=False)
    # # you can start listening before controller is paired, as long as you pair it within the timeout window
    # controller.listen(timeout=60)
    now = datetime.now()
    current_time =str(now).replace(' ','-')
    ps4Recorder = PS4ExpertRecorderDesk(mode="record", save_path=f"/home/ur5/mingxi_ws/DATA/transition_data_{current_time}", dpos=0.02, drot_n=8, obs_size_m=0.6)
