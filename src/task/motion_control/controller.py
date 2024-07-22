from mypath import *

from IPython import embed
import numpy as np
import os, time, pickle, copy, torch, random, copy, math
from datetime import datetime
from abc import ABC, abstractclassmethod

from fairmotion.data import bvh
from fairmotion.ops import conversions
from fairmotion.utils import constants
from fairmotion.utils import spring_utils
from fairmotion.ops.math import lerp, slerp
from task.motion_control.traj_predict_utils import *


class MotionController(ABC):
    def __init__(self, save_path=None, foot_cleanup=False, dbg=False):
        if save_path is None:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.save_path = os.path.join(PRJ_DIR, "record", timestamp)

        self.save_path = save_path
        self.foot_cleanup = foot_cleanup
        self.dbg = dbg
        self.dbg_msg = "BLAH BLAH DBG"

        # necessary variables(assume these exists):
        # self.motion           Motion
        # self.dt               float

        # TODO: get these as input aguments
        self.traj_interval = 5
        self.traj_num = 6
        self.fps = 30
        self.dt = 1.0 / self.fps

        self.control_mode = "velocity"
        self.min_speed = 0.1  # cm scale
        self.max_speed = 600

        ## parameters
        self.sim_vel_halflife = 0.27
        self.sim_rot_halflife = 1.0  # 0.5

        self.sim_run_fwrd_speed = 4.0 * 100
        self.sim_run_side_speed = 3.0 * 100
        self.sim_run_back_speed = 2.5 * 100

        self.sim_walk_fwrd_speed = 1.75 * 100
        self.sim_walk_side_speed = 1.5 * 100
        self.sim_walk_back_speed = 1.25 * 100

        self.synch_pos_factor = 0.5
        self.synch_rot_factor = 0.5

        self.clamping_max_distance = 0.15
        self.clamping_max_angle = 0.5 * np.pi

        # command inputs that are needed # TODO FIX
        self._cam_azimuth = 0
        self._gamepadstick_left = np.array([0, 0, 0])
        self._gamepadstick_right = np.array([0, 0, 0])
        self._desired_strafe = 0
        self._sim_fwrd_speed = self.sim_run_fwrd_speed
        self._sim_side_speed = self.sim_run_side_speed
        self._sim_back_speed = self.sim_run_back_speed

    @abstractclassmethod
    def reset(self):
        pass

    @abstractclassmethod
    def step(self):
        pass

    @abstractclassmethod
    def save_run(self):
        pass

    def updateGoal(self, speed, angle):
        self.desired_vel = np.array([speed * np.sin(angle), 0, speed * np.cos(angle)])
        self.desired_rot = conversions.Ay2R(angle)
        # print(f"speed {speed:.4f}\tangle {angle:.4f}", '\n', self.desired_rot)
        # self._gamepadstick_left = self.desired_vel # TODO

    ## random goal / spring - related ftns
    def updateGoalRandom(self, frame):
        angle = -np.pi + 2 * random.random() * np.pi
        speed = self.min_speed + random.random() * (self.max_speed - self.min_speed)
        # print("controller base updateGoalRandom:: frame ", frame, ' new goal: ', angle, speed)
        self.dbg_msg += f"controller base updateGoalRandom:: frame {frame}, angle: {angle:.4f}, speed: {speed:.4f}\n"
        self.updateGoal(speed, angle)

    def spring_init(self):
        # desired : update directly by Command(input)
        self.desired_vel = np.zeros(3)
        self.desired_rot = constants.eye_R()
        self.desired_gait = 0.0
        self.desired_gait_velocity = 0.0

        # update by accumulation
        self.traj_desired_vel = np.zeros((self.traj_num, 3))
        self.traj_desired_rot = constants.eye_R()[np.newaxis, ...].repeat(
            self.traj_num, 0
        )

        # update by desired traj & spring
        self.traj_pos = np.zeros((self.traj_num, 3))
        self.traj_vel = np.zeros((self.traj_num, 3))
        self.traj_acc = np.zeros((self.traj_num, 3))
        self.traj_rot = constants.eye_R()[np.newaxis, ...].repeat(self.traj_num, 0)
        self.traj_ang_vel = np.zeros((self.traj_num, 3))

        # spring
        self.sim_x = np.zeros(3)
        self.sim_v = np.zeros(3)
        self.sim_a = np.zeros(3)
        self.sim_rot = constants.eye_R()
        self.sim_ang_vel = np.zeros(3)

        # to detect sudden change and force search
        self.search_time = 0.1
        self.search_timer = self.search_time
        self.force_search_timer = self.search_time

        self.desired_vel_change_curr = np.zeros(3)
        self.desired_vel_change_prev = np.zeros(3)
        self.desired_vel_change_thres = 50.0
        self.desired_rot_change_curr = np.zeros(3)
        self.desired_rot_change_prev = np.zeros(3)
        self.desired_rot_change_thres = 50.0

        # log
        self.log_desired_vel = [copy.copy(self.desired_vel)]
        self.log_desired_rot = [copy.copy(self.desired_rot)]
        self.log_desired_gait = [copy.copy(self.desired_gait)]
        self.log_desired_gait_velocity = [copy.copy(self.desired_gait_velocity)]

        # update by accumulation
        self.log_traj_desired_vel = [copy.copy(self.traj_desired_vel)]
        self.log_traj_desired_rot = [copy.copy(self.traj_desired_rot)]

        # update by desired traj & spring
        self.log_traj_pos = [copy.copy(self.traj_pos)]
        self.log_traj_vel = [copy.copy(self.traj_vel)]
        self.log_traj_acc = [copy.copy(self.traj_acc)]
        self.log_traj_rot = [copy.copy(self.traj_rot)]
        self.log_traj_ang_vel = [copy.copy(self.traj_ang_vel)]

        # spring
        self.log_sim_x = [copy.copy(self.sim_x)]
        self.log_sim_v = [copy.copy(self.sim_v)]
        self.log_sim_a = [copy.copy(self.sim_a)]
        self.log_sim_rot = [copy.copy(self.sim_rot)]
        self.log_sim_ang_vel = [copy.copy(self.sim_ang_vel)]

        # dbg msg
        self.log_dbg_msg = ["INIT MSG"]
        self.log_cam = []

        # control
        self.log_goal_action = [self.database.action2int["base"]]

    def spring_adjust(self, curFrame):

        # simulation_positions_update
        self.sim_x, self.sim_v, self.sim_a = (
            spring_utils.simple_spring_damper_implicit_a(
                self.sim_x,
                self.sim_v,
                self.sim_a,
                self.desired_vel,
                self.sim_vel_halflife,
                self.dt,
                # obstacles_positions,
                # obstacles_scales
            )
        )

        # sim_rotations_update
        self.sim_rot, self.sim_ang_vel = (
            spring_utils.simple_spring_damper_implicit_quat(
                self.sim_rot,
                self.sim_ang_vel,
                self.desired_rot,
                self.sim_rot_halflife,
                self.dt,
            )
        )

        cur_T = self.motion.poses[-1].get_root_facing_transform_byRoot(use_height=False)
        synchronized_position = lerp(self.sim_x, cur_T[:3, 3], self.synch_pos_factor)
        synchronized_rotation = slerp(
            self.sim_rot, cur_T[:3, :3], self.synch_rot_factor
        )

        self.sim_x = synchronized_position
        self.sim_rot = synchronized_rotation

    def traj_update(self):
        traj_desired_rotations_predict(
            self.traj_desired_rot,
            self.traj_desired_vel,
            self.desired_rot,
            self._cam_azimuth,  # joystick input or default value
            self._gamepadstick_left,  # joystick input or default value
            self._gamepadstick_right,  # joystick input or default value
            self._desired_strafe,  # joystick input or default value
            self.traj_interval * self.dt,
        )

        traj_rotations_predict(
            self.traj_rot,
            self.traj_ang_vel,
            self.sim_rot,
            self.sim_ang_vel,
            self.traj_desired_rot,
            self.sim_rot_halflife,
            self.traj_interval * self.dt,
        )

        try:
            traj_desired_velocities_predict(
                self.traj_desired_vel,
                self.traj_rot,
                self.desired_vel,
                self._cam_azimuth,
                self._gamepadstick_left,  # joystick input or default value
                self._gamepadstick_right,  # joystick input or default value
                self._desired_strafe,  # joystick input or default value
                self._sim_fwrd_speed,  # joystick input or default value
                self._sim_side_speed,  # joystick input or default value
                self._sim_back_speed,  # joystick input or default value
                self.traj_interval * self.dt,
            )
        except:
            print("cntroller traj_desired_velocities_predict err")
            embed()

        traj_positions_predict(
            self.traj_pos,
            self.traj_vel,
            self.traj_acc,
            self.sim_x,
            self.sim_v,
            self.sim_a,
            self.traj_desired_vel,
            self.sim_vel_halflife,
            self.traj_interval * self.dt,
            # obstacles_positions,
            # obstacles_scales)
        )

    def log(self):
        self.log_desired_vel.append(copy.copy(self.desired_vel))
        self.log_desired_rot.append(copy.copy(self.desired_rot))
        self.log_goal_action.append(
            self.database.action2int[self.goal_action]
        )  # int: no need to copy
        self.log_desired_gait.append(copy.copy(self.desired_gait))
        self.log_desired_gait_velocity.append(copy.copy(self.desired_gait_velocity))

        # update by accumulation
        self.log_traj_desired_vel.append(copy.copy(self.traj_desired_vel))
        self.log_traj_desired_rot.append(copy.copy(self.traj_desired_rot))

        # update by desired traj & spring
        self.log_traj_pos.append(copy.copy(self.traj_pos))
        self.log_traj_vel.append(copy.copy(self.traj_vel))
        self.log_traj_acc.append(copy.copy(self.traj_acc))
        self.log_traj_rot.append(copy.copy(self.traj_rot))
        self.log_traj_ang_vel.append(copy.copy(self.traj_ang_vel))

        # spring
        self.log_sim_x.append(copy.copy(self.sim_x))
        self.log_sim_v.append(copy.copy(self.sim_v))
        self.log_sim_a.append(copy.copy(self.sim_a))
        self.log_sim_rot.append(copy.copy(self.sim_rot))
        self.log_sim_ang_vel.append(copy.copy(self.sim_ang_vel))

        # dbg_msg
        self.log_dbg_msg.append(self.dbg_msg)

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # for j in self.motion.skel.joints:
        #     if 'END' in j.name: j.info['dof'] = 0

        bvh_path = os.path.join(self.save_path, "result.bvh")
        bvh.save(
            self.motion,
            bvh_path,
            scale=1.0,
            rot_order="ZXY",
            verbose=False,
            ee_as_joint=True,
        )
        print("saved bvh to : ", bvh_path)

        # save control
        ctrl_path = os.path.join(self.save_path, "ctrl")

        if not hasattr(self, "log_goal_action"):  # optional variable
            self.log_goal_action = np.zeros(0)  # dummy

        np.savez_compressed(
            ctrl_path,
            traj_x=self.log_traj_x,
            traj_v=self.log_traj_v,
            traj_a=self.log_traj_a,
            traj_rot=self.log_traj_rot,
            traj_ang_vel=self.log_traj_ang_vel,
            goal_action=self.log_goal_action,
            fc=self.fc,
            v_goal=self.log_v_goal,
            db_frame_trace=self.db_frame_trace,
        )

        # TODO: varying_motion
        # if len(self.varying_motions) != 0:
        #     for frame in range(len(self.varying_motions)):
        #         bvh_path = os.path.join(self.save_path, str(frame)+".bvh")
        #         bvh.save(self.varying_motions[frame], bvh_path, scale=1.0, rot_order="ZXY", verbose=False, ee_as_joint=True)
