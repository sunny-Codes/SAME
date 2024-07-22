import numpy as np
import os, copy
from IPython import embed
from functools import partial

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import glfw

from fairmotion.viz import gl_render
from fairmotion.viz.bvh_visualizer import WorldStepper
from fairmotion.utils.contact_utils import Contact

from default_veiwer import get_default_viewer
from task.motion_control.command import AutoCommand
from fairmotion.ops import conversions


class ControllerWorldStepper(WorldStepper):
    def __init__(self, controller, command, viewer, fps=30) -> None:
        self.controller = controller
        self.command = command
        self.viewer = viewer
        super(ControllerWorldStepper, self).__init__(fps)

        self.cur_frame = 0
        self.cur_time = 0
        self.last_update_time = 0
        self.update_interval = 0

    def idle(self):
        if self.isPlaying:
            time_elapsed = self.time_checker.get_time(restart=False)
            self.time_checker.begin()
            self.set_time(self.cur_time + time_elapsed)  # * self.play_speed)

    def set_time(self, new_time):
        frame = int(new_time * self.fps + 1e-05)
        if frame != self.cur_frame:
            if (frame - self.cur_frame) != 1:
                new_time -= (frame - self.cur_frame - 1) / self.fps
                frame = self.cur_frame + 1

            self.update_interval = new_time - self.last_update_time
            self.last_update_time = new_time
            if not self.isReplay():
                self.command.step(frame)
                self.controller.step(frame)
                self.controller.log()
                self.viewer.log()

            if self.viewer:
                self.viewer.trackCameraUpdate(frame)
            self.cur_frame = frame

        self.cur_time = new_time

    def get_update_interval(self):
        return self.update_interval

    def reset(self, is_playing=False):
        super().reset(is_playing)
        self.cur_time = 0
        self.cur_frame = 0
        self.last_update_time = 0
        self.update_interval = 0
        self.command.reset()

    def isReplay(self):
        return self.cur_frame + 1 < self.controller.motion.num_frames()

    def incFrame(self):
        if self.cur_frame + 1 < self.controller.motion.num_frames():
            self.cur_frame += 1
            self.cur_time += 1 / self.fps
            if self.viewer:
                self.viewer.trackCameraUpdate(self.cur_frame)
        else:
            self.set_time(self.cur_time + 1 / self.fps)

    def decFrame(self):
        if self.cur_frame > 0:
            self.cur_frame -= 1
            self.cur_time -= 1 / self.fps
            if self.viewer:
                self.viewer.trackCameraUpdate(self.cur_frame)


def set_control_viewer(
    viewer, controller, command
):  # , command_type='auto', **kwargs):
    viewer.worldStepper = ControllerWorldStepper(controller, command, viewer)
    viewer.overlay = True
    viewer.fillInGround = True
    viewer.hide_origin = True
    if hasattr(controller, "dbg_motion") and controller.vis_dbg:
        viewer.update_motions([controller.motion, controller.dbg_motion])
    else:
        viewer.update_motions([controller.motion])
    viewer.set_trackCamera(True)

    viewer.log_cam_p = []
    viewer.log_cam_R = []

    def log(self):
        cam_cur = self.cam_cur
        cam_p = copy.copy(cam_cur.pos)
        cam_R = cam_cur.get_cam_rotation()
        self.log_cam_p.append(cam_p)
        self.log_cam_R.append(cam_R)

    viewer.log = partial(log, viewer)
    viewer.log()

    def save_run(self, save_dir):
        cam_p_stack = np.stack(self.log_cam_p)
        cam_R_stack = np.stack(self.log_cam_R) @ conversions.E2R(
            [0.5 * 3.141592, 0, 1 * 3.141592]
        )
        cam_R_stack = conversions.R2E(cam_R_stack, "xzy") / 3.14 * 180.0
        cam_R_stack[:, 0] += 180
        cam_stack = np.hstack((cam_p_stack, cam_R_stack))
        save_path = os.path.join(save_dir, "cam.csv")
        np.savetxt(save_path, cam_stack, delimiter=",")

    viewer.save_run = partial(save_run, viewer)

    def pickPoint(x, y):
        vport = glGetIntegerv(GL_VIEWPORT)
        mvmatrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        projmatrix = glGetDoublev(GL_PROJECTION_MATRIX)
        realy = vport[3] - y

        objx1, objy1, objz1 = gluUnProject(
            x, realy, 0, mvmatrix, projmatrix, vport
        )  # nearPos
        objx2, objy2, objz2 = gluUnProject(
            x, realy, 10, mvmatrix, projmatrix, vport
        )  # farPos

        x = objx1 + (objx2 - objx1) * (objy1) / (objy1 - objy2)
        z = objz1 + (objz2 - objz1) * (objy1) / (objy1 - objy2)
        return [x / viewer.bvh_scale, 0, z / viewer.bvh_scale]

    viewer.pickPoint = pickPoint

    def prior_key_callback(key, mode=None):
        if key == b"S" or ((key == glfw.KEY_S) and (mode == glfw.MOD_SHIFT)):
            save_dir = controller.save_run()
            viewer.save_run(save_dir)
            return True

        if key == b"s" or (key == glfw.KEY_S):
            return True
        elif key == b"[" or (key == glfw.KEY_LEFT_BRACKET):
            viewer.worldStepper.decFrame()
            return True
        elif key == b"]" or (key == glfw.KEY_RIGHT_BRACKET):
            viewer.worldStepper.incFrame()
            return True
        elif key == b"e" or (key == glfw.KEY_E):
            embed()
            return True
        elif key == b"z" or (key == glfw.KEY_Z):
            if viewer.mouse_last_pos is not None:
                mx, my = viewer.mouse_last_pos
                controller.x_goal_world = viewer.pickPoint(mx, my)
                print(
                    "picked global position(",
                    mx,
                    ",",
                    my,
                    ") ->",
                    controller.x_goal_world,
                )
            return True
        elif key == b"x" or (key == glfw.KEY_X):
            if viewer.mouse_last_pos is not None:
                pass  # TODO
                # mx, my = viewer.mouse_last_pos
                # curT = self.getCurrentRootTransform()
                # controller.x_goal_world = viewer.pickPoint(mx, my)
                # controller.v_goal_wordl = np.linalg.inv(curT) @
                # print('picked global position(', mx, ',', my , ') ->', controller.x_goal_world)
            return True

        # elif key == b"q" or (key == glfw.KEY_Q):
        #     controller.start_seed = int(input("press [seed] to start\n"))
        #     print(controller.start_seed)
        #     return True
        elif key == b"r" or (key == glfw.KEY_R):
            print("Reset")
            viewer.worldStepper.reset()
            controller.reset()
            viewer.update_motions([controller.motion])
            return True
        else:
            return False

    def extra_key_callback(key, mode=None):
        # SAVE
        if key == b"S" or ((key == glfw.KEY_S) and (mode == glfw.MOD_SHIFT)):
            controller.save_run()
            return True

        elif key == b"f" or (key == glfw.KEY_F):
            controller.foot_cleanup = not (controller.foot_cleanup)
            print("foot cleanup: ", ("yes" if controller.foot_cleanup else "no"))
            return True

        return False

    def extra_render_callback():
        frame = viewer.worldStepper.cur_frame
        control_color = [0.5, 0, 0, 1]
        try:
            p = controller.motion.poses[frame].get_root_transform()[:3, 3]
        except:
            print("ERR, Frame: ", frame)
        d = controller.log_desired_rot[frame][:3, 2]
        p[1] = 2
        gl_render.render_point(p, radius=2, color=control_color)
        gl_render.render_line(p, p + d * 20, line_width=5, color=control_color)

        traj_color_s = np.array([239 / 255, 175 / 255, 150 / 255, 1])
        traj_color_e = np.array([157 / 255, 152 / 255, 174 / 255, 1])
        for i in range(controller.traj_num):
            color_i = traj_color_s * i / controller.traj_num + traj_color_e * (
                1 - i / controller.traj_num
            )
            p = controller.log_traj_pos[frame][i]
            d = controller.log_traj_rot[frame][i][:3, 2]
            p[1] = 1
            gl_render.render_point(p, radius=2, color=color_i)
            gl_render.render_line(p, p + d * 20, line_width=3, color=color_i)
            if i != 0:
                gl_render.render_line(
                    controller.log_traj_pos[frame][i - 1],
                    p,
                    line_width=3,
                    color=color_i,
                )

        # if controller.foot_cleanup:
        #     for i, (fid, fc) in enumerate(
        #         zip(controller.feet_idx, controller.fc_trace[frame])
        #     ):
        #         if fc > Contact.contact_label_thres:
        #             contact_cls = controller.motion.contacts[fid]
        #             # gl_render.draw_fc(controller.motion.poses[frame], fid, color=[1,0,0])
        #             color = [1, 0, 0] if i == 0 else [0, 0, 1]
        #             color2 = [1, 0, 1]
        #             gl_render.render_circle(
        #                 conversions.p2T(contact_cls.contact_point_log[frame]),
        #                 r=5,
        #                 color=color,
        #                 draw_plane="zx",
        #             )
        #             gl_render.render_circle(
        #                 conversions.p2T(contact_cls.contact_position_log[frame]),
        #                 r=5,
        #                 color=color2,
        #                 draw_plane="zx",
        #             )

    def overlay_callback():
        w, h = viewer.window_size
        frame = viewer.worldStepper.cur_frame
        while (controller.log_dbg_msg[frame] == "") and (frame > 0):
            frame -= 1
        dbg_msg_closest = controller.log_dbg_msg[frame]  # closest non

        dbg_msg_list = dbg_msg_closest.split("\n")
        for i, dbg_msg in enumerate(dbg_msg_list):
            gl_render.render_text(
                dbg_msg,
                pos=[0.02 * w, 0.02 * (i + 1) * h],
                font=GLUT_BITMAP_HELVETICA_18,
            )

    viewer.overlay_callback = overlay_callback
    viewer.extra_render_callback = extra_render_callback
    viewer.prior_key_callback = prior_key_callback
    viewer.extra_key_callback = extra_key_callback


def run_screenless(controller, step_len=500):
    command = AutoCommand(controller)
    worldStepper = ControllerWorldStepper(controller, command, None)
    while worldStepper.cur_frame < step_len:
        worldStepper.idle()


def prepare_run(controller, command_mode="auto"):
    import argparse

    args = argparse.Namespace()
    args.imgui = True
    args.one_each = False
    viewer = get_default_viewer(args)
    if command_mode == "auto":
        command = AutoCommand(controller)
    # elif command_mode == 'joystick':
    #     command = JoystickCommand(controller, viewer.cam_cur)
    else:
        assert False, "other control mode not implemented yet."

    set_control_viewer(viewer, controller, command)
    return viewer, controller, command
