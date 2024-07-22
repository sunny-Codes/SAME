# Copyright (c) Facebook, Inc. and its affiliates.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from fairmotion.viz.bvh_visualizer import MocapViewer

import numpy as np
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer as ImguiRenderer
from IPython import embed
from argparse import Namespace


class Viewer_imgui(MocapViewer):
    def __init__(
        self,
        motions,
        play_speed=1,
        joint_scale=1,
        link_scale=1,
        render_overlay=False,
        hide_origin=False,
        bvh_scale=1,
        trackCamera=False,
        **kwargs,
    ):
        super().__init__(
            motions,
            play_speed,
            joint_scale,
            link_scale,
            render_overlay,
            hide_origin,
            bvh_scale,
            trackCamera,
            **kwargs,
        )
        self.init()
        self.imgui_setup_done = False
        self.ui_variables = Namespace()
        self.ui_ftns = []

    # def assign_ftns(self):
    def should_close(self):
        glfw.set_window_should_close(self.window, True)

    def set_imgui_variables(self):
        pass

    def ui(self):
        for ftn in self.ui_ftns:
            ftn()

    def init(self):
        imgui.create_context()
        if not glfw.init():
            return

        self.window = glfw.create_window(
            self.window_size[0], self.window_size[1], "title", None, None
        )
        if not self.window:
            glfw.terminate()
            return

        glfw.make_context_current(self.window)

        glutInit()
        if self.use_msaa:
            glutInitDisplayMode(
                GLUT_RGBA
                | GLUT_DOUBLE
                | GLUT_ALPHA
                | GLUT_DEPTH
                # | GLUT_MULTISAMPLE
            )
        else:
            glutInitDisplayMode(
                GLUT_RGBA
                | GLUT_DOUBLE
                | GLUT_ALPHA
                | GLUT_DEPTH
                | GLUT_MULTISAMPLE  # SM) ON for better quality
            )
        self._init_GL(*self.window_size)
        self.resize_GL(*self.window_size)

        self.impl = ImguiRenderer(self.window, attach_callbacks=False)

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_window_size_callback(self.window, self._on_resize)
        glfw.set_char_callback(self.window, self._on_char)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        # glfw.set_joystick_callback(self._on_joystick)

        self.time_checker.begin()

        print("init done")

    def run(self):
        self.set_imgui_variables()
        self.imgui_setup_done = True

        # previous_time = glfw.get_time()
        # Loop until the user closes the window
        while not glfw.window_should_close(self.window):
            self.idle_callback()
            glfw.poll_events()
            self.impl.process_inputs()

            # current_time = glfw.get_time()
            # delta_time = current_time - previous_time
            # previous_time = current_time
            # self.update(current_time, delta_time)
            self.draw_GL(False)

            imgui.new_frame()
            self.ui()
            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

        self.impl.shutdown()
        glfw.terminate()

    def prior_key_callback(self, key, mode):
        return False

    def on_key(self, key, mode):
        if self.prior_key_callback(key, mode):
            return True

        if len(self.motions) > 0:
            motion = self.motions[self.file_idx]

        if key == glfw.KEY_ESCAPE:
            print("Hit ESC key to quit.")
            self.impl.shutdown()
            glfw.terminate()
        elif key == glfw.KEY_E:
            embed()
            return True

        elif key == glfw.KEY_S:
            self.worldStepper.reset()

            return True

        elif key == glfw.KEY_RIGHT_BRACKET:
            if len(self.motions) <= 0:
                return False
            next_frame = min(motion.num_frames() - 1, self.worldStepper.cur_frame + 1)
            self.worldStepper.set_time(motion.frame_to_time(next_frame))
            return True

        elif key == glfw.KEY_LEFT_BRACKET:
            if len(self.motions) <= 0:
                return False
            prev_frame = max(0, self.worldStepper.cur_frame - 1)
            self.worldStepper.set_time(motion.frame_to_time(prev_frame))
            return True

        elif (key == glfw.KEY_EQUAL) and (mode == glfw.MOD_SHIFT):  # +
            self.play_speed = min(self.play_speed + 0.2, 5.0)
            self.worldStepper.play_speed = self.play_speed
            return True

        elif key == glfw.KEY_MINUS:
            self.play_speed = max(self.play_speed - 0.2, 0.2)
            self.worldStepper.play_speed = self.play_speed
            return True
        elif key == glfw.KEY_R or key == glfw.KEY_V:
            if len(self.motions) <= 0:
                return False
            self.worldStepper.reset()
            # end_time = motion.length()
            end_frame = motion.num_frames()
            fps = motion.fps
            save_path = input("Enter directory/file to store screenshots/video: ")
            cnt_screenshot = 0
            dt = 1 / fps
            gif_images = []
            while self.worldStepper.cur_frame < end_frame:
                # while self.worldStepper.cur_time <= end_time:
                print(
                    f"Recording progress: {self.worldStepper.cur_frame}/{end_frame} ({int(100*self.worldStepper.cur_frame/end_frame)}%) \r",
                    end="",
                )
                if key == b"r":
                    utils.create_dir_if_absent(save_path)
                    name = "screenshot_%04d" % (cnt_screenshot)
                    self.save_screen(dir=save_path, name=name, render=True)
                else:
                    image = self.get_screen(render=True)
                    gif_images.append(image.convert("P", palette=Image.ADAPTIVE))
                self.worldStepper.incFrame()
                # self.worldStepper.set_time(self.worldStepper.cur_time + dt)

                cnt_screenshot += 1
            if key == glfw.KEY_V:
                utils.create_dir_if_absent(os.path.dirname(save_path))
                gif_images[0].save(
                    save_path,
                    save_all=True,
                    optimize=False,
                    append_images=gif_images[1:],
                    loop=0,
                )
            return True

        elif key == glfw.KEY_SPACE:
            self.worldStepper.togglePlaying()
            return True

        elif self.extra_key_callback(key):  # , mode):
            return True
        else:
            return False

    def _on_key(self, window, key, scancode, action, mods):
        self.impl.keyboard_callback(window, key, scancode, action, mods)
        if not imgui.get_io().want_capture_keyboard:
            if (action == glfw.PRESS) or (action == glfw.REPEAT):
                self.on_key(key, mods)

    def _on_char(self, window, codepoint):
        self.impl.char_callback(window, codepoint)

    def _on_mouse_move(self, window, x, y):
        self.impl.mouse_callback(window, x, y)
        if self.mouse_last_pos is not None:
            self.motion_func(x, y)  # Viewer ftn

    def on_mouse_button(self, button, action, mods):
        if action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(self.window)
            self.mouse_last_pos = np.array([x, y])
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.pressed_button = 0
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.pressed_button = 2
        if action == glfw.RELEASE:  # and button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_last_pos = None
            self.pressed_button = None

    def _on_mouse_button(self, window, button, action, mods):
        if not imgui.get_io().want_capture_mouse:
            self.on_mouse_button(button, action, mods)

    def on_scroll(self, x, y):
        if y > 0:
            self.cam_cur.zoom(0.95)
        elif y < 0:
            self.cam_cur.zoom(1.05)

    def _on_scroll(self, window, xoffset, yoffset):
        self.impl.scroll_callback(window, xoffset, yoffset)
        if not imgui.get_io().want_capture_mouse:
            self.on_scroll(xoffset, yoffset)

    def _on_resize(self, window, width, height):
        self.impl.resize_callback(window, width, height)
        self.resize_GL(width, height)

    # def _on_joystick(self, jid, event):
    #     print(f'joysticK :: jid: {jid}\tevent:{event}')
    #     print(glfw.get_gamepad_name(0), glfw.get_gamepad_state(0))


if __name__ == "__main__":
    from fairmotion.viz import camera
    from fairmotion.utils import utils
    from fairmotion.data import bvh
    from mypath import *

    cam = camera.Camera(
        pos=np.array([2.0, 2.0, 2.0]),
        origin=np.array([0.0, 0.0, 0.0]),
        vup=utils.str_to_axis("y"),
        fov=45.0,
    )

    motions = [
        bvh.load(
            os.path.join(DATA_DIR, "sample/motion/bvh/lafan1/aiming1_subject1_0.bvh")
        )
    ]

    viewer = Viewer_imgui(
        motions=motions,
        joint_scale=4.5,
        link_scale=3.0,
        bvh_scale=0.01,
        render_overlay=True,
        hide_origin=False,
        title="Motion Viewer",
        cam=cam,
        size=(2560, 1440),
        # size = (1280, 720),
        use_msaa=True,
    )
    viewer.run()
