from fairmotion.viz.bvh_visualizer import MocapViewer
from fairmotion.utils import utils
import argparse, os
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from fairmotion.viz import camera
from fairmotion.data import bvh
from fairmotion.utils.contact_utils import get_all_joint_ground_contact
from fairmotion.viz import gl_render
from mypath import *

""" IMGUI """
from imgui_viewer import Viewer_imgui
from utils.imgui_utils import list_motion_widget
import glfw


def valid_key(args, key):
    if hasattr(args, key):
        return getattr(args, key) is not None
    else:
        return False


def set_default_key(args, key, def_val):
    if not hasattr(args, key):
        setattr(args, key, def_val)


def get_default_viewer(args=argparse.Namespace()):
    # copied from args
    set_default_key(args, "bvh_files", None)
    set_default_key(args, "bvh_dir", None)
    set_default_key(args, "retarget_dir", None)
    set_default_key(args, "npz_path", None)
    set_default_key(args, "lazy", 0)
    set_default_key(args, "inc_motion_num", 1)
    set_default_key(args, "vis_motion_num", 1)
    set_default_key(args, "grid_size", 0.0)
    set_default_key(args, "imgui", False)

    set_default_key(args, "one_each", 1)

    cam = camera.Camera(
        pos=np.array([2.0, 2.0, 2.0]),
        origin=np.array([0.0, 0.0, 0.0]),
        vup=utils.str_to_axis("y"),
        fov=45.0,
    )

    viewer_class = Viewer_imgui if args.imgui else MocapViewer

    viewer = viewer_class(
        motions=[],
        joint_scale=4.5,
        link_scale=3.0,
        bvh_scale=0.01,
        render_overlay=True,
        hide_origin=False,
        title="Motion Viewer",
        cam=cam,
        size=(2560, 1440),
        # size = (1280, 720),
        # use_msaa = False,
        use_msaa=True,
    )

    viewer.all_motions = []
    viewer.file_names = []
    viewer.file_names = []

    if valid_key(args, "bvh_dir"):
        viewer.file_names = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(args.bvh_dir)
            for f in filenames
        ]
        viewer.file_names = list(
            filter(lambda x: x.endswith(".bvh"), viewer.file_names)
        )

    if valid_key(args, "bvh_files"):
        viewer.file_names += args.bvh_files

    if valid_key(args, "retarget_dir"):
        retarget_dir_path = os.path.join(DATA_DIR, args.retarget_dir)
        orig_path = os.path.join(retarget_dir_path, "input/")
        result_path = os.path.join(retarget_dir_path, "result/")
        result_log_path = os.path.join(retarget_dir_path, "result_log.txt")

        # add file_paths in order that retargeted files are listed in a row (consecutive order)
        # assume we have same number of retargeted files for each motion (ex: N motions X M variations)
        path_a_id_dict = dict()
        file_pairs_list = []
        with open(result_log_path, "r") as f:
            for line in f:
                words = line.rstrip().split(", ")
                batch_i, charname_a, basename_a, skel_i, basename_b = words
                basename_b = basename_b + ".bvh"

                path_a = os.path.join(orig_path, charname_a, basename_a)
                path_b = os.path.join(result_path, basename_b)
                if path_a in path_a_id_dict.keys():
                    path_a_id = path_a_id_dict[path_a]
                    file_pairs_list[path_a_id].append(path_b)
                else:
                    path_a_id = len(file_pairs_list)
                    path_a_id_dict[path_a] = path_a_id
                    file_pairs_list.append([path_a, path_b])
        for file_pairs in file_pairs_list:
            viewer.file_names.extend(file_pairs)

    # print(viewer.file_names)
    if valid_key(args, "lazy") and args.lazy:
        viewer.all_motions += [None] * len(viewer.file_names)
    else:
        viewer.all_motions += [bvh.load(file_name) for file_name in viewer.file_names]

    viewer.vis_motion_num = args.vis_motion_num
    if args.inc_motion_num < 0:  # -1 means visualize all at once
        if args.lazy:
            viewer.all_motions = [bvh.load(fn) for fn in viewer.file_names]
        viewer.update_motions(viewer.all_motions, update_vis_num=False)

    else:
        num_motions = len(viewer.all_motions)
        viewer.m_ids = list(range(min(args.inc_motion_num, num_motions)))
        for m_id in viewer.m_ids:
            if viewer.all_motions[m_id] is None:
                viewer.all_motions[m_id] = bvh.load(viewer.file_names[m_id])
            print(
                m_id,
                "\t",
                viewer.file_names[m_id],
                "\t",
                viewer.all_motions[m_id].num_frames() - 1,
            )
        viewer.update_motions(
            [viewer.all_motions[m_id] for m_id in viewer.m_ids],
            grid_size=args.grid_size,
            update_vis_num=False,
        )

        def extra_key_callback(key, mode=None):
            if (key == b"m") or (key == glfw.KEY_M):
                if num_motions - viewer.m_ids[-1] > args.inc_motion_num:
                    viewer.m_ids = [
                        (m_id + args.inc_motion_num) for m_id in viewer.m_ids
                    ]
                elif num_motions - viewer.m_ids[-1] <= args.inc_motion_num:
                    viewer.m_ids = list(
                        range(num_motions - args.inc_motion_num, num_motions)
                    )
                elif num_motions - viewer.m_ids[-1] == 1:
                    viewer.m_ids = list(range(min(args.inc_motion_num, num_motions)))
            elif (key == b"n") or (key == glfw.KEY_N):
                if viewer.m_ids[0] == 0:
                    viewer.m_ids = list(
                        range(num_motions - args.inc_motion_num, num_motions)
                    )
                else:
                    viewer.m_ids = [
                        (m_id - args.inc_motion_num) for m_id in viewer.m_ids
                    ]
            else:
                return False

            for m_id in viewer.m_ids:
                if viewer.all_motions[m_id] is None:
                    viewer.all_motions[m_id] = bvh.load(viewer.file_names[m_id])
                if not hasattr(viewer.all_motions[m_id], "alljoint_contact"):
                    viewer.all_motions[m_id].alljoint_contact = (
                        get_all_joint_ground_contact(viewer.all_motions[m_id])
                    )
                print(
                    m_id,
                    "\t",
                    viewer.file_names[m_id],
                    "\t",
                    viewer.all_motions[m_id].num_frames() - 1,
                )
            print()
            viewer.update_motions(
                [viewer.all_motions[m_id] for m_id in viewer.m_ids],
                grid_size=args.grid_size,
                update_vis_num=False,
            )
            return True

        def extra_render_callback():
            pass

        viewer.extra_key_callback = extra_key_callback
        viewer.extra_render_callback = extra_render_callback

        def overlay_callback():  # overlay test

            display_text = f"Frame: {viewer.worldStepper.cur_frame}\n"
            for m_id in viewer.m_ids:
                display_text += f"{viewer.file_names[m_id]}\n"

            w, h = viewer.window_size
            display_text = display_text.split("\n")
            for i, text in enumerate(display_text):
                gl_render.render_text(
                    text,
                    pos=[0.02 * w, 0.02 * (i + 1) * h],
                    font=GLUT_BITMAP_HELVETICA_18,
                )

            if viewer.extra_overlay_callback is not None:
                viewer.extra_overlay_callback()

        #     img_path = os.path.join(SRC_DIR, f'pca.png')
        #     if os.path.exists(img_path):
        #         gl_render.render_img_on_window_from_path(img_path)

        viewer.overlay_callback = overlay_callback

    return viewer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # motion sources
    parser.add_argument("--bvh_files", type=str, nargs="+", default=None)
    parser.add_argument("--bvh_dir", type=str, default=None)
    parser.add_argument("--retarget_dir", type=str, default=None)
    parser.add_argument("--npz_path", type=str, default=None)
    # motion load options
    parser.add_argument("--lazy", type=int, default=0)
    # visualize options
    parser.add_argument("--inc_motion_num", type=int, default=1)
    parser.add_argument("--vis_motion_num", type=int, default=1)
    parser.add_argument("--grid_size", type=float, default=0.0)
    parser.add_argument("--imgui", type=bool, default=False)
    args = parser.parse_args()
    viewer = get_default_viewer(args)

    # TODO: imgui currently lists only a single motion, should change to multi list
    if args.imgui:

        def set_imgui_variables():
            pass

        def ui():
            if viewer.imgui_setup_done:
                list_motion_widget(viewer)

        viewer.set_imgui_variables = set_imgui_variables
        viewer.ui = ui

    viewer.run()
