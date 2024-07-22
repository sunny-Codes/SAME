from collections import defaultdict
from email.policy import default
import imgui, os
from fairmotion.data import bvh
from IPython import embed


# TODO: list multiple motions (currently, temp fix; viewer.m_id = viewer.m_ids[0])
def update_motion_by_mid(viewer, mid, ignore_root_skel, ee_as_joint):
    viewer.m_id = viewer.m_ids[0]
    if mid != viewer.m_id:
        viewer.m_id = mid
        viewer.worldStepper.reset()
        if viewer.all_motions[viewer.m_id] is None:
            viewer.all_motions[viewer.m_id] = bvh.load(
                viewer.file_names[viewer.m_id],
                ignore_root_skel=ignore_root_skel,
                ee_as_joint=ee_as_joint,
            )
        new_motion = viewer.all_motions[viewer.m_id]
        viewer.update_motions([new_motion])
        print(viewer.file_names[viewer.m_id][:-4], "\t", new_motion.num_frames() - 1)


def list_motion_widget(viewer, ignore_root_skel, ee_as_joint):
    # motion list
    viewer.m_id = viewer.m_ids[0]
    if imgui.button(" < "):
        new_mid = (viewer.m_id + 1) % len(viewer.all_motions)
        update_motion_by_mid(viewer, new_mid, ignore_root_skel, ee_as_joint)

    imgui.same_line()

    _, new_mid = imgui.combo("", viewer.m_id, viewer.file_names)
    update_motion_by_mid(viewer, new_mid, ignore_root_skel, ee_as_joint)

    imgui.same_line()
    imgui.text(f"{viewer.m_id}/{len(viewer.file_names)}")
    imgui.same_line()

    if imgui.button(" > "):
        new_mid = (viewer.m_id + 1) % len(viewer.all_motions)
        update_motion_by_mid(viewer, new_mid, ignore_root_skel, ee_as_joint)


def check_setup(viewer, save_dir="", loaded_check=None):
    viewer.all_check_button_check = True
    if loaded_check is None:
        viewer.checkbox_enabled = [False] * len(viewer.file_names)
    else:
        assert len(loaded_check) == len(viewer.file_names)
        viewer.checkbox_enabled = loaded_check

    viewer.save_dir = save_dir


def check_widget(viewer, ee_as_joint):
    # select / deselect all (contact solved)
    if viewer.all_check_button_check:
        if imgui.button("select"):
            viewer.all_check_button_check = False
            for i in range(len(viewer.checkbox_enabled)):
                viewer.checkbox_enabled[i] = True
    else:
        if imgui.button("deselect"):
            viewer.all_check_button_check = True
            for i in range(len(viewer.checkbox_enabled)):
                viewer.checkbox_enabled[i] = False

    # check if contact ok (solved)
    for i, fn_i in enumerate(viewer.file_names):
        _, viewer.checkbox_enabled[i] = imgui.checkbox(
            os.path.basename(fn_i), viewer.checkbox_enabled[i]
        )

    changed, text_val = imgui.input_text("save dir", viewer.save_dir, 60)
    if changed:
        viewer.save_dir = text_val

    if imgui.button(" save motions "):
        if not os.path.exists(viewer.save_dir):
            os.makedirs(viewer.save_dir)
        try:
            for m_id in range(len(viewer.all_motions)):
                if viewer.checkbox_enabled[m_id] and (
                    viewer.all_motions[m_id] is not None
                ):
                    file_path = os.path.join(viewer.save_dir, viewer.file_names[m_id])
                    bvh.save(
                        viewer.all_motions[m_id],
                        file_path,
                        rot_order="ZXY",
                        verbose=True,
                        ee_as_joint=ee_as_joint,
                    )
        except:
            print("Err while saving")
            embed()


def tag_setup(viewer, save_dir):
    viewer.save_dir = save_dir
    viewer.cur_tag = ""
    viewer.tag_dicts = defaultdict(set)


def tag_widget(viewer):
    _, viewer.cur_tag = imgui.input_text("tag", viewer.cur_tag, 60)
    viewer.m_id = viewer.m_ids[0]
    if imgui.button("mark tag"):
        viewer.tag_dicts[viewer.cur_tag].add(viewer.file_names[viewer.m_id])

    if imgui.button("remove tag"):
        viewer.tag_dicts[viewer.cur_tag].remove(viewer.file_names[viewer.m_id])

    imgui.separator()

    _, viewer.save_dir = imgui.input_text("save dir", viewer.save_dir, 60)
    if imgui.button(" save tags "):
        if not os.path.exists(viewer.save_dir):
            os.makedirs(viewer.save_dir)
        try:
            for tag_key in viewer.tag_dicts:
                file_path = os.path.join(viewer.save_dir, tag_key + ".txt")
                print(file_path)
                with open(file_path, "w") as file:
                    for item in viewer.tag_dicts[tag_key]:
                        file.write(item + "\n")
        except:
            print("Err while saving")
            embed()

    imgui.separator()

    for tag_key in viewer.tag_dicts:
        if imgui.tree_node(tag_key, imgui.TREE_NODE_DEFAULT_OPEN):
            for elem in viewer.tag_dicts[tag_key]:
                imgui.text(os.path.basename(elem))
            imgui.tree_pop()
