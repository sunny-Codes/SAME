import numpy as np
import os, random
from IPython import embed
from functools import partial

from mypath import *

# from task.motion_control.z_database import P_database
# from task.motion_control.z_mm_controller import P_mm_controller
from task.motion_control.z_database import Z_database
from task.motion_control.z_mm_controller import Z_mm_controller
from task.motion_control.control_world_stepper import run_screenless, prepare_run
from fairmotion.utils.contact_utils import Contact
from fairmotion.ops import conversions
from fairmotion.data import bvh
from utils import motion_utils

import imgui


def get_controller(
    db,
    search_interval,
    blend_type,
    blend_margin,
    foot_cleanup=False,
    dbg=False,
    **kwargs,
):
    print("db total frames : ", db.total_frames)
    for action in db.action2int.keys():
        action_frame = np.where(db.action_type_int == db.action2int[action])[0].shape[0]
        print(" - ", action, action_frame)

    act_weight_dicts = dict()
    traj_start_idx = db.features_size - db.traj_num * 4 - 2
    traj_dir_start_idx = db.features_size - db.traj_num * 2 - 2

    # base
    base_weights = np.ones(db.features_size)
    base_weights[:traj_start_idx] = 1.1
    # base_weights[traj_start_idx:traj_dir_start_idx] = 0.5*(db.features_size - db.traj_num*4)/(db.traj_num*4)
    # base_weights[traj_dir_start_idx: ] = 1.1
    base_weights[traj_start_idx:traj_dir_start_idx] = 0.14
    base_weights[traj_dir_start_idx:-2] = 0.28
    base_weights[-2:] = 1.1  # fc

    act_weight_dicts["base"] = base_weights

    crawl_weights = base_weights.copy()
    act_weight_dicts["crawl"] = crawl_weights

    jump_weights = np.ones(db.features_size)
    jump_weights[traj_start_idx:] = 0.0
    act_weight_dicts["jump"] = jump_weights

    jump_weights = np.ones(db.features_size)
    jump_weights[traj_start_idx:] = 0.0
    act_weight_dicts["kick"] = jump_weights

    jump_weights = np.ones(db.features_size)
    jump_weights[traj_start_idx:] = 0.0
    act_weight_dicts["kick_combo"] = jump_weights

    controller = Z_mm_controller(
        db,
        act_weight_dicts,
        ms_dict,
        control_mode="velocity",
        blend_type=blend_type,
        blend_margin=blend_margin,
        foot_cleanup=foot_cleanup,
        search_interval=search_interval,
        traj_num=db.traj_num,
        traj_interval=db.traj_interval,
        dbg=dbg,
        **kwargs,
    )

    """controller custom params
        property
            active_actions      str
        runtime - cur state
            set from 'set_goal_action' ftn: 
                prev_action         str
                goal_action         str
                action_set_frame    int
                action_done         bool - modified at 'post_step_strategy' ftn
            set from 'prev_step_strategy' ftn :
                search_action_list  list(str)
    """
    controller.active_actions = list()
    controller.search_limit_score = 100
    controller.jump_z_limit_score = 1.5
    controller.base_z_limit_score = 1.3
    controller.crawl_z_limit_score = 1.5
    controller.kick_combo_z_limit_score = 1.0

    def set_active_actions(active_action):
        controller.active_actions.append(active_action)

    def set_goal_action(goal_action_str, new_frame, init=False):
        controller.prev_action = goal_action_str if init else controller.goal_action
        controller.goal_action = goal_action_str
        controller.action_set_frame = new_frame
        controller.action_done = False
        controller.weights = controller.weight_dict[goal_action_str]
        # print(controller.goal_action)

    def get_search_limit():
        return controller.search_limit_score

    def get_transition_z_limit():
        if controller.goal_action == "jump":
            return controller.jump_z_limit_score
        elif controller.goal_action == "crawl":
            return controller.crawl_z_limit_score
        elif controller.goal_action == "base":
            return controller.base_z_limit_score
        # elif controller.goal_action == 'kick_combo':
        #     return controller.kick_combo_z_limit_score
        else:
            return controller.search_limit_score

    def prev_step_strategy(new_frame, release_time=0.6):
        cur_dbframe = (
            controller.db_frame_trace[-1]
            if controller.blend_weight_trace[-1] < 0.5
            else controller.db_frame_trace_blend[-1]
        )
        cur_action_int = int(controller.database.action_type_int[cur_dbframe])
        cur_action = controller.database.int2action[cur_action_int]
        active_goal = controller.goal_action in controller.active_actions

        if active_goal:
            # print("active_goal", cur_dbframe, controller.database.interaction_time[cur_dbframe, 0], release_time)
            controller.dbg_msg += f"active_goal! cur_dbframe: {cur_dbframe}, \tinteraction_time:{controller.database.interaction_time[cur_dbframe, 0]:.4f}\trelease_time:{release_time:.4f}\n"
            controller.action_done = (
                controller.database.interaction_time[cur_dbframe, 0] > release_time
            )
            if controller.action_done:
                controller.set_goal_action("base", new_frame)
                controller.dbg_msg += f"newframe: {new_frame}, action done\n"

        # adjust 'search_action_list' prior to stepping
        base_int = controller.database.action2int["base"]
        goal_int = controller.database.action2int[controller.goal_action]

        controller.search_action_list = [goal_int]
        if controller.prev_action == "base" and controller.goal_action == "crawl":
            controller.search_action_list.append(
                controller.database.action2int["transition"]
            )
        elif controller.prev_action == "crawl" and controller.goal_action == "base":
            controller.search_action_list.append(
                controller.database.action2int["transition"]
            )

        elif controller.prev_action == "base" and controller.goal_action == "kick":
            controller.search_action_list.append(
                controller.database.action2int["kick_transition"]
            )
        elif (
            controller.prev_action == "base" and controller.goal_action == "kick_combo"
        ):
            controller.search_action_list.append(
                controller.database.action2int["kick_combo_transition"]
            )

        elif (
            (controller.prev_action == "kick")
            or (controller.prev_action == "kick_combo")
        ) and (controller.goal_action == "base"):
            controller.search_action_list.append(
                controller.database.action2int["kick_transition"]
            )

        cur_dbframe = (
            controller.db_frame_trace[-1]
            if controller.blend_weight_trace[-1] < 0.5
            else controller.db_frame_trace_blend[-1]
        )
        cur_action_int = int(controller.database.action_type_int[cur_dbframe])
        cur_action = controller.database.int2action[cur_action_int]

        if controller.goal_action in controller.active_actions:
            # active action A -> active action B : allow dropping by base (only in the beginning)
            # [A -> base -> B] or [A-> B]
            if (
                (cur_action in controller.active_actions)
                and (controller.goal_action != cur_action)
                and (new_frame - controller.action_set_frame < 20)
            ):
                controller.search_action_list = [base_int, goal_int]

        if controller.dbg:
            controller.dbg_msg += f"prev strategy done, @{new_frame} ["
            for sa in controller.search_action_list:
                controller.dbg_msg += f"{controller.database.int2action[sa]}, "
            controller.dbg_msg += "]\n"

    def mark_frame_mask():
        frame_mask_idx = []
        database = controller.database

        # prevent transitioning to the last frame of each motion segment
        for fi, file_name, s_frame, e_frame in database.db_file_frames:
            frame_mask_idx += list(range(e_frame - blend_margin, e_frame))

        # force transitioning : CAUTION: PROBABLY NOT WORKING
        if len(controller.buffer_poses) < controller.blend_margin:
            frame_mask_idx += controller.buffer_db_frame_traces
            dbtrace = ",".join(
                [dbf for dbf in controller.buffer_db_frame_traces if dbf is not None]
            )
            controller.dbg_msg += f"mask buffer db frame traces; {dbtrace}\n"

        active_action_int = [
            database.action2int[aa]
            for aa in controller.active_actions
            if aa in database.action2int
        ]
        frames_inactive = np.array(
            [not (ati in active_action_int) for ati in database.action_type_int]
        )

        if len(controller.db_frame_trace_blend) > 5:
            last_blend = controller.db_frame_trace_blend[-1]
            if last_blend is not None:
                last_3_blend_trace = controller.db_frame_trace_blend[-3:]
                same_jump = (last_3_blend_trace[0] == last_3_blend_trace[1]) and (
                    last_3_blend_trace[1] == last_3_blend_trace[2]
                )
                remain_frame = database.get_remain_frame(last_blend)
                controller.dbg_msg += (
                    f"last_blend: {last_blend} / remain_frame: {remain_frame}\n"
                )
                if same_jump:
                    mask_begin = database.get_trajectory_index_clamp(
                        last_blend, -controller.blend_margin
                    )
                    mask_end = database.get_trajectory_index_clamp(
                        last_blend, +controller.blend_margin
                    )
                    frame_mask_idx.extend(list(range(mask_begin, mask_end)))

        if controller.search_action_list is not None:
            search_action_frames = []
            # if controller.goal_action == 'crawl': from IPython import embed; embed();

            for action_int in controller.search_action_list:
                search_action_frames += list(
                    np.where(action_int == database.action_type_int)[0]
                )

            # do not allow transitioning into interacting_frame > -0.7 (continue is handled in post-process(withdraw))
            transition_ok_frames = list(
                np.where((database.interaction_time < -0.7)[:, 0] | frames_inactive)
            )[
                0
            ]  #  (database.action_type_int == database.default_type))[0])

            ok_frames = set(search_action_frames).intersection(transition_ok_frames)
            frame_mask_idx += list(set(range(database.total_frames)) - ok_frames)

            # do not allow transitioning to frames that leads to interaction of unwanted actions
            for action_type, start, end in database.action_frame_list:
                if database.action2int[action_type] in controller.search_action_list:
                    continue
                prev_frame = database.get_trajectory_index_clamp(
                    start, -controller.search_interval - controller.blend_margin
                )
                frame_mask_idx += list(range(prev_frame, start))

        frame_mask_idx = list(set(frame_mask_idx))
        return frame_mask_idx

    def post_step_strategy(
        searched_frame, searched_distance, new_frame, transition_force
    ):
        cur_dbframe = (
            controller.db_frame_trace[-1]
            if controller.blend_weight_trace[-1] < 0.5
            else controller.db_frame_trace_blend[-1]
        )
        cur_action_int = int(controller.database.action_type_int[cur_dbframe])
        cur_action = controller.database.int2action[cur_action_int]
        active_goal = controller.goal_action in controller.active_actions

        if transition_force:
            trans_withdraw = False
            withdraw_msg = " withdraw (transition_force)"
        elif searched_distance > controller.get_search_limit():
            trans_withdraw = True
            withdraw_msg = " withdraw (exceed search limit)"
        elif (
            active_goal
            and (cur_action == controller.goal_action)
            and (not controller.action_done)
        ):
            trans_withdraw = True
            withdraw_msg = " withdraw (active action done)"
        else:
            trans_withdraw = False
            withdraw_msg = ""

        if controller.dbg:
            controller.dbg_msg += f"searched_frame: {searched_frame}, searched_distance: {searched_distance:.3f}, search_limit: {controller.get_search_limit():.3f}\n"
            controller.dbg_msg += (
                f"cur: {cur_action}, goal: {controller.goal_action} / {withdraw_msg}\n"
            )

        return trans_withdraw

    def loadGoalRecord(run_load_dir):
        controller.load_goal_vel = []
        controller.load_goal_rot = []
        controller.load_goal_action = []
        goal_load_path = os.path.join(run_load_dir, "goal.csv")
        with open(goal_load_path, "r") as file:
            for line_no, line in enumerate(file):
                words = line.split(",")
                words = [float(word) for word in words]
                controller.load_goal_vel.append(np.array(words[:3]))
                controller.load_goal_rot.append(np.array(words[3:12]).reshape(3, 3))
                controller.load_goal_action.append(
                    controller.database.int2action[int(words[12])]
                )
            controller.max_load_frame = line_no
            print(">>> controller.max_load_frame: ", controller.max_load_frame)

    def updateGoal_fromRecord(frame):
        controller.desired_vel = controller.load_goal_vel[frame]
        controller.desired_rot = controller.load_goal_rot[frame]
        controller.goal_action = controller.load_goal_action[frame]

    def updateGoalRandom(frame):
        # default
        release_time = 0.6
        prev_active_goal = controller.goal_action in controller.active_actions
        cur_dbframe = (
            controller.db_frame_trace[-1]
            if controller.blend_weight_trace[-1] < 0.5
            else controller.db_frame_trace_blend[-1]
        )
        action_done = (
            controller.database.interaction_time[cur_dbframe, 0] > release_time
        )
        if prev_active_goal and (not action_done):
            return

        angle = -np.pi + 2 * random.random() * np.pi
        speed = controller.min_speed + random.random() * (
            controller.max_speed - controller.min_speed
        )
        controller.updateGoal(speed, angle)
        # custom
        # print(list(controller.database.action2int.keys()))
        # rnd_action = random.choice(list(controller.database.action2int.keys()))

        rnd_action = "base"
        if frame > 200:
            rnd_p = random.random()
            if rnd_p < 0.1:
                rnd_action = "kick_combo"
            elif rnd_p < 0.2:
                rnd_action = "kick"
            elif controller.goal_action == "base" and rnd_p < 0.3:
                rnd_action = "crawl"

        if rnd_action == "crawl":
            cur_rot_y = conversions.R2A(controller.desired_rot)[1]
            angle = cur_rot_y + 2 * random.random() * 0.3 * np.pi
            print(frame, speed)
            controller.updateGoal(speed, angle)
            # conversions.Ay2R(conversions.R2A(controller.desired_rot)[1]) == controller.desired_rot

        print("RND", frame, rnd_action)
        # print(f'>>>>>>>>>>>>>>>>>>>>>>>>>> {frame:4d} | angle {angle:4f}, speed {speed:4f}, {rnd_action}')
        controller.dbg_msg += f">>>>>>>>>>>>>>>>>>>>>>>>>> {frame:4d} | angle {angle:4f}, speed {speed:4f}, {rnd_action}\n"
        controller.set_goal_action(rnd_action, frame)

    # set custom functions
    controller.set_active_actions = set_active_actions
    controller.get_search_limit = get_search_limit
    controller.get_transition_z_limit = get_transition_z_limit
    controller.set_goal_action = set_goal_action
    controller.prev_step_strategy = prev_step_strategy
    controller.post_step_strategy = post_step_strategy
    controller.mark_frame_mask = mark_frame_mask
    controller.updateGoalRandom = updateGoalRandom

    controller.database.default_type = controller.database.action2int["base"]
    controller.set_active_actions("jump")
    controller.set_active_actions("kick")
    controller.set_active_actions("kick_combo")

    # init with base
    controller.set_goal_action("base", 0, init=True)

    # save / load
    controller.loadGoalRecord = loadGoalRecord
    controller.updateGoal_fromRecord = updateGoal_fromRecord
    controller.max_load_frame = 0

    return controller


def get_character(bvh_fp):
    motion = bvh.load(bvh_fp)
    n_motion, n_tpose = motion_utils.motion_normalize_h2s(motion)
    skel = n_motion.skel
    return motion, n_motion, skel


if __name__ == "__main__":
    import argparse, torch
    from same.test import prepare_model_test

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_epoch", type=str, default="ckpt0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="motion_control")
    #
    parser.add_argument("--screenless", type=int, default=0)
    parser.add_argument("--dbg", type=int, default=1)
    parser.add_argument("--construct_db", type=int, default=0)
    parser.add_argument("--traj_num", type=int, default=6)
    parser.add_argument("--traj_interval", type=int, default=5)

    # run related arguments
    parser.add_argument("--search_interval", type=int, default=10)
    parser.add_argument("--blend_type", type=str, default="slerp")
    parser.add_argument("--blend_margin", type=int, default=8)
    parser.add_argument("--foot_cleanup", type=int, default=0)

    parser.add_argument("--command", type=str, default="auto")
    parser.add_argument("--load_run_dir", type=str)
    parser.add_argument("--demo", type=str, default="")

    args = parser.parse_args()
    model, cfg, ms_dict = prepare_model_test(args.model_epoch, args.device)
    torch.manual_seed(404)
    np.random.seed(404)
    random.seed(404)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # if CTRL_TYPE == 'p_mm':
    #     parser = argparse.ArgumentParser()
    #     add_mm_args(parser)
    #     args = parser.parse_args()

    #     database = get_p_db(data_dir, ann_dir_path, db_file_name, args.construct_db)
    #     controller = get_controller('p_mm', database, \
    #         args.search_interval, args.blend_type, args.blend_margin, args.foot_cleanup, args.dbg)

    db = Z_database()
    data_dir = os.path.join(DATA_DIR, args.data_dir)
    db_save_fp = os.path.join(data_dir, "zmm_db.pkl")
    if args.construct_db:
        db.construct_database(model, data_dir, args.traj_num, args.traj_interval)
        db.save_db(db_save_fp)
    else:
        db.load_db(db_save_fp)

    motion_megan, n_motion_megan, skel_megan = get_character(
        os.path.join(data_dir, "megan.bvh")
    )
    controller = get_controller(
        db,
        args.search_interval,
        args.blend_type,
        args.blend_margin,
        args.foot_cleanup,
        args.dbg,
        model=model,
        tgt_skel=skel_megan,
        tgt_skel_motion=motion_megan,
        save_dir=os.path.join(DATA_DIR, data_dir, "runs"),
    )

    motion_mousey, n_motion_mousey, skel_mousey = get_character(
        os.path.join(data_dir, "mousey.bvh")
    )
    motion_ty, n_motion_ty, skel_ty = get_character(os.path.join(data_dir, "Ty.bvh"))

    if args.load_run_dir is not None:
        zmm_path = os.path.join(DATA_DIR, data_dir, "runs", args.load_run_dir)
        controller.loadGoalRecord(zmm_path)

    if args.screenless:
        run_screenless(controller)
    else:
        viewer, controller, command = prepare_run(controller, args.command)

        def mm_ui():
            imgui.begin("MM Parameters", True)

            _, controller.sim_vel_halflife = imgui.slider_float(
                "sim_vel_halflife",
                controller.sim_vel_halflife,
                min_value=0.0,
                max_value=1.0,
                format="%.02f",
            )

            _, controller.sim_rot_halflife = imgui.slider_float(
                "sim_rot_halflife",
                controller.sim_rot_halflife,
                min_value=0.0,
                max_value=1.0,
                format="%.02f",
            )

            _, controller.sim_run_fwrd_speed = imgui.slider_float(
                "sim_run_fwrd_speed",
                controller.sim_run_fwrd_speed,
                min_value=1.0 * 100,
                max_value=5.0 * 100,
                format="%.02f",
            )

            _, controller.sim_run_side_speed = imgui.slider_float(
                "sim_run_side_speed",
                controller.sim_run_side_speed,
                min_value=1.0 * 100,
                max_value=5.0 * 100,
                format="%.02f",
            )

            _, controller.sim_run_back_speed = imgui.slider_float(
                "sim_run_back_speed",
                controller.sim_run_back_speed,
                min_value=1.0 * 100,
                max_value=5.0 * 100,
                format="%.02f",
            )

            _, controller.sim_walk_fwrd_speed = imgui.slider_float(
                "sim_walk_fwrd_speed",
                controller.sim_walk_fwrd_speed,
                min_value=1.0 * 100,
                max_value=5.0 * 100,
                format="%.02f",
            )

            _, controller.sim_walk_side_speed = imgui.slider_float(
                "sim_walk_side_speed",
                controller.sim_walk_side_speed,
                min_value=1.0 * 100,
                max_value=5.0 * 100,
                format="%.02f",
            )

            _, controller.sim_walk_back_speed = imgui.slider_float(
                "sim_walk_back_speed",
                controller.sim_walk_back_speed,
                min_value=1.0 * 100,
                max_value=5.0 * 100,
                format="%.02f",
            )

            _, controller.synch_pos_factor = imgui.slider_float(
                "synch_pos_factor",
                controller.synch_pos_factor,
                min_value=0.0,
                max_value=1.0,
                format="%.02f",
            )
            _, controller.synch_rot_factor = imgui.slider_float(
                "synch_rot_factor",
                controller.synch_rot_factor,
                min_value=0.0,
                max_value=1.0,
                format="%.02f",
            )
            _, controller.search_limit_score = imgui.slider_float(
                "search_limit_score",
                controller.search_limit_score,
                min_value=0.0,
                max_value=100.0,
                format="%.02f",
            )
            _, controller.z_ = imgui.slider_float(
                "base_z_limit_score",
                controller.base_z_limit_score,
                min_value=0.0,
                max_value=5.0,
                format="%.02f",
            )
            _, controller.z_ = imgui.slider_float(
                "jump_z_limit_score",
                controller.jump_z_limit_score,
                min_value=0.0,
                max_value=5.0,
                format="%.02f",
            )
            imgui.end()

            imgui.begin("MM weights", True)

            if hasattr(controller.database, "key_pos_joints"):
                kp_len = len(controller.database.key_pos_joints)
                for i in range(kp_len):
                    _, controller.weight_dict["base"][3 * i : 3 * (i + 1)] = (
                        imgui.slider_float(
                            f"kp_{i}",
                            controller.weight_dict["base"][3 * i],
                            min_value=0.0,
                            max_value=10.0,
                            format="%.02f",
                        )
                    )

            if hasattr(controller.database, "key_vel_joints"):
                kv_len = len(controller.database.key_vel_joints)
                for i in range(kv_len):
                    (
                        _,
                        controller.weight_dict["base"][
                            3 * kp_len + 3 * i : 3 * kp_len + 3 * (i + 1)
                        ],
                    ) = imgui.slider_float(
                        f"kv_{i}",
                        controller.weight_dict["base"][3 * kp_len + 3 * i],
                        min_value=0.0,
                        max_value=10.0,
                        format="%.02f",
                    )
            if hasattr(controller, "z_dim"):
                z_dim = controller.z_dim
                _, controller.weight_dict["base"][:z_dim] = imgui.slider_float(
                    f"Z",
                    controller.weight_dict["base"][0],
                    min_value=0.0,
                    max_value=10.0,
                    format="%.02f",
                )

            traj_start_idx = (
                controller.database.features_size - controller.traj_num * 4 - 2
            )
            traj_dir_start_idx = (
                controller.database.features_size - controller.traj_num * 2 - 2
            )
            _, controller.weight_dict["base"][traj_start_idx:traj_dir_start_idx] = (
                imgui.slider_float(
                    f"traj pos",
                    controller.weight_dict["base"][traj_start_idx],
                    min_value=0.0,
                    max_value=10.0,
                    format="%.02f",
                )
            )
            _, controller.weight_dict["base"][traj_dir_start_idx:-2] = (
                imgui.slider_float(
                    f"traj dir",
                    controller.weight_dict["base"][traj_dir_start_idx],
                    min_value=0.0,
                    max_value=10.0,
                    format="%.02f",
                )
            )
            _, controller.weight_dict["base"][-2:] = imgui.slider_float(
                f"foot contact",
                controller.weight_dict["base"][-2],
                min_value=0.0,
                max_value=10.0,
                format="%.02f",
            )

            imgui.end()

            imgui.begin("Contact Cleanup Params", True)
            _, Contact.ik_max_length_buffer = imgui.slider_float(
                "ik_max_length_buffer",
                Contact.ik_max_length_buffer,
                min_value=0.0,
                max_value=5.0,
                format="%.02f",
            )
            _, Contact.ik_unlock_radius = imgui.slider_float(
                "ik_unlock_radius",
                Contact.ik_unlock_radius,
                min_value=0.0,
                max_value=10.0,
                format="%.02f",
            )
            _, Contact.ik_blending_halflife = imgui.slider_float(
                "ik_blending_halflife",
                Contact.ik_blending_halflife,
                min_value=0.0,
                max_value=20.0,
                format="%.02f",
            )
            _, Contact.contact_label_thres = imgui.slider_float(
                "contact_label_thres",
                Contact.contact_label_thres,
                min_value=0.0,
                max_value=1.0,
                format="%.02f",
            )
            _, controller.foot_cleanup = imgui.checkbox(
                "foot_cleanup", controller.foot_cleanup
            )
            _, vis_dbg_update = imgui.checkbox("vis_dbg", controller.vis_dbg)
            if controller.vis_dbg != vis_dbg_update:
                if hasattr(controller, "dbg_motion") and vis_dbg_update:
                    viewer.update_motions(
                        [controller.motion, controller.dbg_motion], reset=False
                    )
                else:
                    viewer.update_motions([controller.motion], reset=False)
                controller.vis_dbg = vis_dbg_update

            blend_type_list = ["none", "slerp"]
            _, controller.blend_type_int = imgui.combo(
                "blend type", controller.blend_type_int, blend_type_list
            )
            controller.blend_type = blend_type_list[controller.blend_type_int]

            _, controller.blend_margin = imgui.slider_int(
                "blend margin",
                controller.blend_margin,
                min_value=0,
                max_value=10,
                format="%.02d",
            )

            _, trackCamera = imgui.checkbox("track Camera", viewer.trackCamera)
            if trackCamera != viewer.trackCamera:
                viewer.set_trackCamera(trackCamera)

            imgui.end()

        viewer.ui_ftns.append(mm_ui)

        if args.demo == "skel_change":
            to_mousey_start = 480
            to_mousey_end = to_mousey_start + 100

            to_mousey_change_length = to_mousey_end - to_mousey_start

            to_ty_start = 650
            to_ty_end = to_ty_start + 120
            to_ty_change_length = to_ty_end - to_ty_start

            controller.start_character_dict = {}
            controller.start_character_dict[0] = motion_megan
            controller.start_character_dict[to_mousey_end] = motion_mousey
            controller.start_character_dict[to_ty_end] = motion_ty

            def extra_step(frame):
                if (frame >= to_mousey_start) and (frame <= to_mousey_end):
                    skel_new = motion_utils.skel_interpolate(
                        skel_megan,
                        skel_mousey,
                        (frame - to_mousey_start) / (to_mousey_change_length),
                    )
                    controller.change_skel(skel_new, frame)

                if (frame >= to_ty_start) and (frame <= to_ty_end):
                    skel_new = motion_utils.skel_interpolate(
                        skel_mousey,
                        skel_ty,
                        (frame - to_ty_start) / (to_ty_change_length),
                    )
                    controller.change_skel(skel_new, frame)

            controller.extra_step = extra_step

        viewer.run()
