from mypath import *

from IPython import embed
import numpy as np
import os, time, pickle, copy, torch, random, copy, sys, datetime

from fairmotion.data import bvh
from fairmotion.core.motion import Motion
from fairmotion.ops import motion as motion_ops
from fairmotion.ops import conversions

from fairmotion.utils.spring_utils import *
from fairmotion.utils.contact_utils import Contact, init_contact, pose_foot_cleanup
from utils import motion_utils

from task.motion_control.controller import MotionController
from utils.skel_gen_utils import create_random_skel
from conversions.motion_to_graph import skel_2_graph
from utils.motion_utils import motion_denormalize
from fairmotion.ops.motion import translate

from same.test import decode_z_skelgraph


class Z_mm_controller(MotionController):
    def __init__(
        self,
        database,
        weight_dict,
        ms_dict,
        init_idx=None,
        control_mode="position",
        blend_type="none",
        blend_margin=0,
        foot_cleanup=False,
        dbg=False,
        search_interval=10,
        traj_num=3,
        traj_interval=10,
        model=None,
        tgt_skel=None,
        tgt_skel_motion=None,
        save_dir=None,
    ) -> None:
        super(Z_mm_controller, self).__init__(foot_cleanup=foot_cleanup, dbg=dbg)

        self.model = model
        self.z_dim = self.model.z_dim
        self.save_dir = save_dir
        self.ms_dict = ms_dict

        # strategy related
        self.database = database
        self.search_interval = search_interval
        self.weight_dict = weight_dict
        self.blend_type = blend_type
        self.blend_type_int = 1
        self.blend_margin = blend_margin
        self.control_mode = control_mode
        self.traj_num = traj_num
        self.traj_interval = traj_interval
        self.base_z_limit_score = 1.3  # 5.0

        # running related
        self.reset(init_idx, tgt_skel, tgt_skel_motion)
        self.extra_step = None
        self.vis_dbg = False

    def reset(self, init_frame=None, skel=None, tgt_skel_motion=None):
        # init motion
        self.skel = skel
        if skel is None:
            self.skel = create_random_skel()
        if tgt_skel_motion is not None:
            self.tgt_skel_motion = tgt_skel_motion
        self.skel_graph = skel_2_graph(self.skel)
        self.feet_idx = []
        self.toe_chains = []
        links = motion_utils.get_links(self.skel)
        for link in links:
            if self.skel.get_joint(link[-1]).xform_global[1, 3] < 5:  # TODO FIX
                # self.feet_idx.extend(link[-2:])
                print("leg links:", link)
                self.feet_idx.append(link[4])  # toe
                self.toe_chains.append(link[:4])  # root-upleg-leg-foot
        print("feet idx: ", self.feet_idx)
        # self.feet_idx = np.array(self.feet_idx)

        self.motion = Motion(skel=self.skel)
        self.dbg_motion = Motion(skel=self.skel)
        if init_frame is None:
            ok_frames = []
            for file_i, file_name, start, end in self.database.db_file_frames:
                ok_frames += range(start + self.blend_margin, end - self.blend_margin)
            init_frame = random.choice(ok_frames)
            init_frame = 15741  # standing still

        init_zs = self.database.get_z_from_db_frame(
            init_frame, self.search_interval + self.blend_margin
        )
        out_motion, (feet_idx, out_contact) = decode_z_skelgraph(
            self.model,
            torch.Tensor(init_zs).to(device=self.model.device),
            self.skel_graph,
            self.ms_dict,
        )
        init_poses = out_motion.poses
        init_T = init_poses[0].get_root_facing_transform_byRoot(use_height=False)
        motion_ops.transform_poses(
            init_poses, np.linalg.inv(init_T), pivot=0, local=False
        )
        out_contact = out_contact[:, self.feet_idx, 0]
        out_contact = np.minimum(np.maximum(out_contact, 0.0), 1.0)

        ifn = 1  # init_frame_num
        self.motion.poses = init_poses[:ifn]
        self.dbg_motion.poses = init_poses[:ifn]
        self.zs_trace = init_zs[:ifn]
        self.db_frame_trace = list(range(init_frame, init_frame + ifn))
        self.db_frame_trace_blend = [None] * ifn
        self.blend_weight_trace = [0] * ifn
        self.fc_trace = out_contact[:ifn]

        self.buffer_poses = init_poses[ifn:]
        self.buffer_zs_trace = init_zs[ifn:]
        self.buffer_db_frame_traces = list(
            range(init_frame + ifn, init_frame + len(init_poses))
        )
        self.buffer_db_frame_traces_blend = [None] * len(self.buffer_db_frame_traces)
        self.buffer_blend_weight_trace = [0] * len(self.buffer_db_frame_traces)
        self.buffer_fc = out_contact[ifn:]

        self.spring_init()
        # self.updateGoal((self.min_speed+self.max_speed)/2., 0)
        self.search_action_list = ["base"]
        self.weights = self.weight_dict["base"]

        """ TODO : handle foot contact & cleanup """
        init_contact(
            self.motion, 0, self.feet_idx
        )  # [fid.item() for fid in self.feet_idx])

    def create_query(self):
        query = np.zeros(self.database.features_size)

        traj_start_idx = self.database.features_size - self.traj_num * 4 - 2
        traj_dir_start_idx = self.database.features_size - self.traj_num * 2 - 2

        query[:traj_start_idx] = self.zs_trace[-1]
        query[-2:] = self.fc_trace[-1]

        # pose = self.motion.poses[-1]
        pose = self.buffer_poses[0]
        T = pose.get_root_facing_transform_byRoot(use_height=False)
        for i in range(self.traj_num):
            traj_x_i_local = (np.linalg.inv(T) @ conversions.p2T(self.traj_pos[i]))[
                :3, 3
            ]
            traj_r_i_local = (np.linalg.inv(T[:3, :3]) @ self.traj_rot[i])[
                :3, 2
            ]  # z-axis

            query[traj_start_idx + i * 2 : traj_start_idx + (i + 1) * 2] = (
                traj_x_i_local[[0, 2]]
            )
            query[traj_dir_start_idx + i * 2 : traj_dir_start_idx + (i + 1) * 2] = (
                traj_r_i_local[[0, 2]]
            )
        return query

    def update_contact(self, frame):
        pose = self.motion.poses[-1]
        fc = self.fc_trace[-1]
        if self.foot_cleanup:
            pose_foot_cleanup(pose, self.motion.contacts, fc, self.toe_chains, self.dt)

    def step(self, new_frame):
        self.dbg_msg = ""  # reset

        assert len(self.buffer_poses) != 0, f"{new_frame}, buffer_poses empty"
        assert (
            len(self.buffer_db_frame_traces) != 0
        ), f"{new_frame}, buffer_db_frame_traces empty"
        assert (
            len(self.buffer_db_frame_traces_blend) != 0
        ), f"{new_frame}, buffer_db_frame_traces_blend empty"

        self.traj_update()
        self.motion.poses.append(self.buffer_poses.pop(0))
        self.dbg_motion.poses.append(copy.deepcopy(self.motion.poses[-1]))
        self.db_frame_trace.append(self.buffer_db_frame_traces.pop(0))
        self.db_frame_trace_blend.append(self.buffer_db_frame_traces_blend.pop(0))
        self.blend_weight_trace.append(self.buffer_blend_weight_trace.pop(0))
        self.zs_trace = np.vstack((self.zs_trace, self.buffer_zs_trace[0:1]))
        self.buffer_zs_trace = self.buffer_zs_trace[1:]  # pop
        self.fc_trace = np.vstack((self.fc_trace, self.buffer_fc[0:1]))
        self.buffer_fc = self.buffer_fc[1:]  # pop

        new_db_frame = self.getDBFrame(new_frame)
        end_of_file = (
            self.database.get_trajectory_index_clamp(new_db_frame, 1) == new_db_frame
        )
        first_search = True

        # self.search = False
        if len(self.buffer_poses) <= self.blend_margin:
            self.prev_step_strategy(new_frame)  # leave space for customized ftn

            query = self.create_query()
            transition_force = len(self.buffer_poses) < self.blend_margin
            frame_mask_idx = self.mark_frame_mask()
            traj_start_idx = self.database.features_size - self.traj_num * 4 - 2
            traj_dir_start_idx = self.database.features_size - self.traj_num * 2 - 2

            while first_search or (end_of_file and (not transition)):

                next_frame, distance, weighted_feature_diff = (
                    self.database.search_closest(
                        query,
                        weights=self.weights,
                        frame_mask_idx=frame_mask_idx,
                        transition_z_limit=self.get_transition_z_limit(),
                    )
                )

                withdraw = self.post_step_strategy(
                    next_frame, distance, new_frame, transition_force
                )
                if withdraw:
                    next_frame = self.db_frame_trace[-1] + 1

                transition = self.db_frame_trace[-1] + 1 != next_frame
                first_search = False

                if end_of_file and (not transition):
                    print(
                        f"Err:: new_frame:{new_frame}/ new_db_frame:{new_db_frame} /{next_frame} / End of file, but best search result is the same frame"
                    )
                    frame_mask_idx.append(next_frame)

                    dist_z = np.linalg.norm(weighted_feature_diff[: self.model.z_dim])
                    dist_p = np.linalg.norm(
                        weighted_feature_diff[traj_start_idx:traj_dir_start_idx]
                    )
                    dist_d = np.linalg.norm(
                        weighted_feature_diff[traj_dir_start_idx:-2]
                    )
                    dist_fc = np.linalg.norm(weighted_feature_diff[-2:])
                    self.dbg_msg += f"distance: {distance:.3f}\ndist_z: {dist_z:.3f}\ndist_p: {dist_p:.3f}\ndist_d: {dist_d:.3f}\ndist_fc: {dist_fc:.3f}\n"

            dist_z = np.linalg.norm(weighted_feature_diff[: self.model.z_dim])
            dist_p = np.linalg.norm(
                weighted_feature_diff[traj_start_idx:traj_dir_start_idx]
            )
            dist_d = np.linalg.norm(weighted_feature_diff[traj_dir_start_idx:-2])
            dist_fc = np.linalg.norm(weighted_feature_diff[-2:])
            self.dbg_msg += f"distance: {distance:.3f}\ndist_z: {dist_z:.3f}\ndist_p: {dist_p:.3f}\ndist_d: {dist_d:.3f}\ndist_fc: {dist_fc:.3f}\n"

            cur_blend_margin = len(self.buffer_poses)
            next_zs = self.database.get_z_from_db_frame(
                next_frame, self.search_interval + self.blend_margin
            )
            out_motion, (feet_idx, out_contact) = decode_z_skelgraph(
                self.model,
                torch.tensor(next_zs, device=self.model.device),
                self.skel_graph,
                self.ms_dict,
            )
            out_contact = out_contact[:, self.feet_idx, 0]

            out_contact = np.minimum(np.maximum(out_contact, 0.0), 1.0)
            next_poses = out_motion.poses

            _, next_file_name, next_file_frame = self.database.get_file_frame(
                next_frame
            )
            self.dbg_msg += f"step @{new_frame}: {new_db_frame}(blend: {self.db_frame_trace_blend[-1]})->{next_frame}: ({next_file_name}/{next_file_frame}, len {len(next_poses)}), blend: {self.blend_type}\n"

            if self.blend_type == "none" or (not transition):
                next_poses = motion_utils.simple_stitch(
                    self.buffer_poses[0], next_poses
                )  # align by candidate next pose
                self.buffer_db_frame_traces = list(
                    range(next_frame, next_frame + len(next_poses))
                )
                self.buffer_db_frame_traces_blend = [None] * len(next_poses)
                self.buffer_blend_weight_trace = [0] * len(next_poses)
                self.buffer_zs_trace = next_zs
                self.buffer_fc = out_contact
                # print("transition / no blending")

            elif self.blend_type == "slerp":
                next_poses = motion_utils.simple_stitch(
                    self.buffer_poses[0], next_poses
                )  # align by candidate next pose
                next_poses[:cur_blend_margin], ratios = motion_utils.overlap_blend(
                    self.buffer_poses, next_poses[:cur_blend_margin], skel_preserve=True
                )
                self.buffer_db_frame_traces += list(
                    range(next_frame + cur_blend_margin, next_frame + len(next_poses))
                )
                self.buffer_db_frame_traces_blend = list(
                    range(next_frame, next_frame + cur_blend_margin)
                ) + [None] * (len(next_poses) - cur_blend_margin)
                self.buffer_blend_weight_trace = ratios + [0] * (
                    len(next_poses) - cur_blend_margin
                )
                # interpolate z as well

                ratio = np.array(self.buffer_blend_weight_trace)[
                    : len(self.buffer_fc), None
                ]
                # for i in range(len(ratios)):
                #     next_zs[i] = self.buffer_zs_trace[i] *(1-ratios[i]) + next_zs[i] * ratios[i]
                # self.buffer_zs_trace = next_zs
                self.buffer_zs_trace = (
                    self.buffer_zs_trace * (1 - ratio) + next_zs[: len(ratio)] * ratio
                )
                self.buffer_zs_trace = np.vstack(
                    (self.buffer_zs_trace, next_zs[len(ratio) :])
                )

                self.buffer_fc = (
                    self.buffer_fc * (1 - ratio) + out_contact[: len(ratio)] * ratio
                )
                self.buffer_fc = np.vstack(
                    (self.buffer_fc, out_contact[len(self.buffer_fc) :])
                )
                self.dbg_msg += f"cur_blend_margin:{cur_blend_margin}, buffer blend: {self.buffer_db_frame_traces_blend[0]}, ... {self.buffer_db_frame_traces_blend[-1]}\n"

            self.buffer_poses = next_poses

            if not len(self.buffer_db_frame_traces) == len(self.buffer_poses):
                print(f"{new_frame} buffer_db_frame_traces err")
                embed()  # assert
            if not len(self.buffer_db_frame_traces_blend) == len(self.buffer_poses):
                print(f"{new_frame} buffer_db_frame_traces_blend err")
                embed()  # assert
            if not len(self.buffer_blend_weight_trace) == len(self.buffer_poses):
                print(f"{new_frame} buffer_blend_weight_trace err")
                embed()  # assert
            if not len(self.buffer_zs_trace) == len(self.buffer_poses):
                print(f"{new_frame} buffer_blend_weight_trace err")
                embed()  # assert
            if not len(self.buffer_fc) == len(self.buffer_poses):
                print(f"{new_frame} buffer_fc err")
                embed()  # assert

        self.update_contact(new_frame)

        # self.record(new_frame)
        self.spring_adjust(new_frame)

        if self.extra_step is not None:
            self.extra_step(new_frame)

    def save_run(self):
        now = datetime.datetime.now()
        formattedDate = now.strftime("%Y%m%d_%H%M%S")
        run_save_dir = os.path.join(self.save_dir, formattedDate)
        print("SAVE AT :: ", run_save_dir)
        os.makedirs(run_save_dir)

        if not hasattr(self, "skel_start_frame"):
            # bvh.save(self.motion, os.path.join(run_save_dir, '0.bvh'))
            # from utils.motion_utils import denormalize
            # dn_motion = denormalize(self.motion, self.tgt_skel_motion, add_missing_joints=False)

            root_h = self.motion.skel.joints[0].xform_from_parent_joint[1, 3]
            dn_motion = motion_denormalize(
                self.motion, self.tgt_skel_motion, add_missing_joints=False
            )
            dn_motion = translate(dn_motion, [0, root_h, 0], 0, False)
            # orig_offset = np.stack([joint.xform_from_parent_joint[:3,3] for joint in tgt_skel_motion.skel.joints])
            # n_offset = np.stack([joint.xform_from_parent_joint[:3,3] for joint in n_motion.skel.joints])
            # dn_offset = np.stack([joint.xform_from_parent_joint[:3,3] for joint in dn_motion.skel.joints])
            # np.isclose(dn_offset, orig_offset, atol=1e-4).all()
            # np.isclose(dn_motion.rotations(), tgt_skel_motion.rotations()[1:-1], atol=1e-4).all()
            # np.isclose(dn_motion.positions(False), tgt_skel_motion.positions(False)[1:-1], atol=1e-4).all()

            bvh.save(dn_motion, os.path.join(run_save_dir, "0.bvh"))
        else:
            for i in range(len(self.skel_start_frame)):
                s = self.skel_start_frame[i]
                e = (
                    self.skel_start_frame[i + 1]
                    if (i + 1) < len(self.skel_start_frame)
                    else self.motion.num_frames()
                )
                partial_motion = Motion(skel=self.motion.poses[s].skel)
                partial_motion.poses = self.motion.poses[s:e]

                if s in self.start_character_dict:
                    orig_motion = self.start_character_dict[s]
                    root_h = partial_motion.skel.joints[0].xform_from_parent_joint[1, 3]
                    dn_motion = motion_denormalize(
                        partial_motion, orig_motion, add_missing_joints=False
                    )
                    dn_motion = translate(dn_motion, [0, root_h, 0], 0, False)
                    bvh.save(dn_motion, os.path.join(run_save_dir, f"{i}_{s}.bvh"))
                else:
                    bvh.save(partial_motion, os.path.join(run_save_dir, f"{i}_{s}.bvh"))

        traj_pos_stack = np.stack(self.log_traj_pos)
        traj_rot_stack = conversions.R2Q(
            np.stack(self.log_traj_rot) @ conversions.Ay2R(0.5 * np.pi)
        )  # Quaternion
        traj_stack = np.dstack((traj_pos_stack, traj_rot_stack))
        for i in range(self.traj_num):
            save_path = os.path.join(run_save_dir, f"traj_{i}.csv")
            np.savetxt(save_path, traj_stack[:, i], delimiter=",")

        control_pos = self.motion.positions(local=False)[:, 0]
        character_rot = conversions.R2Q(
            self.motion.rotations(local=False)[:, 0]
        )  # pivot rotation needed to draw control input
        control_input = np.hstack((control_pos, character_rot))
        save_path = os.path.join(run_save_dir, "control_input.csv")
        np.savetxt(save_path, control_input, delimiter=",")

        goal_vel_stack = np.stack(self.log_desired_vel)  # [T, 3]
        goal_rot_stack = np.stack(self.log_desired_rot).reshape(-1, 9)  # [T, 3, 3]
        goal_action_stack = np.stack(self.log_goal_action).reshape(-1, 1)  # [T, 1]
        goal_stack = np.hstack((goal_vel_stack, goal_rot_stack, goal_action_stack))
        save_path = os.path.join(run_save_dir, "goal.csv")
        np.savetxt(save_path, goal_stack, delimiter=",")

        save_path = os.path.join(run_save_dir, "z.npy")
        np.save(save_path, self.zs_trace)

        return run_save_dir

    def getRootTransform(self, frame=-1, use_height=False):
        return self.motion.poses[frame].get_root_facing_transform_byRoot(
            use_height=use_height
        )

    def getDBFrame(self, frame=-1):
        return self.db_frame_trace[frame]

    def change_skel(self, new_skel, new_frame):
        if not hasattr(self, "skel_start_frame"):
            self.skel_start_frame = [0]  # list of skel_start_frame
            # list of [skel_no, start_frame]
            # self.skel_motions = [self.motion]

        # skel_no = len(self.skel_start_frame)
        self.skel_start_frame.append(new_frame)
        # new_skel_motion = Motion(skel=new_skel)
        # for _ in range(new_frame+1):
        #     new_skel_motion.add_one_frame() # dummy (T-pose)
        # self.skel_motions.append(new_skel_motion)

        self.skel = new_skel
        # self.motion = new_skel_motion

        ## skel_graph, feet_idx
        self.skel_graph = skel_2_graph(self.skel)
        self.feet_idx = []
        links = motion_utils.get_links(self.skel)
        for link in links:
            if self.skel.get_joint(link[-1]).xform_global[1, 3] < 5:  # TODO FIX
                # self.feet_idx.extend(link[-2:])
                # print('leg links:', link)
                self.feet_idx.append(link[4])
        # print('feet idx: ',self.feet_idx)
        # self.feet_idx = np.array(self.feet_idx)

        ## buffer
        new_out_motion, (feet_idx, new_out_contact) = decode_z_skelgraph(
            self.model,
            torch.tensor(
                self.buffer_zs_trace, dtype=torch.float32, device=self.model.device
            ),
            self.skel_graph,
            self.ms_dict,
        )
        new_out_contact = new_out_contact[:, self.feet_idx, 0]

        new_out_contact = np.minimum(np.maximum(new_out_contact, 0.0), 1.0)
        self.buffer_fc = new_out_contact[-len(self.buffer_fc) :]
        new_poses = new_out_motion.poses[-len(self.buffer_poses) :]
        self.buffer_poses = motion_utils.simple_stitch(self.buffer_poses[0], new_poses)
        # self.buffer_poses = motion_utils.simple_stitch(self.motion.poses[-1], new_poses)
