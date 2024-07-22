import os, time, sys, math, gc, torch
import numpy as np
from fairmotion.core.motion import Pose, Motion
from torch_geometric.data import Batch

from mypath import *
from utils.tensor_utils import cdn, Tensor
from utils.data_utils import safe_normalize
from fairmotion.utils.contact_utils import get_foot_contact_ratio
from fairmotion.data import bvh
from conversions.motion_to_graph import motion_normalize_h2s, motion_2_graph
from task.motion_control.database import Database, parse_ann
from utils.motion_utils import get_links
from IPython import embed


def copmute_db_z(model, motion_list, mi_graph_frame_range_list):
    z_list = []
    for motion_i in motion_list:
        graph_batch = motion_2_graph(motion_i, True).to(device=model.device)
        z = model.encoder[0](graph_batch)
        z_list.append(cdn(z))
        gc.collect()
        torch.cuda.empty_cache()

    zs = []
    for mi, start, end in mi_graph_frame_range_list:
        zs.append(z_list[mi][start:end])
    zs = np.concatenate(zs)
    return zs


class Z_database(Database):
    def __init__(self) -> None:
        super().__init__()

    # @abstractmethod
    def construct_database(
        self,
        model,
        data_dir,
        traj_num,
        traj_interval,
        store_pdata=True,
    ):
        t0 = time.time()
        self.traj_num = traj_num
        self.traj_interval = traj_interval

        self.db_file_frames = []
        self.file_load_frames = []
        self.action_frame_list = []
        self.skel_list = []
        self.extra_features = []
        self.action_type_int = np.zeros((0, 1), dtype=int)
        self.interaction_time = np.zeros((0, 1))

        if store_pdata:
            self.p_datas = []

        ann_dir_path = os.path.join(data_dir, "annotation")
        ann, self.action2int, self.int2action = parse_ann(ann_dir_path)

        self.total_frames = 0
        self.mi_graph_frame_range_list = []

        motion_dir = os.path.join(data_dir, "motion")
        motion_list = []
        for mi, file_name in enumerate(os.listdir(motion_dir)):
            if (ann_dir_path is not None) and (file_name not in ann):
                continue

            file_path = os.path.abspath(os.path.join(motion_dir, file_name))
            motion = bvh.load(file_path)
            motion, tpose = motion_normalize_h2s(motion)
            motion_list.append(motion)
            data = motion.to_matrix(local=True)

            # frame_list: [start, end)
            # - CAUTION: does not include end : length = end-start
            if ann_dir_path is None:
                frame_list = [(1, motion.num_frames(), "base", None)]
            elif file_name in ann:
                frame_list = ann[file_name]
            else:
                assert False

            frame_list.sort(key=lambda x: x[0])

            F = motion.num_frames()
            frames = np.array(range(F))

            ff = []
            for ffi in range(self.traj_num):
                ff.append(
                    np.where(
                        frames + self.traj_interval * ffi >= F,
                        F - 1,
                        frames + self.traj_interval * ffi,
                    )
                )

            facingT_ground = np.zeros((F, 4, 4))
            for frame in range(F):
                facingT_ground[frame] = motion.poses[
                    frame
                ].get_root_facing_transform_byRoot(use_height=False)

            feet_idx = []
            links = get_links(motion.skel)
            for link in links:
                if motion.skel.get_joint(link[-1]).xform_global[1, 3] < 5:  # TODO FIX
                    feet_idx.extend(link[-2:])
            assert len(feet_idx) == 4, "WRONG FEET IDX"
            fc = get_foot_contact_ratio(motion.poses, feet_idx)
            # graph_poses, graph_skel = motion_2_graph(motion, normalized=True, tpose_at_first_frame=False, mean_std=ms_dict, set_max_map=True)

            for fi, (start, end, action_type, interaction_frame) in enumerate(
                frame_list
            ):
                if start == 0:
                    start = 1
                assert end <= motion.num_frames()
                segment_length = end - start

                self.mi_graph_frame_range_list.append(
                    [mi, start - 1, end - 1]
                )  # graph_frame: frame-1

                trajpos, trajdir = [], []
                for ffi in ff:
                    trajTi = (
                        np.linalg.inv(facingT_ground[start:end])
                        @ facingT_ground[ffi[start:end]]
                    )
                    trajpos.append(trajTi[:, [0, 2], [3, 3]])
                    trajdir.append(trajTi[:, [0, 2], [2, 2]])
                trajpos = np.hstack((trajpos))
                trajdir = np.hstack((trajdir))

                self.extra_features.append(
                    np.hstack((trajpos, trajdir, fc[start:end, ::2]))
                )

                ## action
                action_int = self.action2int[action_type] if ann_dir_path else 0
                action_repeat = np.repeat(
                    np.array([[action_int]]), segment_length, axis=0
                )
                self.action_type_int = np.vstack((self.action_type_int, action_repeat))

                if (interaction_frame is None) or (interaction_frame == -1):
                    interaction_time_i = np.array([0] * segment_length)
                else:
                    interaction_time_i = (
                        np.arange(segment_length) - (interaction_frame - start)
                    ).astype("float32")
                    interaction_time_i[: interaction_frame - start] /= (
                        interaction_frame - start
                    )
                    interaction_time_i[interaction_frame - start :] /= (
                        end - interaction_frame - 1
                    )

                self.interaction_time = np.vstack(
                    (self.interaction_time, interaction_time_i.reshape(-1, 1))
                )

                ## db, file frames
                # cont. segment (not new)
                if fi > 0 and (start == frame_list[fi - 1][1]):
                    self.db_file_frames[-1][-1] += segment_length
                    self.file_load_frames[-1][-1] = end
                    if store_pdata:
                        self.p_datas[-1] = np.vstack(
                            (self.p_datas[-1], data[start:end])
                        )

                # new segment
                else:
                    segment_id = len(self.db_file_frames)
                    self.db_file_frames.append(
                        [
                            segment_id,
                            f"{file_name}_{start}_{end}",
                            self.total_frames,
                            self.total_frames + segment_length,
                        ]
                    )  # does not include last frame
                    self.file_load_frames.append([file_path, start, end])
                    self.skel_list.append(motion.skel)
                    if store_pdata:
                        self.p_datas.append(data[start:end])

                if action_type != "base":
                    self.action_frame_list.append(
                        [
                            action_type,
                            self.total_frames,
                            self.total_frames + segment_length,
                        ]
                    )

                    print(
                        f"{action_type}({self.total_frames}, {self.total_frames+segment_length}): {interaction_frame-start}/{end-interaction_frame}"
                    )

                self.total_frames += segment_length

        self.extra_features = np.vstack((self.extra_features))
        self.extra_features, (self.ef_mean, self.ef_std) = safe_normalize(
            self.extra_features
        )

        """ DEBUGGING """
        # import pickle

        # with open(os.path.join(data_dir, "ckpt0.pkl"), "rb") as save_file:
        #     saved_var_dict = pickle.load(save_file)

        # for k in saved_var_dict.keys():
        #     try:

        #         if hasattr(self, k):
        #             print(k, end="\t\t")
        #             v = saved_var_dict[k]
        #             if type(v) == int:
        #                 print(saved_var_dict[k], getattr(self, k))
        #             elif torch.is_tensor(v):
        #                 print(
        #                     torch.isclose(
        #                         saved_var_dict[k], getattr(self, k), atol=1e-4
        #                     ).all()
        #                 )
        #             elif isinstance(v, np.ndarray):
        #                 print(
        #                     np.isclose(
        #                         saved_var_dict[k], getattr(self, k), atol=1e-4
        #                     ).all()
        #                 )
        #             elif isinstance(v, dict):
        #                 for kk, vv in v.items():
        #                     print(kk, vv, getattr(self, k)[kk])
        #             else:
        #                 print(saved_var_dict[k] == getattr(self, k))
        #     except:
        #         print(k, "FAILED")

        self.zs = copmute_db_z(model, motion_list, self.mi_graph_frame_range_list)
        self.features = np.hstack((self.zs, self.extra_features))

        self.features_size = self.features.shape[1]

        t1 = time.time()
        print(f"construct z_db :: {t1-t0:.4f}s")

    def get_pose(self, db_frame):
        file_i, file_name, file_frame_i = self.get_file_frame(db_frame)
        pose_i = Pose(self.skel_list[file_i], self.p_datas[file_i][file_frame_i])
        return pose_i

    def get_motion(self, db_frame_start, db_frame_end):
        file_i_s, file_name_s, file_frame_i_s = self.get_file_frame(db_frame_start)
        file_i_e, file_name_e, file_frame_i_e = self.get_file_frame(db_frame_end)
        assert file_i_s == file_i_e
        motion = Motion(skel=self.skel_list[file_i_s])
        p_data_i = self.p_datas[file_i_s]
        for i in range(file_frame_i_s, file_frame_i_e):
            motion.add_one_frame(p_data_i[i])

        return motion

    # def get_z_from_file_frame(self, file_id, file_frame, num_frames):
    #     print('get_z_from_file_frame')
    #     embed()
    #     db_frame = self.get_dbframe(file_id, file_frame)
    #     clamped = self.get_trajectory_index_clamp(db_frame, num_frames-1)
    #     num_frames = clamped - db_frame+1
    #     return self.zs[file_id][file_frame: file_frame+ num_frames]

    def get_z_from_db_frame(self, db_frame, num_frames):
        # file_id, file_frame = self.get_file_frame(db_frame)
        # return self.get_z_from_file_frame(file_id, file_frame, num_frames)
        clamped = self.get_trajectory_index_clamp(db_frame, num_frames - 1)
        num_frames = clamped - db_frame + 1
        return self.zs[db_frame : db_frame + num_frames]

    def search_closest(
        self,
        query,
        weights=None,
        frame_mask_idx=None,
        transition_z_limit=sys.float_info.max,
    ):
        # query: [z_dim + extra_feature_dim] : EXTRA_FEATURES are NOT "normalized"
        ef_size = len(self.ef_mean)
        query[-ef_size:] = (query[-ef_size:] - self.ef_mean) / self.ef_std
        # query = (query-self.mean)/self.std

        diff = self.features - query

        if weights is None:
            weights = np.ones(query.shape[0])
        w_diff = weights * diff

        if frame_mask_idx is not None:
            w_diff[frame_mask_idx] = sys.float_info.max

        z_exceed_ids = np.where(
            np.linalg.norm(diff[:, :-ef_size], axis=-1) > transition_z_limit
        )[0]
        w_diff[z_exceed_ids] = sys.float_info.max

        distance = np.linalg.norm(w_diff, axis=-1)
        # distance[distance > search_limit] = sys.float_info.max

        argmin_i = np.argmin(distance)

        return argmin_i, distance[argmin_i], w_diff[argmin_i]
