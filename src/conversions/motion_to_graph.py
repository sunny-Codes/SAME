import torch
import numpy as np
from fairmotion.ops import conversions
from fairmotion.utils import constants
from fairmotion.data import bvh
from fairmotion.utils.contact_utils import get_all_joint_ground_contact

from utils.motion_utils import motion_normalize_h2s
from same.skel_pose_graph import SkelPoseGraph
from same.mydataset import npz_2_data, SkelData
from torch_geometric.data import Batch


def skel_2_state(skel):
    """
    input: skel
        - must be normalized
        - tpose height should be applied to skel
    output: skel_state
        - offset(local, global)
        - feet-idx
        - cid, qid
    """
    offsets = np.array([joint.xform_from_parent_joint[:3, 3] for joint in skel.joints])
    global_offsets = np.array([joint.xform_global[:3, 3] for joint in skel.joints])

    q_bool = np.array(
        [1 if len(j.child_joints) != 0 else 0 for j in skel.joints], "int32"
    )

    root_idx = 0
    depth = 0
    stack = [root_idx]
    edges = {root_idx: [root_idx, root_idx, -1]}  # dummy edge
    ascendant = {root_idx: [root_idx]}

    while len(stack) > 0:
        next_level_stack = []
        for parent_idx in stack:
            for child in skel.joints[parent_idx].child_joints:
                child_idx = skel.get_index_joint(child.name)
                edges[child_idx] = [parent_idx, child_idx, depth]
                next_level_stack.append(child_idx)

                ascendant[child_idx] = ascendant[parent_idx].copy()
                ascendant[child_idx].append(child_idx)

            if len(skel.joints[parent_idx].child_joints) == 0:
                for reverse_depth, joint_idx in enumerate(
                    reversed(ascendant[parent_idx])
                ):
                    if len(edges[joint_idx]) > 3:
                        edges[joint_idx][3] = max(reverse_depth, edges[joint_idx][3])
                    else:
                        edges[joint_idx].append(reverse_depth)

        stack = next_level_stack
        depth += 1

    edges_np = np.array([edges[j_idx] for j_idx in edges.keys()], "int32")
    edges_np = edges_np[np.argsort(edges_np[:, 1])]  # sort by child idx

    # edges_np: [J, 4] : [parent, child, depth, reverse_depth]
    # extra features:
    #     depth: for faster fk compute
    #     reverse_depth: for masking

    return offsets.astype("float32"), global_offsets.astype("float32"), q_bool, edges_np


def motion_2_states(motion, exclude_wov=True):
    """
    input: motion
            - must be normalized
            - tpose height should be applied to skel
            - no tpose on the first frame (can use all frames)
    output: skel_state, poses_state
        (pose's' state: merged state, not a list of [pose_state])
        ** return features joints are not re-ordered. (follows BVH hierarchy order)
    """
    rotations = motion.rotations()
    global_positions = motion.positions(local=False)

    # skel (static)
    skel_state = skel_2_state(motion.skel)
    # skel_state : offsets, global_offsets, q_bool, edges

    # poses (dynamic)
    F = motion.num_frames()
    J = global_positions.shape[1]

    facing_transforms = np.zeros((F, 4, 4))
    for frame in range(F):
        facing_transforms[frame] = motion.poses[frame].get_root_facing_transform_byRoot(
            use_height=False
        )

    # smooth filter?
    # import scipy.signal as signal
    # sim_position = signal.savgol_filter(sim_position, 31, 3, axis=0, mode='interp')

    facing_diff = constants.eye_T()[None, ...].repeat(facing_transforms.shape[0], 0)
    facing_diff[1:] = np.linalg.inv(facing_transforms[:-1]) @ facing_transforms[1:]
    dcos, dsin = facing_diff[..., 0, 0], facing_diff[..., 0, 2]
    dx, dz = facing_diff[..., 0, 3], facing_diff[..., 2, 3]
    dtheta = np.arctan2(dsin, dcos)

    r_data = np.vstack((dtheta, dx, dz, global_positions[:, 0, 1])).transpose(1, 0)

    rotations[:, 0] = np.linalg.inv(facing_transforms)[..., :3, :3] @ rotations[:, 0]
    rotations_Rsix = conversions.R2Rsix(rotations)

    curT_rel_positions = (
        np.linalg.inv(facing_transforms)[:, np.newaxis, ...].repeat(J, 1)
        @ conversions.p2T(global_positions)
    )[..., :3, 3]

    p_vel = np.zeros((F, J, 3))
    p_vel[1:] = global_positions[1:] - global_positions[:-1]

    curT_rel_p_vel = (
        np.linalg.inv(facing_transforms)[:, np.newaxis, ...].repeat(J, 1)
    )[..., :3, :3] @ p_vel[..., np.newaxis]
    curT_rel_p_vel = curT_rel_p_vel[..., 0] * 30  # fps

    q_vel = constants.eye_R()[np.newaxis, np.newaxis, ...].repeat(F, 0).repeat(J, 1)
    q_vel[1:] = rotations[:-1].swapaxes(-2, -1) @ rotations[1:]
    q_vel_Rsix = conversions.R2Rsix(q_vel)

    p_prev = np.copy(global_positions)
    p_prev[1:] = global_positions[:-1]
    curT_rel_p_prev = (
        np.linalg.inv(facing_transforms)[:, np.newaxis, ...].repeat(J, 1)
        @ conversions.p2T(p_prev)
    )[..., :3, 3]

    fc_data = get_all_joint_ground_contact(motion)

    # q, p, r, pv, qv, pprev, c
    poses_state = (
        rotations_Rsix.astype("float32"),
        curT_rel_positions.astype("float32"),
        r_data.astype("float32"),
        curT_rel_p_vel.astype("float32"),
        q_vel_Rsix.astype("float32"),
        curT_rel_p_prev.astype("float32"),
        fc_data.astype("float32"),
    )

    if exclude_wov:
        poses_state = tuple([state[1:] for state in poses_state])

    return skel_state, poses_state


def skel_2_graph(skel):
    (lo, go, qb, edges) = skel_2_state(skel)
    sd = SkelData(
        torch.Tensor(lo),
        torch.Tensor(go),
        torch.BoolTensor(qb),
        torch.LongTensor(edges[:, :2]).transpose(1, 0),
        torch.LongTensor(edges[:, 2:]),
    )
    return SkelPoseGraph(sd, None)


def motion_2_graph(motion, normalized=False):
    if not normalized:
        motion, tpose = motion_normalize_h2s(motion, False)  # normalize motion
    # skel_state, poses_state = motion_2_states
    (lo, go, qb, edges), (q, p, r, pv, qv, pprev, c) = motion_2_states(motion)
    skel_data, pose_list = npz_2_data(lo, go, qb, edges, q, p, qv, pv, pprev, c, r)
    graph_batch = Batch.from_data_list(
        [SkelPoseGraph(skel_data, pose_i) for pose_i in pose_list]
    )
    return graph_batch


def bvh_2_graph(bvh_filepath):
    motion = bvh.load(bvh_filepath, ignore_root_skel=True, ee_as_joint=True)
    return motion_2_graph(motion)
