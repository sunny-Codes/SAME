from fairmotion.ops import conversions
import numpy as np
from IPython import embed

from fairmotion.core import motion as motion_class
from fairmotion.ops import motion as fm_motion_util
from fairmotion.utils import constants
import copy

from fairmotion.ops.motion import transform_poses

""" ====================== STITCHING METHODS ======================= """


def simple_stitch(cur_pose, next_poses, align_id=0):
    next_pose = next_poses[align_id]
    nextT = next_pose.get_root_facing_transform_byRoot(use_height=False)
    curT = cur_pose.get_root_facing_transform_byRoot(use_height=False)
    alignT = curT @ np.linalg.inv(nextT)
    fm_motion_util.transform_poses(next_poses, alignT, local=False)
    return next_poses


def overlap_blend(from_poses, to_poses, skel_preserve=False):
    # CAUTION: assert already aligned
    assert len(from_poses) == len(to_poses)

    F = len(from_poses)
    ratios = [0.5 - 0.5 * np.cos(i / (F - 1) * np.pi) for i in range(F)]
    for i in range(F):
        to_poses[i] = fm_motion_util.blend(
            from_poses[i], to_poses[i], ratios[i], skel_preserve
        )

    return to_poses, ratios


""" ====================== NORMALIZE/DENORMALIZE MOTION ======================="""


## HERE WE MEAN "NORMALIZE" == make every joint xform from parent joint to be identity matrix
## adjust height according to tpose contacts
def motion_normalize(motion, tpose=None):
    if tpose is None:
        tpose = motion.poses[0]
    recon_motion = motion_class.Motion(skel=copy.deepcopy(motion.skel))
    recon_motion.set_fps(motion.fps)

    for i, joint in enumerate(recon_motion.skel.joints):
        if joint.parent_joint is not None:
            joint.xform_from_parent_joint = constants.eye_T()
        joint.xform_global = constants.eye_T()

    for i, joint in enumerate(recon_motion.skel.joints):
        jidx = recon_motion.skel.get_index_joint(joint)
        jpos = tpose.get_transform(jidx, local=False)[0:3, 3]
        if joint.parent_joint is not None:
            pidx = recon_motion.skel.get_index_joint(joint.parent_joint)
            ppos = tpose.get_transform(pidx, local=False)[0:3, 3]
        else:
            ppos = jpos

        T_new = constants.eye_T()
        T_new[0:3, 3] = jpos - ppos

        joint.xform_from_parent_joint = T_new
        if joint.parent_joint is not None:
            joint.xform_global = np.dot(
                joint.parent_joint.xform_global,
                joint.xform_from_parent_joint,
            )
        else:
            joint.xform_global = joint.xform_from_parent_joint

        for el in joint.extra_links:
            elpos = np.dot(
                tpose.get_transform(jidx, local=False), joint.extra_links[el]
            )[0:3, 3]
            joint.extra_links[el] = constants.eye_T()
            joint.extra_links[el][0:3, 3] = elpos - jpos

    num_frames = len(motion.poses)
    T_relative = np.zeros((num_frames, motion.skel.num_joints(), 4, 4))
    T_relative[...] = constants.eye_T()
    for i in range(num_frames):
        recon_motion.add_one_frame(T_relative[i])

    motion_local = motion.to_matrix(local=True)  # [F, J, 4, 4]
    motion_global = motion.to_matrix(local=False)

    for j, (joint, recon_joint) in enumerate(
        zip(motion.skel.joints, recon_motion.skel.joints)
    ):
        if joint.parent_joint is None:
            try:
                # broadcast ok : [F, 3, 3] @ [3, 3] = [F, 3, 3]
                T_relative[:, j, :3, :3] = motion_local[:, j, :3, :3] @ np.linalg.inv(
                    tpose.data[j][:3, :3]
                )
                T_relative[:, j, :3, 3] = motion_local[:, j, :3, 3]
            except:
                embed()
        else:
            pidx = motion.skel.get_index_joint(joint.parent_joint)
            jidx = motion.skel.get_index_joint(joint)
            assert j == jidx

            gp = motion_global[:, pidx, :3, :3]
            gj = motion_global[:, jidx, :3, :3]
            p2j = joint.xform_from_parent_joint[:3, :3]

            delta_global = constants.eye_R()
            delta_global = gj @ np.linalg.inv(gp @ p2j @ tpose.data[j][:3, :3])

            new_gp = recon_motion.get_joint_transforms(joint, local=False)[:, :3, :3]
            new_p2j = recon_joint.xform_from_parent_joint[:3, :3]
            new_pg_p2j = new_gp @ new_p2j

            delta_local = constants.eye_R()
            delta_local = np.linalg.inv(new_pg_p2j) @ delta_global @ new_pg_p2j

            T_relative[:, j, :3, :3] = delta_local

    return recon_motion


# TH)
def copy_normalized_joint(norm_motion, out_motion, joint_name):
    # assume that parent joint's name is identical
    # and skeleton hierarchy is the same other than missing joints
    norm_joint = norm_motion.skel.get_joint(joint_name)
    norm_parent_joint = norm_joint.parent_joint
    out_parent_joint = out_motion.skel.get_joint(norm_parent_joint.name)

    # multiply output parent joint's xform inverse
    # and multiply input parent joint's xform inverse
    I = constants.eye_T()
    xform_bw_skel = (
        np.linalg.inv(out_parent_joint.xform_global) @ norm_parent_joint.xform_global
    )
    orig_xform = norm_joint.xform_from_parent_joint
    out_xform = xform_bw_skel @ orig_xform

    out_new_joint = motion_class.Joint(
        joint_name, xform_from_parent_joint=out_xform, parent_joint=out_parent_joint
    )
    out_motion.skel.add_joint(out_new_joint, out_parent_joint)
    for pose in out_motion.poses:
        assert pose.skel is out_motion.skel
        # add tpose xform for fingers which should be zero in input motion
        data = pose.data
        pose.data = np.concatenate(
            (pose.data, np.expand_dims(out_xform, axis=0)), axis=0
        )


# TH)
def motion_denormalize(norm_motion, tpose_motion, add_missing_joints=False):
    # Resize tpose_motion's transform
    norm_bvh_joints = list(map(lambda x: x.name, norm_motion.skel.joints))
    tpose_bvh_joints = list(map(lambda x: x.name, tpose_motion.skel.joints))

    parent_names = set(
        map(
            lambda x: x.parent_joint.name if x.parent_joint is not None else ":None:",
            norm_motion.skel.joints,
        )
    )

    error = False

    for joint_name in norm_bvh_joints:
        in_joint = norm_motion.skel.get_joint(joint_name)
        in_parent_name = (
            in_joint.parent_joint.name
            if in_joint.parent_joint is not None
            else ":None:"
        )

        if joint_name not in tpose_bvh_joints:
            if in_parent_name in tpose_bvh_joints and add_missing_joints:
                copy_normalized_joint(norm_motion, tpose_motion, joint_name)
                tpose_bvh_joints.append(joint_name)
                continue
            else:
                print("Required joint {} is missing in tpose".format(joint_name))
                error = True

        t_joint = tpose_motion.skel.get_joint(joint_name)
        t_parent_name = (
            t_joint.parent_joint.name if t_joint.parent_joint is not None else ":None:"
        )

        if in_parent_name != t_parent_name:
            print(
                "{}'s parent joint name is inconsistent: {} and {}".format(
                    joint_name, in_parent_name, t_parent_name
                )
            )
            error = True

    if error:
        return "Failed to denormalize motion {}".format(norm_motion.name)

    # Normalize zero-pose of denormalized skeleton
    zero_pose = motion_class.Pose(tpose_motion.skel)
    tpose_motion.add_one_frame(zero_pose.data)

    norm_tpose_motion = motion_normalize(tpose_motion)
    norm_zero_pose = norm_tpose_motion.poses[-1]

    # Denormalize motion with normalized zero position skel pose
    in_zero_pose = motion_class.Pose(norm_motion.skel)
    for i, joint in enumerate(norm_motion.skel.joints):
        if joint.name in tpose_bvh_joints:
            data_idx = tpose_motion.skel.get_index_joint(joint.name)
            in_zero_pose.data[i] = norm_zero_pose.data[data_idx]

    norm_motion_denorm = motion_normalize(norm_motion, tpose=in_zero_pose)

    return norm_motion_denorm


def tpose_height_to_skel(motion, tpose, apply_=False):
    if apply_:
        res_motion = motion
    else:
        res_motion = copy.deepcopy(motion)

    tpose_root = tpose.data[0][:3, 3]
    tpose_root[0] = 0
    tpose_root[2] = 0

    for n_pose in res_motion.poses:
        n_pose.data[0, :3, 3] -= tpose_root

    root_new_xform_global = res_motion.skel.joints[0].xform_global
    root_new_xform_global[:3, 3] += tpose_root
    res_motion.skel.joints[0].set_xform_global_recursive(root_new_xform_global)

    return res_motion


def motion_normalize_h2s(motion, handle_penetration=True):

    tpose = motion.poses[0]
    if handle_penetration:
        skel = motion.skel
        joint_names = [j.name for j in skel.joints]
        lt = "LeftToeBase_End" if "LeftToeBase_End" in joint_names else "LeftToe_End"
        rt = "RightToeBase_End" if "RightToeBase_End" in joint_names else "RightToe_End"
        if not (lt in joint_names) or not (rt in joint_names):
            print("handle penetration Err")
            embed()
            exit()
        lty = tpose.get_transform(lt, local=False)[1, 3]
        rty = tpose.get_transform(rt, local=False)[1, 3]
        if (lty + rty) / 2 < 0:
            print("handle penetrating t-pose")
            print(motion.name)
            # embed()
            transform_poses([tpose], conversions.p2T([0, -(lty + rty) / 2.0, 0]))

    n_motion = motion_normalize(motion, tpose)
    n_tpose = n_motion.poses[0]
    n_motion.poses = n_motion.poses[1:]
    tpose_height_to_skel(n_motion, n_tpose, apply_=True)

    # f_ids = get_foot_indices(n_motion.skel)
    # n_motion.contact = get_foot_contact_ratio(n_motion.poses, f_ids)

    return n_motion, n_tpose


""" ====================== other ... ======================="""


def get_links(skel):
    """
    ex)
        [[0, 1, 2, 3, 4, 5],
        [0, 6, 7, 8, 9, 10],
        [0, 11, 12, 13],
        [13, 14, 15, 16],
        [13, 17, 18, 19, 20],
        [13, 21, 22, 23, 24]]
    """

    tree = []
    mask = np.array([0] * len(skel.joints))

    def dfs_tweak(joint):
        ji = skel.get_index_joint(joint)
        mask[ji] = 1
        if ji != 0 and len(joint.child_joints) > 1:
            tree[-1].append(ji)
        for child in joint.child_joints:
            ci = skel.get_index_joint(child)
            if len(joint.child_joints) > 1:
                tree.append([])
            tree[-1].append(ji)
            if not mask[ci]:
                dfs_tweak(child)
        if len(joint.child_joints) == 0:
            tree[-1].append(ji)

    dfs_tweak(skel.joints[0])
    return tree


def get_edge(skel, bidirection=True):
    """
    input: skeletona
        - bidiretion
            - TRUE: (parent, child) and (child, parent)
            - FALSE: (parent, child) only
        - include_ee: treat end-effector as joint or not
    output: edge [2, E]
    """
    edge = []
    for joint in skel.joints:
        ji = skel.get_index_joint(joint)
        for child in joint.child_joints:
            ci = skel.get_index_joint(child)
            edge.append([ji, ci])
            if bidirection:
                edge.append([ci, ji])

    return np.array(edge).transpose(1, 0)


def skel_interpolate(skel_a, skel_b, ratio):
    # assert joint names are the same
    joint_names_a = [joint.name for joint in skel_a.joints]
    joint_names_b = [joint.name for joint in skel_b.joints]
    assert set(joint_names_a) == set(joint_names_b)

    skel_new = copy.deepcopy(skel_a)
    for joint_name in joint_names_a:
        joint_a = skel_a.get_joint(joint_name)
        joint_b = skel_b.get_joint(joint_name)
        joint_new = skel_new.get_joint(joint_name)
        joint_new.xform_from_parent_joint[:3, 3] = (
            1 - ratio
        ) * joint_a.xform_from_parent_joint[
            :3, 3
        ] + ratio * joint_b.xform_from_parent_joint[
            :3, 3
        ]

    for joint_new in skel_new.joints:
        if joint_new.parent_joint is None:
            continue
        joint_new.xform_global = np.dot(
            joint_new.parent_joint.xform_global,
            joint_new.xform_from_parent_joint,
        )
    return skel_new


from fairmotion.utils import contact_utils


def make_motion(
    skel,
    qR,
    ra_T,
    c,
    first_frame_zero=False,
    contact_cleanup=False,
    cid=None,
    motion=None,
):
    """
    skel    motion_class.Skel
    qR      arr [T, J, 3, 3]
    ra_T    arr [T, 4, 4] # root height applied here
    c       arr [T, J, 1]
    cid     list [int]
    """
    frame_num, joint_num = qR.shape[0], qR.shape[1]

    poses_T = np.zeros((frame_num, joint_num, 4, 4))
    poses_T[...] = constants.eye_T()
    poses_T[..., :3, :3] = qR

    if first_frame_zero:
        ra_T[0] = constants.eye_T()

    poses_T[:, 0] = ra_T @ poses_T[:, 0]
    poses_T[:, 0, 1, 3] = ra_T[:, 1, 3] - skel.joints[0].xform_from_parent_joint[1, 3]

    if not motion:
        motion = motion_class.Motion(skel=skel)
    for f in range(frame_num):
        motion.add_one_frame(poses_T[f])

    if contact_cleanup:
        assert cid is not None
        toe_idx = [int(cid[1]), int(cid[3])]
        if contact_cleanup:
            # print("toe_idx: ", toe_idx)
            contact_utils.init_contact(motion, 0, toe_idx)
            contact_utils.motion_foot_cleanup(motion, c[:, toe_idx, 0])
            np.set_printoptions(precision=5, suppress=True)
        return motion, (toe_idx, c)  # c[:, toe_idx, 0])
    else:
        return motion, (list(range(joint_num)), c)


if __name__ == "__main__":

    import os
    from mypath import *
    from fairmotion.data import bvh

    motion_megan = bvh.load(
        os.path.join(DATA_DIR, "characters", "polished_bvh", "megan.bvh")
    )
    n_motion_megan, n_tpose_megan = motion_normalize_h2s(motion_megan)
    skel_megan = n_motion_megan.skel

    motion_mousey = bvh.load(
        os.path.join(DATA_DIR, "characters", "polished_bvh", "mousey.bvh")
    )
    n_motion_mousey, n_tpose_mousey = motion_normalize_h2s(motion_mousey)
    skel_mousey = n_motion_mousey.skel

    new_skel = skel_interpolate(skel_megan, skel_mousey, 0.5)
