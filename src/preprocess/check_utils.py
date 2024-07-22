from scipy.spatial.transform.rotation import Rotation
import numpy as np


# Check motion
def is_root_flip_motion(motion, filter_thres=0.45, start_frame=0):
    max_diff, max_diff_frame = 0, -1
    for frame, pose in enumerate(motion.poses[start_frame:]):
        root_proj = pose.get_root_facing_transform_byRoot(use_height=True)[:3, :3]
        root_orig = pose.get_transform(0, local=True)[:3, :3]
        root_diff = Rotation.from_matrix(
            np.linalg.inv(root_proj) @ root_orig
        ).as_rotvec()
        root_diff_norm = np.linalg.norm(root_diff) / np.pi
        if root_diff_norm > max_diff:
            max_diff = root_diff_norm
            max_diff_frame = frame
    return max_diff > filter_thres


def is_root_low_or_high(motion, lb, ub, consq_n_thres, start_frame=0):
    # assert first frame tpose
    root_T_height = motion.poses[0].get_transform(0, False)[1, 3]
    root_h = motion.positions()[start_frame:, 0, 1]
    too_low = root_h < root_T_height * lb
    too_high = root_h > root_T_height * ub
    # for pose in motion.poses[start_frame:]:
    #     r_h = pose.get_root_transform()[1,3]
    #     if (r_h < root_T_height * lb) or (r_h > root_T_height * ub):
    #         return True
    # return False
    for frame in range(start_frame + consq_n_thres, motion.num_frames()):
        if (
            too_low[frame - consq_n_thres : frame].all()
            or too_high[frame - consq_n_thres : frame].all()
        ):
            return True
    return False


def is_discontinuous_motion(motion, diff_bound=50, start_frame=0):
    for frame, pose in enumerate(motion.poses):
        if frame - 1 >= start_frame:
            r_prev = motion.poses[frame - 1].get_root_transform()[:3, 3]
            r_frame = pose.get_root_transform()[:3, 3]
            r_diff = r_prev - r_frame
            r_diff[1] = 0
            if np.linalg.norm(r_diff) > diff_bound:
                print(frame, ":", np.linalg.norm(r_diff), ">", diff_bound)
                return True
    return False


def is_penetrated_motion(motion, penetrate_thres=0.1, method="any", start_frame=0):
    y = np.min(motion.positions(local=False)[start_frame:, :, 1], axis=1)  # [F, J]
    if method == "any":
        return (y < penetrate_thres).any()
    elif method == "mean":
        return np.mean(y < 0) > penetrate_thres  # percentage of penetration
    else:
        assert False, "method not defined: " + method


def is_still_pose(motion, start_frame=0):
    ma_positions = motion.positions()[start_frame:]
    da = np.linalg.norm(ma_positions[1:] - ma_positions[:-1], axis=(1, 2))

    if np.isclose(da, 0.0, atol=1e-6).any():
        close_frames = np.where(np.isclose(da, 0.0, atol=1e-5))[0]
        # if len(close_frames) > 5:
        # print('close_frames:', close_frames)
        # print(fpath_a)
        # embed()
        consq_close_frames_max = 1
        consq_close_frames_start = 0
        for cfi in range(1, len(close_frames)):
            if close_frames[cfi] - close_frames[cfi - 1] == 1:
                cur_consq_len = cfi - consq_close_frames_start + 1
                if cur_consq_len > consq_close_frames_max:
                    consq_close_frames_max = cur_consq_len
            else:
                consq_close_frames_start = cfi

        return consq_close_frames_max > 5
    return False


def is_root_upsidedown_motion(motion, filter_thres=-0.7, start_frame=1):
    # assert first frame tpose
    first_frame_Yaxis = motion.poses[0].get_transform(0, local=False)[1, :3]
    for frame, pose in enumerate(motion.poses):
        if frame < start_frame:
            continue
        Ti = pose.get_transform(0, local=False)[:3, :3]
        yi = Ti @ first_frame_Yaxis
        if yi[1] < filter_thres:
            # print(frame, yi)
            return True
    return False
