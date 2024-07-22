import os, argparse
from mypath import *
from fairmotion.data import bvh
from conversions.motion_to_graph import motion_normalize_h2s

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="sample")
args = parser.parse_args()
in_char_path = os.path.join(DATA_DIR, args.data, "character", "bvh")
out_log_path = os.path.join(DATA_DIR, args.data, "character", "joint_pos")

if not os.path.exists(out_log_path):
    os.makedirs(out_log_path)

jointList = [
    "Reference",
    "Hips",
    "LowerBack",
    "LeftHipJoint",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeBase_End",
    "RightHipJoint",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "RightToeBase_End",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Spine4",
    "Spine5",
    "Spine6",
    "Spine7",
    "Spine8",
    "Spine9",
    "Neck",
    "Neck1",
    "Neck2",
    "Neck3",
    "Neck4",
    "Neck5",
    "Neck6",
    "Neck7",
    "Neck8",
    "Neck9",
    "Head",
    "Head_End",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHand_End",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHand_End",
]

"""
[TODO]: 
If bvh skeleton contains joints that are not in the MotionBuilder's biped character map,
register the joint name in `alternative_map`.
"""

alternative_map = {
    "ToSpine": "LowerBack",
    "LHipJoint": "LeftHipJoint",
    "RHipJoint": "RightHipJoint",
    "LeftToe": "LeftToeBase",
    "RightToe": "RightToeBase",
    "LeftToe_End": "LeftToeBase_End",
    "RightToe_End": "RightToeBase_End",
}

ee_joints = [
    "LeftToeBase_End",
    "RightToeBase_End",
    "Head_End",
    "LeftHand_End",
    "RightHand_End",
]


def write_skel_list(filename):
    in_filepath = os.path.join(in_char_path, filename)
    motion = bvh.load(in_filepath)

    joint_names = [joint.name for joint in motion.skel.joints]
    print("#", len(joint_names), "\t", filename)
    missing_ee = False
    for ee in ee_joints:
        if ee not in joint_names:
            print(">>> ", filename, "missing: ", ee)
            missing_ee = True
    if missing_ee:
        # MotionBuilder does not allow auto-characterizing without all end-effectors
        # Those files will be excluded from Skeleton Database to avoid error while automatic characterizing &=and creating paired motions
        return
        # ex)
        # >>> Prisoner B Styperek.bvh missing:  RightHand_End
        # >>> Mutant.bvh missing:  LeftHand_End

    n_motion, n_tpose = motion_normalize_h2s(motion)
    out_filepath = os.path.join(
        out_log_path, filename.replace(" ", "_").replace(".bvh", ".txt")
    )
    # print(out_filepath)
    with open(out_filepath, "w") as file:
        for ji, joint in enumerate(n_motion.skel.joints):

            if joint.name in alternative_map:
                joint_name = alternative_map[joint.name]
            else:
                joint_name = joint.name
            assert joint_name in jointList, f"{filename}, {joint.name}, {joint_name}"
            if ji == 0:
                file.write(f"{joint_name}, 0, 0, 0\n")
            else:
                offset = joint.xform_from_parent_joint[:3, 3]
                file.write(
                    f"{joint_name}, {offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}\n"
                )


for filename in os.listdir(in_char_path):
    write_skel_list(filename)
