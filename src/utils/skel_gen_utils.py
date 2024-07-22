import os, random, itertools
import numpy as np


def parse_scale_txt(data_path):
    (
        offset_dir_mean,
        offset_dir_std,
        offset_dir_min,
        offset_dir_max,
        len_mean,
        len_std,
        len_min,
        len_max,
    ) = [dict() for _ in range(8)]

    with open(data_path, "r") as file:
        for line in file:
            tokens = [tok.strip() for tok in line.split(",")]
            jname = tokens[0]
            tokens = [float(tok) for ti, tok in enumerate(tokens) if ti != 0]
            i = 0
            offset_dir_mean[jname] = tokens[i : i + 3]
            i += 3
            offset_dir_std[jname] = tokens[i : i + 3]
            i += 3
            offset_dir_min[jname] = tokens[i : i + 3]
            i += 3
            offset_dir_max[jname] = tokens[i : i + 3]
            i += 3
            len_mean[jname] = tokens[i]
            i += 1
            len_std[jname] = tokens[i]
            i += 1
            len_min[jname] = tokens[i]
            i += 1
            len_max[jname] = tokens[i]
            i += 1
            assert len(tokens) == i
    # print(offset_dir_mean, offset_dir_std, offset_dir_min, offset_dir_max, len_mean, len_std, len_min, len_max)
    return (
        offset_dir_mean,
        offset_dir_std,
        offset_dir_min,
        offset_dir_max,
        len_mean,
        len_std,
        len_min,
        len_max,
    )


def parse_scale_txt(data_path):
    offset_mean = dict()
    with open(data_path, "r") as file:
        for line in file:
            if line.strip() == "":
                continue
            tokens = [tok.strip() for tok in line.split(",")]
            jname = tokens[0].replace("_END", "_End")
            offset_mean[jname] = np.array(
                [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            )
    return offset_mean


def rnd_offset_from_default(
    jointScaleStat,
    all_joint_list,
    head_joint_list,
    lleg_joint_list,
    root_joint,
    rand_range=0.4,
):
    spineCnt = len(
        [joint_name for joint_name in all_joint_list if "Spine" in joint_name]
    )
    spine_offset = jointScaleStat["Spine"]
    spine_offset = [
        spine_offset[0] / spineCnt,
        spine_offset[1] / spineCnt,
        spine_offset[2] / spineCnt,
    ]

    neckCnt = len([joint_name for joint_name in all_joint_list if "Neck" in joint_name])
    neck_offset = jointScaleStat["Neck"]
    neck_offset = [
        neck_offset[0] / neckCnt,
        neck_offset[1] / neckCnt,
        neck_offset[2] / neckCnt,
    ]

    rnd_offset = {}

    def rand_scale():
        return (1 - rand_range) + 2 * rand_range * random.random()

    gs = rand_scale()  # global_scale
    for jn in all_joint_list:
        if jn.startswith("Hip"):
            continue
        if jn.startswith("Right"):
            continue

        if "Spine" in jn:
            rnd_offset[jn] = spine_offset
        elif "Neck" in jn:
            rnd_offset[jn] = neck_offset
        elif ("HipJoint" in jn) or ("LowerBack" in jn):
            rnd_offset[jn] = [0, 0, 0]
        elif ("Shoulder" in jn) and (random.random() < 0.2):
            rnd_offset[jn] = [0, 0, 0]
        else:
            rnd_offset[jn] = jointScaleStat[jn]

        # ls = [0.6+ 0.8*random.random(), 0.6+ 0.8*random.random(), 0.6+ 0.8*random.random()] #local scale
        ls = [rand_scale(), rand_scale(), rand_scale()]  # local scale
        rnd_offset[jn] = [
            rnd_offset[jn][0] * ls[0] * gs,
            rnd_offset[jn][1] * ls[1] * gs,
            rnd_offset[jn][2] * ls[2] * gs,
        ]

        # Mirror
        if jn.startswith("Left"):
            # must replace like this because list is immutable (rnd_offset[jn.replace('Left', 'Right')][0]*=-1 doesn't work)
            rnd_offset[jn.replace("Left", "Right")] = [
                -rnd_offset[jn][0],
                rnd_offset[jn][1],
                rnd_offset[jn][2],
            ]

    height = -sum([rnd_offset[lljoint][1] for lljoint in lleg_joint_list])
    height += random.random() * 3  # [0, 3]cm random offset (+y)
    rnd_offset[root_joint] = [0, height, 0]
    return rnd_offset


def createSkelProperty_fromStat(jointScaleStat, rnd_hierarchy=True):
    """joint list, order"""
    head_joint_list = [
        ["Hips", "LowerBack"],
        ["Spine", "Spine1", "Spine2", "Spine3", "Spine4"],
        ["Neck", "Neck1"],
        ["Head", "Head_End"],
    ]  # 'Head_End']]
    leg_joint_list = [
        ["HipJoint"],
        ["UpLeg", "Leg", "Foot", "ToeBase", "ToeBase_End"],
    ]  #'ToeBase_End']]
    arm_joint_list = [
        ["Shoulder"],
        ["Arm", "ForeArm", "Hand", "Hand_End"],
    ]  #'Hand_End']]

    spineCnt, neckCnt = 3, 1
    useHipJoint = False
    useLowerBack = False
    if rnd_hierarchy:
        spineCntRange = range(1, 5)
        weights = [1.5 + 1 / x for x in spineCntRange]
        sum_weights = sum(weights)
        weights = [x / sum_weights for x in weights]
        spineRnd = random.random()
        sum_w = 0
        spineCnt = -1

        for i, w in enumerate(weights):
            sum_w += w
            if spineRnd < sum_w:
                spineCnt = i + 2
                break
        if spineCnt == -1:
            spineCnt = 5

        useHipJoint = random.random() < 0.2
        useLowerBack = random.random() < 0.2
        if random.random() < 0.2:
            neckCnt = 2

    max_spineCnt = len(head_joint_list[1])
    for i in range(max_spineCnt - spineCnt):
        head_joint_list[1].pop()

    max_neckCnt = len(head_joint_list[2])
    for i in range(max_neckCnt - neckCnt):
        head_joint_list[2].pop()

    if not useHipJoint:
        leg_joint_list[0].pop()
    if not useLowerBack:
        head_joint_list[0].pop()

    root_joint = head_joint_list[0][0]
    last_spine_joint = head_joint_list[1][-1]

    def flatten(folded_list):
        return list(itertools.chain(*folded_list))

    def add_prefix(namelist, prefix):
        return [prefix + v for v in namelist]

    head_joint_list = flatten(head_joint_list)
    leg_joint_list = flatten(leg_joint_list)
    arm_joint_list = flatten(arm_joint_list)
    lleg_joint_list = add_prefix(leg_joint_list, "Left")
    rleg_joint_list = add_prefix(leg_joint_list, "Right")
    larm_joint_list = add_prefix(arm_joint_list, "Left")
    rarm_joint_list = add_prefix(arm_joint_list, "Right")

    parent_map = {}
    for joint_list in [
        head_joint_list,
        lleg_joint_list,
        rleg_joint_list,
        larm_joint_list,
        rarm_joint_list,
    ]:
        for i in range(1, len(joint_list)):
            parent_map[joint_list[i]] = joint_list[i - 1]

    parent_map[head_joint_list[0]] = None
    parent_map[lleg_joint_list[0]] = root_joint
    parent_map[rleg_joint_list[0]] = root_joint
    parent_map[larm_joint_list[0]] = last_spine_joint
    parent_map[rarm_joint_list[0]] = last_spine_joint

    all_joint_list = flatten(
        [
            head_joint_list,
            lleg_joint_list,
            rleg_joint_list,
            larm_joint_list,
            rarm_joint_list,
        ]
    )
    # rnd_offset = random_offset_from_stat(jointScaleStat, all_joint_list, head_joint_list, lleg_joint_list, root_joint)
    rnd_offset = rnd_offset_from_default(
        jointScaleStat, all_joint_list, head_joint_list, lleg_joint_list, root_joint
    )

    """
    @return / rnd_offset: Dict[str, [x,y,z(floats)]]
    @return / parent_map: Dict[str, str]
    @return / all_joint_list: List[str] # order matters! parents should come first (for addJoint in MB)
    """
    ## must be "_End" instead of "_End" (but our jointScaleStat has "_End", so we should convert)
    rnd_offset = {
        key.replace("_End", "_End"): value for key, value in rnd_offset.items()
    }
    parent_map = {
        key.replace("_End", "_End"): (value.replace("_End", "_End") if value else None)
        for key, value in parent_map.items()
    }
    all_joint_list = [
        joint_name.replace("_End", "_End") for joint_name in all_joint_list
    ]

    return rnd_offset, parent_map, all_joint_list


def parse_all_char(data_path):
    filelist = list(filter(lambda x: x.endswith(".txt"), os.listdir(data_path)))
    char_list = []
    for filename in filelist:
        filepath = os.path.join(data_path, filename)
        character = parse_scale_txt(filepath)
        # with open(filepath, 'r') as file:
        #     for line in file:
        #         tokens = line.strip().split(",")
        #         joint_name = tokens[0].replace("_END", "_End")
        #         character[joint_name] = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
        #         # char.append([joint_name, float(tokens[1]), float(tokens[2]), float(tokens[3])])
        char_list.append([filename, character])
    # print("char_list: ", len(char_list))
    # for character in char_list:
    #     for joint, offset in character.items():
    #         print(joint, offset)
    #     print()
    return char_list


def rand_scale(rand_range):
    # random [1-rand_range, 1+rand_range]
    return (1 - rand_range) + 2 * rand_range * random.random()


def tweakSkelProperty(char_skel_list, rnd_hierarchy=True):
    char_i = random.randint(0, len(char_skel_list) - 1)
    char_file_i, char_stat_i = char_skel_list[char_i]
    # print(char_file_i)
    char_stat_i_names = list(
        char_stat_i.keys()
    )  # [char_stat_ij[0] for char_stat_ij in char_stat_i]

    head_joint_list = [
        ["Hips", "LowerBack"],
        ["Spine", "Spine1", "Spine2", "Spine3", "Spine4"],
        ["Neck", "Neck1"],
        ["Head", "Head_End"],
    ]  # 'Head_End']]
    leg_joint_list = [
        ["HipJoint"],
        ["UpLeg", "Leg", "Foot", "ToeBase", "ToeBase_End"],
    ]  #'ToeBase_End']]
    arm_joint_list = [
        ["Shoulder"],
        ["Arm", "ForeArm", "Hand", "Hand_End"],
    ]  #'Hand_End']]
    """
    @return / rnd_offset: Dict[str, [x,y,z(floats)]]
    @return / parent_map: Dict[str, str]
    @return / all_joint_list: List[str] # order matters! parents should come first (for addJoint in MB)
    """
    spineCnt = len(
        [spine for spine in head_joint_list[1] if spine in char_stat_i_names]
    )
    neckCnt = len([spine for spine in head_joint_list[2] if spine in char_stat_i_names])
    useHipJoint = "LeftHipJoint" in char_stat_i_names
    useLowerBack = "LowerBack" in char_stat_i_names
    if rnd_hierarchy:
        if random.random() < 0.2 and (not useHipJoint):
            useHipJoint = True
            char_stat_i["LeftHipJoint"] = np.array([0, 0, 0])
            char_stat_i["RightHipJoint"] = np.array([0, 0, 0])

        if random.random() < 0.2 and (not useLowerBack):
            useLowerBack = True
            char_stat_i["LowerBack"] = np.array([0, 0, 0])

        # randomize spine cnt
        default_spineCnt = spineCnt
        spineCnt += max(min(int(random.gauss(0, 2.0)), 2), -2)
        spineCnt = max(min(spineCnt, 5), 1)
        # modify default spine length
        if spineCnt != default_spineCnt:
            all_spine_list = head_joint_list[1]
            sum_spine = np.sum(
                [char_stat_i[all_spine_list[si]] for si in range(default_spineCnt)],
                axis=0,
            )

            if default_spineCnt < spineCnt:
                for si in range(default_spineCnt, spineCnt):
                    char_stat_i[all_spine_list[si]] = sum_spine * 1 / spineCnt
                for si in range(default_spineCnt):
                    char_stat_i[all_spine_list[si]] = (
                        char_stat_i[all_spine_list[si]] * default_spineCnt / spineCnt
                    )

            elif default_spineCnt > spineCnt:
                sum_spine_remain = np.sum(
                    [char_stat_i[all_spine_list[si]] for si in range(spineCnt)], axis=0
                )
                ratio = np.where(
                    np.abs(sum_spine_remain) < 1e-4,
                    np.ones_like(sum_spine),
                    sum_spine / sum_spine_remain,
                )
                for si in range(spineCnt):
                    char_stat_i[all_spine_list[si]] *= ratio
                for si in range(spineCnt, default_spineCnt):
                    del char_stat_i[all_spine_list[si]]

        # randomize neck cnt
        default_neckCnt = neckCnt
        neckCnt += max(min(int(random.gauss(0, 2.0)), 1), -1)
        neckCnt = max(min(neckCnt, 3), 1)
        if neckCnt != default_neckCnt:
            all_neck_list = head_joint_list[2]
            sum_neck = np.sum(
                [char_stat_i[all_neck_list[si]] for si in range(default_neckCnt)],
                axis=0,
            )

            if default_neckCnt < neckCnt:
                for si in range(default_neckCnt, neckCnt):
                    char_stat_i[all_neck_list[si]] = sum_neck * 1 / neckCnt
                for si in range(default_neckCnt):
                    char_stat_i[all_neck_list[si]] = (
                        char_stat_i[all_neck_list[si]] * default_neckCnt / neckCnt
                    )

            elif default_neckCnt > neckCnt:
                sum_neck_remain = np.sum(
                    [char_stat_i[all_neck_list[si]] for si in range(neckCnt)], axis=0
                )
                ratio = np.where(
                    np.abs(sum_neck_remain) < 1e-4,
                    np.ones_like(sum_neck),
                    sum_neck / sum_neck_remain,
                )
                for si in range(neckCnt):
                    char_stat_i[all_neck_list[si]] *= ratio
                for si in range(neckCnt, default_neckCnt):
                    del char_stat_i[all_neck_list[si]]

    max_spineCnt = len(head_joint_list[1])
    for i in range(max_spineCnt - spineCnt):
        head_joint_list[1].pop()

    max_neckCnt = len(head_joint_list[2])
    for i in range(max_neckCnt - neckCnt):
        head_joint_list[2].pop()

    if not useHipJoint:
        leg_joint_list[0].pop()
    if not useLowerBack:
        head_joint_list[0].pop()

    root_joint = head_joint_list[0][0]
    last_spine_joint = head_joint_list[1][-1]

    def flatten(folded_list):
        return list(itertools.chain(*folded_list))

    def add_prefix(namelist, prefix):
        return [prefix + v for v in namelist]

    head_joint_list = flatten(head_joint_list)
    leg_joint_list = flatten(leg_joint_list)
    arm_joint_list = flatten(arm_joint_list)
    lleg_joint_list = add_prefix(leg_joint_list, "Left")
    rleg_joint_list = add_prefix(leg_joint_list, "Right")
    larm_joint_list = add_prefix(arm_joint_list, "Left")
    rarm_joint_list = add_prefix(arm_joint_list, "Right")

    # parent_map
    parent_map = {}
    for joint_list in [
        head_joint_list,
        lleg_joint_list,
        rleg_joint_list,
        larm_joint_list,
        rarm_joint_list,
    ]:
        for i in range(1, len(joint_list)):
            parent_map[joint_list[i]] = joint_list[i - 1]
    parent_map[head_joint_list[0]] = None
    parent_map[lleg_joint_list[0]] = root_joint
    parent_map[rleg_joint_list[0]] = root_joint
    parent_map[larm_joint_list[0]] = last_spine_joint
    parent_map[rarm_joint_list[0]] = last_spine_joint

    # all_joint_list
    all_joint_list = flatten(
        [
            head_joint_list,
            lleg_joint_list,
            rleg_joint_list,
            larm_joint_list,
            rarm_joint_list,
        ]
    )

    # rnd_offset :: caution: Left / Right symmetry
    rnd_offset = {}
    global_scale = rand_scale(0.1)
    for jn, offset_j in char_stat_i.items():
        if jn.startswith("Hip"):
            continue
        if jn.startswith("Right"):
            continue

        local_scale = np.array([rand_scale(0.2), rand_scale(0.2), rand_scale(0.2)])
        rnd_offset[jn] = offset_j * local_scale * global_scale
        if jn.startswith("Left"):
            rnd_offset[jn.replace("Left", "Right")] = np.array(
                [-rnd_offset[jn][0], rnd_offset[jn][1], rnd_offset[jn][2]]
            )

    height = -sum([rnd_offset[lljoint][1] for lljoint in lleg_joint_list])
    height += random.random() * 3  # [0, 3]cm random offset (+y)
    rnd_offset[root_joint] = np.array([0, height, 0])

    return rnd_offset, parent_map, all_joint_list


from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions


def generateSkel(rnd_offset, parent_map, all_joint_list):
    skel = motion_class.Skeleton()
    for joint_name in all_joint_list:
        offset_T = conversions.p2T(rnd_offset[joint_name])

        parent_name = parent_map[joint_name]
        if parent_name is None:
            dof = 6
            parent_joint = None
        else:
            if joint_name in parent_map:
                dof = 3
            else:
                dof = 0
            parent_joint = skel.get_joint(parent_name)

        new_joint = motion_class.Joint(joint_name, dof, offset_T, parent_joint)
        new_joint.set_parent_joint(parent_joint)
        skel.add_joint(new_joint, parent_joint)

    # shift root up
    joint = skel.get_joint("LeftToeBase_End")
    root_height = 0
    while joint.name != "Hips":
        root_height -= joint.xform_from_parent_joint[1, 3]
        joint = joint.parent_joint

    root = skel.get_joint("Hips")
    root.xform_from_parent_joint[1, 3] = root_height + random.random() * 3
    root.set_xform_global_recursive(root.xform_from_parent_joint)

    return skel


from mypath import *

char_skel_list = parse_all_char(
    os.path.join(DATA_DIR, "test", "character", "joint_pos")
)
jointScaleStat = parse_scale_txt(
    os.path.join(DATA_DIR, "test", "character", "default.txt")
)


def create_random_skel(mode="data", rnd_hierarchy=True):
    if mode == "data":
        rnd_offset, parent_map, all_joint_list = tweakSkelProperty(
            char_skel_list, rnd_hierarchy
        )
    elif mode == "single":
        rnd_offset, parent_map, all_joint_list = createSkelProperty_fromStat(
            jointScaleStat, rnd_hierarchy
        )
    return generateSkel(rnd_offset, parent_map, all_joint_list)


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    from fairmotion.core import motion as motion_class
    from default_veiwer import get_default_viewer

    viewer = get_default_viewer()

    def extra_key_callback(key):
        if key == b"m":
            skeleton = create_random_skel("data", rnd_hierarchy=True)
            motion = motion_class.Motion(skel=skeleton)
            motion.add_one_frame(None)
            motion.add_one_frame(None)
            viewer.update_motions([motion])

    def extra_render_callback():
        pass

    viewer.extra_key_callback = extra_key_callback
    viewer.extra_render_callback = extra_render_callback

    viewer.run()
