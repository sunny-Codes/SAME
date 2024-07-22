import os

from pyfbsdk import *
import random
import math
import itertools
import numpy as np

# to install numpy for motionbuilder, please refer to: https://help.autodesk.com/view/MOBPRO/2022/ENU/?guid=GUID-46E090C5-34AD-4E26-872F-F7D21DC57C74

# Tutorials/code I referred to:
# https://help.autodesk.com/view/MOBPRO/2019/ENU/?guid=__files_GUID_A1189AA0_3816_4350_B8F3_5383DEC25A33_htm
# https://github.com/eksod/Retargeter


""" ======================================================  Native Python Functions  ======================================================  """
""" parsing character pool, tweaking skeleton properties, converting bvh format, etc   """


def parse_all_char(data_path):
    filelist = list(filter(lambda x: x.endswith(".txt"), os.listdir(data_path)))
    char_list = []
    for filename in filelist:
        filepath = os.path.join(data_path, filename)
        character = {}
        with open(filepath, "r") as file:
            for line in file:
                tokens = line.strip().split(",")
                joint_name = tokens[0].replace("_END", "_End")
                character[joint_name] = np.array(
                    [float(tokens[1]), float(tokens[2]), float(tokens[3])]
                )
        char_list.append([filename, character])
    print("PARSE_ALL_CHAR :: DONE :: ", len(char_list))
    return char_list


def rand_scale(rand_range):
    # random between [1-rand_range, 1+rand_range]
    return (1 - rand_range) + 2 * rand_range * random.random()


def tweakSkelProperty(char_skel_list, rnd_hierarchy=True, charFile=None):
    char_i = random.randint(0, len(char_skel_list) - 1)
    char_file_i, char_stat_i = char_skel_list[char_i]
    char_stat_i_names = list(char_stat_i.keys())

    head_joint_list = [
        ["Hips", "LowerBack"],
        ["Spine", "Spine1", "Spine2", "Spine3", "Spine4"],
        ["Neck", "Neck1"],
        ["Head", "Head_End"],
    ]
    lleg_joint_list = [
        ["LeftHipJoint"],
        ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBase_End"],
    ]
    larm_joint_list = [
        ["LeftShoulder", "LeftDummyShoulder"],
        ["LeftArm", "LeftForeArm", "LeftHand", "LeftHand_End"],
    ]
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
    useDummyShoulder = "DummyShoulder" in char_stat_i_names

    ## Heuristics to augment the skeleton
    if rnd_hierarchy:
        if random.random() < 0.1:
            useHipJoint = True
            char_stat_i["LeftHipJoint"] = np.array([0, 0, 0])

        if random.random() < 0.1:
            useLowerBack = True
            char_stat_i["LowerBack"] = np.array([0, 0, 0])  # print(char_stat_i_names)

        if random.random() < 0.05:
            useDummyShoulder = True
            char_stat_i["LeftDummyShoulder"] = [5, 0, 0]
        elif random.random() < 0.05:
            useDummyShoulder = True
            char_stat_i["LeftDummyShoulder"] = [0, 0, 0]

        # randomize spine cnt
        default_spineCnt = spineCnt
        spineCnt += max(min(int(random.gauss(0, 1.0)), 2), -2)
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
        neckCnt += max(min(int(random.gauss(0, 1.0)), 1), -1)
        neckCnt = max(min(neckCnt, 2), 1)
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

    if not useLowerBack:
        head_joint_list[0].pop()
    if not useHipJoint:
        lleg_joint_list[0].pop()
    if not useDummyShoulder:
        larm_joint_list[0].pop()

    root_joint = head_joint_list[0][0]
    last_spine_joint = head_joint_list[1][-1]

    def flatten(folded_list):
        return list(itertools.chain(*folded_list))

    # def add_prefix(namelist, prefix): return [prefix+v for v in namelist]
    def change_prefix(namelist, old_prefix, new_prefix):
        newlist = []
        for v in namelist:
            if v.startswith(old_prefix):
                newlist.append(new_prefix + v[len(old_prefix) :])
            else:
                newlist.append(v)
        return newlist

    head_joint_list = flatten(head_joint_list)
    lleg_joint_list = flatten(lleg_joint_list)
    larm_joint_list = flatten(larm_joint_list)
    rleg_joint_list = change_prefix(lleg_joint_list, "Left", "Right")
    rarm_joint_list = change_prefix(larm_joint_list, "Left", "Right")
    ee_joint_list = [
        head_joint_list[-1],
        lleg_joint_list[-1],
        rleg_joint_list[-1],
        larm_joint_list[-1],
        rarm_joint_list[-1],
    ]

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

    # if there are duplicate, remove except for the first one(ex.lowerback in lleg_ and rleg_)
    names_used = []
    for jn in all_joint_list:
        if jn in names_used:
            all_joint_list.remove(jn)
        names_used.append(jn)

    # rnd_offset :: caution: Left / Right symmetry
    rnd_offset = {}
    for jn, offset_j in char_stat_i.items():
        if jn.startswith("Hip"):
            continue
        if jn.startswith("Right"):
            continue

        local_scale = np.array([rand_scale(0.2), rand_scale(0.2), rand_scale(0.2)])
        rnd_offset[jn] = offset_j * local_scale
        if jn.startswith("Left"):
            rnd_offset[jn.replace("Left", "Right")] = np.array(
                [-rnd_offset[jn][0], rnd_offset[jn][1], rnd_offset[jn][2]]
            )

        if jn in ee_joint_list:
            if random.random() < 0.05:
                rnd_offset[jn] = np.zeros(3)

    default_height = -sum([rnd_offset[lljoint][1] for lljoint in lleg_joint_list])
    min_ratio = 30 / default_height
    max_ratio = 120 / default_height
    global_scale = random.uniform(min_ratio, max_ratio)
    if charFile is not None:
        charFile.write(
            f", {char_file_i}, {default_height}, {default_height*global_scale}\n"
        )
    for jn in rnd_offset:
        rnd_offset[jn] *= global_scale

    height = -sum([rnd_offset[lljoint][1] for lljoint in lleg_joint_list])
    height += random.random() * 3  # [0, 3]cm random offset (+y)
    rnd_offset[root_joint] = np.array([0, height, 0])

    return rnd_offset, parent_map, all_joint_list


def convert(inputPath, outputPath):
    """
    currently, Motionbuilder scripting only supports exporting with rotation & translation for all joints
    while translation remains unchanged for joints except root.
    This function reads bvh, removes translation from joints except root, and re-save it to bvh.
    """
    outputFile = open(outputPath, "w")
    outputFile.write("HIERARCHY\n")

    indent = 0
    jointDOF = list()
    writeMode = 0  # 0: hierarchy, 1: metadata, 2: data
    prevJoint = ""
    offset = 0
    with open(inputPath, "r") as file:
        for line in file:
            # reading each word
            words = line.split()

            if writeMode == 0:
                if (words[0] == "JOINT") or (words[0] == "ROOT"):
                    prevJoint = words[1]
                    if prevJoint.endswith("_"):
                        prevJoint = prevJoint[:-1]
                    if prevJoint == "Hips" or prevJoint == "Hips_":
                        outputFile.write("  " * indent + "ROOT " + prevJoint + "\n")
                        jointDOF.append((prevJoint, 6))
                        if words[0] == "JOINT":
                            offset = 6
                    else:
                        outputFile.write("  " * indent + "JOINT " + prevJoint + "\n")
                        jointDOF.append((prevJoint, 3))
                elif words[0] == "End":
                    outputFile.write("  " * indent + "End Site\n")
                elif words[0] == "CHANNELS":
                    if prevJoint == "":
                        continue
                    if prevJoint == "Hips" or prevJoint == "Hips_":
                        outputFile.write(
                            "  " * indent
                            + "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
                        )
                    else:
                        outputFile.write(
                            "  " * indent + "CHANNELS 3 Zrotation Xrotation Yrotation\n"
                        )
                elif words[0] == "{":
                    if prevJoint == "":
                        continue
                    outputFile.write("  " * indent + "{" + "\n")
                    indent += 1
                elif words[0] == "OFFSET":
                    if prevJoint == "":
                        continue
                    outputFile.write(
                        "  " * indent
                        + "OFFSET "
                        + words[1]
                        + " "
                        + words[2]
                        + " "
                        + words[3]
                        + "\n"
                    )
                elif words[0] == "}":
                    indent -= 1
                    if indent >= 0:
                        outputFile.write("  " * indent + "}" + "\n")
                elif words[0] == "MOTION":
                    writeMode = 1
                    outputFile.write(line)

            elif writeMode == 1:
                outputFile.write(line)
                if line.startswith("Frame Time:"):
                    writeMode = 2

            elif writeMode == 2:
                words = line.split()
                offset_local = offset
                newLine = str()
                # print(len(words))
                for jointName, dof in jointDOF:
                    # print(jointName, dof, offset_local)
                    if dof == 3:
                        offset_local += 3
                    for i in range(dof):
                        newLine += words[offset_local + i] + " "
                    offset_local += dof
                newLine += "\n"
                outputFile.write(newLine)

    outputFile.close()


""" ======================================================  Mobupy Functions  ======================================================  """
""" create skeleton, characterize, plot animation, switch take, etc                 """


def generateSkel(pNamespace, rnd_offset, parent_map, all_joint_list):
    # Populate the skeleton
    skeleton = {}
    for jointName in all_joint_list:
        # add underbar to avoid conflict with reserved names
        jointName_ = jointName + "_"

        if jointName == "Reference" or jointName == "Hips":
            # If it is the reference node, create an FBModelRoot.
            joint = FBModelRoot(jointName_)

        else:
            # Otherwise, create an FBModelSkeleton.
            joint = FBModelSkeleton(jointName_)

        joint.LongName = (
            pNamespace + ":" + joint.Name
        )  # Apply the specified namespace to each joint.
        joint.Color = FBColor(0.3, 0.8, 1)  # Cyan
        joint.Size = 150  # Arbitrary size: big enough to see in viewport
        joint.Show = True  # Make the joint visible in the scene.

        # Add the joint to our skeleton.
        skeleton[jointName] = joint

    def connectPlaceJoint(jointName):
        translation = rnd_offset[jointName]
        parentName = parent_map[jointName]
        # Only assign a parent if it exists.
        if parentName != None and parentName in parent_map.keys():
            skeleton[jointName].Parent = skeleton[parentName]

        if jointName not in skeleton:
            print(jointName, "not in skeleton")
            return
        # The translation should be set after the parent has been assigned.
        skeleton[jointName].Translation = FBVector3d(translation)

    for jointName in reversed(all_joint_list):
        connectPlaceJoint(jointName)
    return skeleton


def createSkeleton_fromdata(
    pNamespace, char_skel_list, rnd_hierarchy=True, charFile=None
):
    rnd_offset, parent_map, all_joint_list = tweakSkelProperty(
        char_skel_list, rnd_hierarchy, charFile
    )
    return generateSkel(pNamespace, rnd_offset, parent_map, all_joint_list)


def characterizeSkeleton(pCharacterName, pSkeleton):
    # Create a new character.
    character = FBCharacter(pCharacterName)
    app.CurrentCharacter = character

    # Add each joint in our skeleton to the character.
    for jointName, joint in pSkeleton.items():
        slot = character.PropertyList.Find(jointName + "Link")
        if slot is not None:
            slot.append(joint)

    # Flag that the character has been characterized.
    character.SetCharacterizeOn(True)

    return character


def deselect_all():
    modelList = FBModelList()
    FBGetSelectedModels(modelList, None, True)
    for model in modelList:
        model.Selected = False


# List of all Mobu Joints
jointList = [
    "Reference",
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
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
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
]

""" [TODO]
If bvh skeleton contains joints that are not in the MotionBuilder's biped character map,
append the bvh joint name to `joint_candidates[mobu joint name]` list.

# joint_candidates: A dictionary mapping mobu joints to bvh joints (all candidates)
#     - key: mobu Joint Names
#     - value: [bvhJointName...] candidate List
"""
joint_candidates = {"LeftToeBase": ["LeftToe"], "RightToeBase": ["RightToe"]}


def CharacterizeBiped(namespace):
    myBiped = FBCharacter(f"{namespace}: mycharacter")
    app.CurrentCharacter = myBiped

    myBiped.LongName = f"{namespace}: mycharacter"

    # assign Biped to Character Mapping.
    for mobuJoint in jointList:
        modelLongName = f"{namespace}:{mobuJoint}" if namespace else mobuJoint
        myJoint = FBFindModelByLabelName(modelLongName)
        if (not myJoint) and (mobuJoint in joint_candidates):
            for bvh_joint_candidate in joint_candidates[mobuJoint]:
                modelLongName = (
                    f"{namespace}:{bvh_joint_candidate}"
                    if namespace
                    else bvh_joint_candidate
                )
                myJoint = FBFindModelByLabelName(modelLongName)
                if myJoint:
                    break
        # print(modelLongName, myJoint)
        if myJoint:
            proplist = myBiped.PropertyList.Find(mobuJoint + "Link")
            proplist.append(myJoint)

    switchOn = myBiped.SetCharacterizeOn(True)
    # print "Character mapping created for " + (myBiped.LongName)

    return myBiped


def plotAnim(char, animChar):
    """
    Receives two characters, sets the input of the first character to the second
    and plot. Return ploted character.
    """
    # if char.GetCharacterize:
    #    switchOn = char.SetCharacterizeOn(True)

    plotoBla = FBPlotOptions()
    plotoBla.ConstantKeyReducerKeepOneKey = True
    plotoBla.PlotAllTakes = False
    plotoBla.PlotOnFrame = True
    plotoBla.PlotPeriod = FBTime(0, 0, 0, 1)
    plotoBla.PlotTranslationOnRootOnly = True
    plotoBla.PreciseTimeDiscontinuities = True
    # plotoBla.RotationFilterToApply = FBRotationFilter.kFBRotationFilterGimbleKiller
    plotoBla.UseConstantKeyReducer = False
    plotoBla.ConstantKeyReducerKeepOneKey = True
    char.InputCharacter = animChar
    char.InputType = FBCharacterInputType.kFBCharacterInputCharacter
    char.ActiveInput = True
    if not char.PlotAnimation(
        FBCharacterPlotWhere.kFBCharacterPlotOnSkeleton, plotoBla
    ):
        FBMessageBox(
            "Something went wrong",
            "Plot animation returned false, cannot continue",
            "OK",
            None,
            None,
        )
        return False

    return char


def SwitchTake(pTakeName):
    iDestName = pTakeName
    for iTake in system.Scene.Takes:
        if iTake.Name == iDestName:
            system.CurrentTake = iTake


def skelExists(root, name):
    if root == None:
        return False
    if root.Name == name:
        return True
    for child in root.Children:
        if skelExists(child, name):
            return True
    return False


""" ======================================================  MAIN  ======================================================  """
# [TODO] change below variables
DATA_DIR = "/home/sunminlee/workspace/SAME/data/"
DATA_NAME = "sample"
MOTION_DIR = os.path.join(
    DATA_DIR, DATA_NAME, "motion", "bvh"
)  # place all bvh files here
CHAR_DIR = os.path.join(
    DATA_DIR, DATA_NAME, "character", "joint_pos"
)  # place all character files here
batch_size = 3  # number of motions to retarget per one skeletal variation
iter_num = 2  # number of skeletal variations per one batch
merge_skel = False  # whether all motions have the same skeleton (if True: no need to import multiple times)
fresh_start = True  # whether to start fresh, or to continue from the last point


scale_stat = parse_all_char(CHAR_DIR)
OUT_DIR = os.path.join(MOTION_DIR, "result")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
res_rel_dir = os.path.relpath(OUT_DIR, DATA_DIR)

system = FBSystem()
app = FBApplication()

import time

start_time = time.time()
total_i = 0

sOptions = FBFbxOptions(False)  # false = save options
sOptions.SaveCharacter = True
sOptions.SaveControlSet = False
sOptions.SaveCharacterExtension = False
sOptions.ShowFileDialog = False
sOptions.ShowOptionsDialog = False

app.FileNew()

logFilePath = os.path.join(MOTION_DIR, "pair.txt")
charFilePath = os.path.join(MOTION_DIR, "character.txt")

if fresh_start:
    fileList = []
    for dp, dn, filenames in os.walk(MOTION_DIR):
        if dp == OUT_DIR:
            continue
        for f in filenames:
            if f.endswith(".bvh"):
                fileList.append(os.path.join(dp, f))

    random.shuffle(fileList)
    batch_num = math.ceil(len(fileList) / batch_size)
    with open(os.path.join(MOTION_DIR, "fileList.txt"), "w") as fileListFile:
        for bi in range(batch_num):
            for m_i, filePath in enumerate(
                fileList[bi * batch_size : (bi + 1) * batch_size]
            ):
                fileListFile.write(
                    f"{bi}, {m_i}, {os.path.relpath(filePath, DATA_DIR)}\n"
                )
    logFile = open(logFilePath, "w")
    charFile = open(charFilePath, "w")
else:
    fileList = []
    with open(os.path.join(MOTION_DIR, "fileList.txt"), "r") as fileListFile:
        for line in fileListFile:
            line_list = line.strip().split(",")
            fileList.append(os.path.join(DATA_DIR, line_list[-1].strip()))
    batch_num = math.ceil(len(fileList) / batch_size)

    # find continue_bi
    done = np.zeros((batch_num, iter_num), dtype=int)
    with open(logFilePath, "r") as logfile:
        for line in logfile:
            line_tokens = line.split(",")
            if line.strip() == "":
                continue
            bi = int(line_tokens[0].strip())
            si = int(line_tokens[1].strip())
            done[bi, si] += 1
    for bi in range(batch_num):
        if not (done[bi] == batch_size).all():
            break
        continue_bi = bi + 1

    # remove log after continue_bi (partial done, so remove and redo)
    with open(logFilePath, "r") as logfile:
        lines = logfile.readlines()
    with open(logFilePath, "w") as logfile:
        for line in lines:
            line_tokens = line.split(",")
            if line.strip() == "":
                continue
            bi = int(line_tokens[0].strip())
            if bi < continue_bi:
                logfile.write(line)
            else:
                break

    # remove character log after continue_bi (partial done, so remove and redo)
    with open(charFilePath, "r") as charFile:
        lines = charFile.readlines()
    with open(charFilePath, "w") as charFile:
        for line in lines:
            line_tokens = line.split(",")
            if line.strip() == "":
                continue
            fi = int(line_tokens[0].strip())
            if fi < continue_bi * iter_num * batch_size:
                charFile.write(line)
            else:
                break

    logFile = open(logFilePath, "a")
    charFile = open(charFilePath, "a")


for bi in range(batch_num):
    app.FileNew()
    animCharList = []
    fileNameList = []
    valid_m_num = 0

    if not fresh_start and bi < continue_bi:
        total_i = (bi + 1) * batch_size * iter_num
        continue

    if bi != 0:
        logFile = open(logFilePath, "a")
        charFile = open(charFilePath, "a")

    for m_i, filePath in enumerate(fileList[bi * batch_size : (bi + 1) * batch_size]):
        fileName = os.path.relpath(filePath, DATA_DIR)

        newTake = FBTake(fileName)  #'Take_'+str(m_i))
        system.Scene.Takes.append(newTake)
        SwitchTake(fileName)
        newTake.ClearAllProperties(False)
        success = app.FileImport(filePath, merge_skel)

        valid_m_num += 1

        FBPlayerControl().SetTransportFps(FBTimeMode.kFBTimeMode30Frames)
        if merge_skel:
            prefix = None
        else:
            prefix = "BVH" if m_i == 0 else "BVH " + str(m_i)
            reference_long_name = prefix + ":reference"
            if FBFindModelByLabelName(reference_long_name):
                FBFindModelByLabelName(reference_long_name).FBDelete()

        if (not merge_skel) or (m_i == 0):
            animChar = CharacterizeBiped(prefix)
            animChar.SelectModels(True, True, True, False)
            lPlayer = FBPlayerControl()
            lPlayer.Goto(FBTime(0, 0, 0, 0))
            animCharList.append(animChar)

        fileNameList.append(fileName)
        deselect_all()

    for skel_i in range(iter_num):
        characterName = "newSkel"
        charFile.write(str(total_i))
        skeleton = createSkeleton_fromdata(
            characterName, scale_stat, rnd_hierarchy=True, charFile=charFile
        )
        character = characterizeSkeleton(characterName, skeleton)

        poseOptions = FBCharacterPoseOptions()
        poseOptions.mCharacterPoseKeyingMode = (
            FBCharacterPoseKeyingMode.kFBCharacterPoseKeyingModeFullBody
        )

        for m_i in range(valid_m_num):
            SwitchTake(fileNameList[m_i])

            # key all frames for bvh to prevent unwanted interpolation between frames
            lEndTime = system.CurrentTake.LocalTimeSpan.GetStop()
            lEndFrame = system.CurrentTake.LocalTimeSpan.GetStop().GetFrame()
            lStartFrameTime = system.CurrentTake.LocalTimeSpan.GetStart()
            lStartFrame = system.CurrentTake.LocalTimeSpan.GetStart().GetFrame()

            lRange = min(int(lEndFrame) + 1, 50)
            lPlayer = FBPlayerControl()

            for i in range(lRange):
                lPlayer.Goto(FBTime(0, 0, 0, i))
                system.Scene.Evaluate()
                lPlayer.Key()
                system.Scene.Evaluate()

            lPlayer.Goto(FBTime(0, 0, 0, 0))
            animChar = animCharList[0] if merge_skel else animCharList[m_i]
            plotAnim(character, animChar)

            logFile.write(
                f"{bi}, {skel_i}, {fileNameList[m_i]}, {res_rel_dir}/{total_i}.bvh\n"
            )
            dummyPath = os.path.join(OUT_DIR, str(total_i) + "_dummy.bvh")
            convPath = os.path.join(OUT_DIR, str(total_i) + ".bvh")
            character.SelectModels(True, True, True, True)
            app.FileExport(dummyPath)
            convert(dummyPath, convPath)
            os.remove(dummyPath)
            deselect_all()
            total_i += 1

        ## runnable check
        #     break
        # break

        character.FBDelete()
        for k, v in skeleton.items():
            v.FBDelete()

    logFile.close()
    charFile.close()
    for animChar in animCharList:
        animChar.FBDelete()
    del animCharList
