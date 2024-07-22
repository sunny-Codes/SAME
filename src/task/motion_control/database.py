from abc import ABC, abstractclassmethod
from mypath import *

from IPython import embed
import numpy as np
import os, copy, pickle


def parse_ann(ann_dir_path):
    file_list = os.listdir(ann_dir_path)
    file_list = list(filter(lambda x: x.endswith(".lab"), file_list))

    ann_dict = dict()
    action2int = {"base": 0}
    int2action = {0: "base"}

    for file_name in file_list:
        start_end_list = list()
        file_path = os.path.join(ann_dir_path, file_name)
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("startFrame:"):
                start = int(line.split(":")[-1])
            elif line.startswith("endFrame:"):
                end = int(line.split(":")[-1])
            elif line.startswith("type:"):
                action_type = line.split(":")[-1].strip()
                if action_type == "":
                    action_type = "base"

                if action_type in action2int:
                    action_type_int = action2int[action_type]
                else:
                    if len(int2action) == 0:
                        action_type_int = 0
                    else:
                        action_type_int = max(list(int2action.keys())) + 1
                    action2int[action_type] = action_type_int
                    int2action[action_type_int] = action_type
            elif line.startswith("subtype:"):
                subtype = line.split(":")[-1]
            elif line.startswith("interactionFrame:"):
                if (subtype == "foot_inversion") or "breakage" in subtype:
                    print(file_name, start, end, action_type, subtype, ": PASS(broken)")
                    continue
                intFrame = line.split(":")[-1]
                if intFrame == "":
                    intFrame = None
                else:
                    intFrame = int(intFrame)
                start_end_list.append((start, end, action_type, intFrame))

        bvh_name = file_name[:-4] + ".bvh"  # .lab -> .bvh
        ann_dict[bvh_name] = start_end_list

    for key, value in action2int.items():
        print(key, value)

    # # merge consecutive frames
    # for file_name in ann_dict:
    #     frame_list = ann_dict[file_name]
    #     if len(frame_list) == 1:
    #         merged_frame_list = [frame_list]
    #     else:
    #         frame_list.sort(key= lambda x: x[0])
    #         merged_frame_list = []
    #         for fli, fl in enumerate(frame_list):
    #             if fli == 0:
    #                 merged_frame_list.append([fl])
    #             else:
    #                 if frame_list[fli-1][1]+1 != frame_list[fli][0]:
    #                     merged_frame_list.append([fl])
    #                 else:
    #                     merged_frame_list[-1].append(fl)
    #     ann_dict[file_name] = merged_frame_list

    return ann_dict, action2int, int2action


class Database(ABC):
    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def construct_database(self, data_dir):
        pass

    @abstractclassmethod
    def search_closest(self, query):
        pass

    def get_file_frame(self, db_frame):
        for file_i, file_name, db_file_start, db_file_end in self.db_file_frames:
            if (db_frame >= db_file_start) and (db_frame < db_file_end):
                return file_i, file_name, (db_frame - db_file_start)

    def get_dbframe(self, file_id, file_frame):
        assert (file_id >= 0) and (file_id < len(self.p_datas))
        assert (file_frame >= 0) and (file_frame < len(self.p_datas[file_id]))
        return self.db_file_frames[file_id][1] + file_frame

    def get_trajectory_index_clamp(self, frame, offset):
        for file_i, file_name, db_file_start, db_file_end in self.db_file_frames:
            if (frame >= db_file_start) and (frame < db_file_end):
                return np.clip(frame + offset, db_file_start, db_file_end - 1)
        assert False

    def get_remain_frame(self, frame):
        for file_i, file_name, db_file_start, db_file_end in self.db_file_frames:
            if (frame >= db_file_start) and (frame < db_file_end):
                return (db_file_end - 1) - frame

    def get_feature(self, db_frame):
        assert (db_frame >= 0) and (db_frame < self.total_frames)
        return self.features[db_frame]

    def save_db(self, file_path):
        with open(file_path, "wb") as save_file:
            pickle.dump(self.__dict__, save_file)

    def load_db(self, file_path):
        with open(file_path, "rb") as save_file:
            saved_var_dict = pickle.load(save_file)
            self.__dict__.update(saved_var_dict)
