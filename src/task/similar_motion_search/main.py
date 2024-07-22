import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
from IPython import embed

KeySet = []
MotionSet = []


def dtw_distance(seq1, seq2, window_size):
    m = len(seq1)
    n = len(seq2)

    dtw = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        for j in range(n + 1):
            dtw[i, j] = np.inf
    dtw[0, 0] = 0.0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = np.linalg.norm(np.abs(seq1[i - 1] - seq2[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return (dtw[m, n]) / (m + n)


def process_files_in_folder(key_folder_path, search_folder_path):
    key_name_list = os.listdir(key_folder_path)
    key_name_list.sort()

    for file_name in key_name_list:
        file_path = os.path.join(key_folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".npy"):
            data = np.load(file_path)
            KeySet.append([data, file_name])

    file_name_list = os.listdir(search_folder_path)
    file_name_list.sort()

    for file_name in file_name_list:
        file_path = os.path.join(search_folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".npy"):
            data = np.load(file_path)
            MotionSet.append([data, file_name])


import argparse
import operator
import shutil
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_epoch", default="ckpt0", type=str)
    parser.add_argument("--key_set", default="similar_motion_search/Mixamo", type=str)
    parser.add_argument(
        "--search_set", default="similar_motion_search/MotionBuilder", type=str
    )
    args = parser.parse_args()

    ## convert bvh -> z (save at [key_folder_path/search_folder_path]/model_epoch/*.npy)
    from same.test import save_bvh_z
    from mypath import DATA_DIR

    key_bvh_dir = os.path.join(DATA_DIR, args.key_set, "bvh")
    key_npy_dir = os.path.join(DATA_DIR, args.key_set, args.model_epoch)
    save_bvh_z(args.model_epoch, key_bvh_dir, key_npy_dir)

    search_bvh_dir = os.path.join(DATA_DIR, args.search_set, "bvh")
    search_npy_dir = os.path.join(DATA_DIR, args.search_set, args.model_epoch)
    save_bvh_z(args.model_epoch, search_bvh_dir, search_npy_dir)

    ## load all z
    process_files_in_folder(key_npy_dir, search_npy_dir)

    results = []
    for motion in KeySet:
        result = {}
        r = []
        for m in MotionSet:
            result[m[1]] = dtw_distance(motion[0], m[0], len(motion[0]) * 2)
        sorted_dict = dict(sorted(result.items(), key=operator.itemgetter(1)))
        top5 = sorted(result.items(), key=operator.itemgetter(1))[:5]

        r.append(motion[1])
        for e in top5:
            r.append(e)

        print(r)

        results.append(copy.deepcopy(r))

    ## sort results by the first element of first element of list
    results.sort(key=lambda x: x[1][1])

    import csv

    f = open("search_result.csv", "w", newline="")
    writer = csv.writer(f)
    writer.writerows(results)
    f.close()
