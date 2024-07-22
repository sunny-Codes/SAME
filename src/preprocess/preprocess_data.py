import os, argparse, pathlib, torch
import numpy as np

from mypath import *
from fairmotion.data import bvh
from utils.motion_utils import motion_normalize_h2s
from conversions.motion_to_graph import motion_2_states


def preprocess_motion(motion, save_path, normalized=False):
    if not normalized:
        motion, tpose = motion_normalize_h2s(motion, False)  # 0.2~3s

    skel_state, poses_state = motion_2_states(motion)  # 0.1s

    lo, go, qb, edges = skel_state
    # In [4]: for ss in skel_state: print(ss.shape)
    # (28, 3)
    # (28, 3)
    # (28,)
    # (E, 2+ number of features(=currently 2: depth, reverse_depth))

    q, p, r, pv, qv, pprev, c = poses_state
    # In [5]: for ps in poses_state: print(ps.shape)
    # (300, 28, 6)
    # (300, 28, 3)
    # (300, 4)
    # (300, 28, 3)
    # (300, 28, 6)
    # (300, 28, 3)
    # (300, 28)

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savez_compressed(
        save_path,
        lo=lo,
        go=go,
        qb=qb,
        edges=edges,
        q=q,
        p=p,
        r=r,
        pv=pv,
        qv=qv,
        pprev=pprev,
        c=c,
    )


def preprocess_single_data(
    input_dir_path,
    output_dir_path,
    append_log=False,
):
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    output_pair_path = os.path.join(output_dir_path, "pair.txt")
    valid_cnt = 0

    all_files_recursive = list(pathlib.Path(input_dir_path).rglob("*"))
    with open(output_pair_path, "a" if append_log else "w") as output_file:
        for filepath in all_files_recursive:
            filepath = str(filepath)
            if filepath.endswith(".bvh"):
                relpath = os.path.relpath(filepath, input_dir_path)
                relpath_npz = relpath[:-4] + ".npz"
                out_save_full_path = os.path.join(output_dir_path, relpath_npz)

                if os.path.exists(out_save_full_path):
                    continue  # already processed
                valid_cnt += 1

                motion = bvh.load(filepath, ignore_root_skel=True, ee_as_joint=True)
                preprocess_motion(motion, out_save_full_path, normalized=False)

                outpaths = [relpath_npz, relpath_npz]
                output_file.write("\t".join(outpaths) + "\n")


def preprocess_paired_data(
    input_dir_path,
    output_dir_path,
    append_log=False,  # True when continuing the process (killed unexpectedly) or when merging multiple dataset
):
    data_dir = os.path.relpath(input_dir_path, DATA_DIR)
    input_pair_path = os.path.join(input_dir_path, "pair.txt")

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    output_pair_path = os.path.join(output_dir_path, "pair.txt")

    valid_cnt = 0
    with open(input_pair_path, "r") as file:
        with open(output_pair_path, "a" if append_log else "w") as output_file:
            for line in file:
                words = line.rstrip().split(", ")

                batch_i, skel_i, relpath_a, relpath_b = words

                relpaths = [
                    os.path.relpath(relpath_a, data_dir),
                    os.path.relpath(relpath_b, data_dir),
                ]
                sanity_check = [
                    os.path.exists(os.path.join(input_dir_path, relpath_i))
                    for relpath_i in relpaths
                ]
                if not all(sanity_check):
                    for rp, sc in zip(relpaths, sanity_check):
                        if not sc:
                            print(
                                "err: no such file: ", os.path.join(input_dir_path, rp)
                            )
                    continue

                outpaths = []
                for relpath_i in relpaths:
                    in_motion_full_path = os.path.join(input_dir_path, relpath_i)
                    out_save_full_path = os.path.join(output_dir_path, relpath_i)
                    outpaths.append(relpath_i + ".npz")

                    if os.path.exists(out_save_full_path + ".npz"):
                        continue  # already processed
                    valid_cnt += 1

                    motion = bvh.load(
                        in_motion_full_path, ignore_root_skel=True, ee_as_joint=True
                    )
                    preprocess_motion(motion, out_save_full_path, normalized=False)

                # if there's any error while processing path_a or path_b it won't be written to output_file
                output_file.write("\t".join(outpaths) + "\n")

                # # for testing purpose
                # if valid_cnt >= 90:
                #     break


def compute_statistics(
    output_dir,
):
    pair_path = os.path.join(DATA_DIR, output_dir, "pair.txt")
    assert os.path.exists(pair_path)

    rel_paths = []
    with open(pair_path, "r") as pair_file:
        for line in pair_file:
            if line.strip() == "":
                continue
            rel_paths.extend(line.strip().split())  # src_rel_path, dst_rel_path

    npz_unique_paths = [
        os.path.join(DATA_DIR, output_dir, rel_path) for rel_path in set(rel_paths)
    ]
    npz_unique_vals = [np.load(npz_path) for npz_path in npz_unique_paths]
    normalize_keys = ["lo", "go", "q", "p", "r", "pv", "qv", "pprev"]

    mean_stds_dict = {}
    for key in normalize_keys:
        key_dim = npz_unique_vals[0][key].shape[-1]
        v_stack = np.vstack(
            [vals[key].reshape(-1, key_dim) for vals in npz_unique_vals]
        )
        mean_stds_dict[key + "_m"] = torch.Tensor(v_stack.mean(axis=0))
        mean_stds_dict[key + "_s"] = torch.Tensor(v_stack.std(axis=0))

    ms_dict_path = os.path.join(DATA_DIR, output_dir, "ms_dict.pt")
    torch.save(mean_stds_dict, ms_dict_path)
    print("saved statistics(mean/std) of", len(rel_paths), "files into", ms_dict_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="sample")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--wopair", action="store_true")
    parser.add_argument("--append_log", action="store_true")
    args = parser.parse_args()

    in_path = os.path.join(DATA_DIR, args.data, "motion", "bvh")
    out_path = os.path.join(DATA_DIR, args.data, "motion", "processed")

    if args.wopair:
        preprocess_ftn = preprocess_single_data
    else:
        preprocess_ftn = preprocess_paired_data

    preprocess_ftn(in_path, out_path, args.append_log)

    if args.train:
        compute_statistics(out_path)
