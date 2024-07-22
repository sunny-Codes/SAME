import os, torch, random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, Sampler, DataLoader
from torch_geometric.data import Batch
from functools import partial
from dataclasses import dataclass

from mypath import *
from same.skel_pose_graph import SkelPoseGraph, rnd_mask, find_feet


@dataclass
class SkelData:
    # [nJ, nDim]
    lo: torch.Tensor
    go: torch.Tensor
    qb: torch.BoolTensor
    # [nE, 2]
    edge_index: torch.LongTensor
    edge_feature: torch.LongTensor


@dataclass
class PoseData:
    # [nJ, nDim]
    q: torch.Tensor
    p: torch.Tensor
    qv: torch.Tensor
    pv: torch.Tensor
    pprev: torch.Tensor
    c: torch.BoolTensor
    # [nDim]
    r: torch.Tensor


def npz_2_data(lo, go, qb, edges, q, p, qv, pv, pprev, c, r):
    if not (np.arange(edges.shape[0]) == edges[:, 1]).all():
        edges = edges[np.argsort(edges[:, 1])]  # sort by child idx - just in case ...

    skel_data = SkelData(
        torch.Tensor(lo),
        torch.Tensor(go),
        torch.BoolTensor(qb),
        torch.LongTensor(edges[:, :2]).transpose(1, 0),
        torch.LongTensor(edges[:, 2:]),
    )

    nF = q.shape[0]
    pose_data_list = [
        PoseData(
            torch.Tensor(q[i]),
            torch.Tensor(p[i]),
            torch.Tensor(qv[i]),
            torch.Tensor(pv[i]),
            torch.Tensor(pprev[i]),
            torch.BoolTensor(c[i]).reshape(-1, 1),
            torch.Tensor(r[i]).reshape(1, -1),
        )
        for i in range(nF)
    ]
    return skel_data, pose_data_list


class PairedDataset(Dataset):
    copy_orig_contact = False

    def __init__(self):
        # skel
        self.skel_list = []
        self.pose_list = []

        ## file info.
        # nFile: number of all npz files loaded, including original and retargeted
        self.filepaths = []
        self.frame_cnts = []
        self.start_frames = []
        self.end_frames = []

        ## motion set related info
        self.mi_ri_2_fi = []
        # mi: semantic motion index (same mi means semantically identical motion)
        # ri: 0<=ri<R, R: number of retargeted motions (including original data)
        # mi_ri_2_fi[mi, ri] = fi

    def add_data(self, lo, go, qb, edges, q, p, qv, pv, pprev, c, r, filepath, mi):

        # new data
        sd, pdl = npz_2_data(lo, go, qb, edges, q, p, qv, pv, pprev, c, r)

        self.skel_list.append(sd)
        self.pose_list.extend(pdl)

        # update file info
        fi = len(self.filepaths)
        nFrame = q.shape[0]
        start = sum(self.frame_cnts)
        end = start + nFrame
        self.filepaths.append(filepath)
        self.frame_cnts.append(nFrame)
        self.start_frames.append(start)
        self.end_frames.append(end)

        # update pair-related info
        assert len(self.mi_ri_2_fi) >= mi
        if len(self.mi_ri_2_fi) == mi:
            # new semantic motion set
            self.mi_ri_2_fi.append([])
        else:
            # make sure the number of frames is consistent among retargeted dataset
            orig_fi = self.mi_ri_2_fi[mi][0]
            orig_nFrame = self.frame_cnts[orig_fi]

            assert orig_nFrame == nFrame
            if self.copy_orig_contact:
                # copy contact from the original motion (optional)
                lf, rf = find_feet(sd)
                orig_sd = self.skel_list[orig_fi]
                orig_start = self.start_frames[orig_fi]
                orig_pdl = self.pose_list[orig_start : orig_start + orig_nFrame]
                orig_lf, orig_rf = find_feet(orig_sd)
                for pdi, orig_pdi in zip(pdl, orig_pdl):
                    pdi.c[lf] = orig_pdi.c[orig_lf]
                    pdi.c[rf] = orig_pdi.c[orig_rf]

        self.mi_ri_2_fi[mi].append(fi)

    def add_data_from_npz(self, mi, npz_fp, bvh_fp=None):
        data = np.load(npz_fp)
        if bvh_fp is None:
            bvh_fp = npz_fp  # placeholder
        self.add_data(**data, filepath=bvh_fp, mi=mi)

    def load_data_dir_pairs(self, data_dir):
        pair_path = os.path.join(data_dir, "pair.txt")
        assert os.path.exists(pair_path), pair_path + " does not exist"
        bvh_prefix = os.path.join(os.path.dirname(data_dir), "bvh")

        src_id_map = {}
        with open(pair_path, "r") as pair_file:
            for line in pair_file:
                if line.strip() == "":
                    continue
                src_rel_path, dst_rel_path = line.strip().split()

                if src_rel_path in src_id_map:
                    src_id = src_id_map[src_rel_path]

                else:  # new source
                    src_id = len(self.mi_ri_2_fi)
                    src_id_map[src_rel_path] = src_id
                    npz_fp = os.path.join(data_dir, src_rel_path)
                    bvh_fp = os.path.join(
                        bvh_prefix, Path(src_rel_path).with_suffix("")
                    )
                    self.add_data_from_npz(src_id, npz_fp, bvh_fp)

                if dst_rel_path in src_id_map:
                    continue
                else:
                    npz_fp = os.path.join(data_dir, dst_rel_path)
                    bvh_fp = os.path.join(
                        bvh_prefix, Path(dst_rel_path).with_suffix("")
                    )
                    self.add_data_from_npz(src_id, npz_fp, bvh_fp)

    def get_mi_ri_fi_graph(self, mi, ri, frame):
        fi = self.mi_ri_2_fi[mi][ri]
        si = self.skel_list[fi]
        pi = self.pose_list[self.start_frames[fi] + frame]
        return SkelPoseGraph(si, pi)

    def __getitem__(self, idx):
        mi, src_ri, tgt_ri, frame = idx
        src_graph = self.get_mi_ri_fi_graph(mi, src_ri, frame)
        tgt_graph = self.get_mi_ri_fi_graph(mi, tgt_ri, frame)
        return src_graph, tgt_graph

    def get_mi_src_tgt_all(self, mi, src_ri, tgt_ri):
        assert src_ri >= 0 and src_ri < len(self.mi_ri_2_fi[mi])
        assert tgt_ri >= 0 and tgt_ri < len(self.mi_ri_2_fi[mi])
        frame_cnt = self.frame_cnts[self.mi_ri_2_fi[mi][0]]
        batch = [self[mi, src_ri, tgt_ri, frame] for frame in range(frame_cnt)]
        return batch, frame_cnt


class PairConsqSampler(Sampler):
    def __init__(self, dataset, batch_size, consq_n, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.consq_n = consq_n
        self.shuffle = shuffle

        # valid frames(considering consq_n)
        self.valid_mi_frames = np.zeros((0, 2), dtype=int)
        for mi in range(len(self.dataset.mi_ri_2_fi)):
            nFrame = self.dataset.frame_cnts[self.dataset.mi_ri_2_fi[mi][0]]
            valid_nF = nFrame - self.consq_n + 1
            new_mi = np.array([mi] * valid_nF)
            new_frames = np.arange(0, valid_nF)
            new_mi_frames = np.column_stack((new_mi, new_frames))
            self.valid_mi_frames = np.vstack((self.valid_mi_frames, new_mi_frames))

        # assign it when you want to specify specific src/tgt ri
        self.src_ri = None
        self.tgt_ri = None

    def __iter__(self):
        # random order of valid motion/frame index pairs
        if self.shuffle:
            # * Caution: random.shuffle (X) : this ftn shuffles elements independently
            np.random.shuffle(self.valid_mi_frames)

        # random src/tgt skeletons (including the original ones)
        try:  # if all retargeted motions have the same number of frames
            R = np.array(self.dataset.mi_ri_2_fi).shape[1]
            ris = np.random.randint(0, R, size=(len(self.valid_mi_frames), 2))
        except:
            ris = [
                random.sample(range(len(self.dataset.mi_ri_2_fi[mi])), 2)
                for mi in self.valid_mi_frames[:, 0]
            ]

        batch = []
        n_iter = 0
        for (mi, fi), (src_ri, tgt_ri) in zip(self.valid_mi_frames, ris):

            # override if src/tgt ri is specified
            if self.src_ri is not None:
                src_ri = self.src_ri
            if self.tgt_ri is not None:
                tgt_ri = self.tgt_ri

            batch.append((mi, src_ri, tgt_ri, fi))

            # when batch is full, yield the batch (with consq_n frames each)
            if len(batch) == self.batch_size:
                consq_batch = []
                for offset in range(self.consq_n):
                    for mi, src_ri, tgt_ri, frame in batch:
                        consq_batch.append((mi, src_ri, tgt_ri, frame + offset))
                yield consq_batch
                batch = []
                n_iter += 1

                if n_iter >= len(self):
                    break

    def __len__(self):
        return len(self.valid_mi_frames) // self.batch_size // self.consq_n


def PairedGraph_collate_fn(batch, mask_option=[], consq_n=-1, device="cpu"):
    # 'batch' : List[pair(src_graph, tgt_graph)] returned from __getitem__
    src_batch = Batch.from_data_list([item[0] for item in batch])
    tgt_batch = Batch.from_data_list([item[1] for item in batch])

    # masking can also be done in dataset's __getitem__, but it is efficient to do at once by batch when collating
    if "src" in mask_option:
        assert consq_n > 0, "to apply mask, must provide consq_n > 0"
        src_batch.mask = rnd_mask(src_batch, consq_n=consq_n)
    if "tgt" in mask_option:
        assert consq_n > 0, "to apply mask, must provide consq_n > 0"
        tgt_batch.mask = rnd_mask(tgt_batch, consq_n=consq_n)

    return src_batch.to(device), tgt_batch.to(device)


def get_paired_data_loader(data_dir, batch_size, consq_n, shuffle, mask_option, device):
    ds = PairedDataset()
    ds.load_data_dir_pairs(data_dir)
    sampler = PairConsqSampler(
        ds, batch_size=batch_size, consq_n=consq_n, shuffle=shuffle
    )

    dl = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=partial(
            PairedGraph_collate_fn,
            mask_option=mask_option,
            consq_n=consq_n,
            device=device,
        ),
    )
    return dl


def get_mi_src_tgt_all_graph(dataset, mi, src_ri, tgt_ri, device):
    all_batch_idx, consq_n = dataset.get_mi_src_tgt_all(mi, src_ri, tgt_ri)
    return PairedGraph_collate_fn(all_batch_idx, device=device), consq_n
