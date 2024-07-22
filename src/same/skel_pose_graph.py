import torch
import torch_geometric

rep_dim = {
    "lo": 3,
    "go": 3,
    "q": 6,
    "r": 4,
    "c": 1,
    "p": 3,
    "pprev": 3,
    "pv": 3,
    "qv": 6,
}


class SkelPoseGraph(torch_geometric.data.Data):
    # class variable : should be set right after loading cfg, before using any of the data ...
    skel_cfg = []
    pose_cfg = []
    ms_dict = {}

    def __init__(self, skel_data, pose_data):
        super(SkelPoseGraph, self).__init__()
        # skel
        if skel_data is not None:
            self.lo = skel_data.lo
            self.go = skel_data.go
            self.edge_index = skel_data.edge_index

            # extra
            self.edge_feature = skel_data.edge_feature
            self.qb = skel_data.qb

        # pose
        if pose_data is not None:
            self.q = pose_data.q
            self.p = pose_data.p
            self.qv = pose_data.qv
            self.pv = pose_data.pv
            self.pprev = pose_data.pprev
            self.c = pose_data.c
            self.r_nopad = pose_data.r  # [1,rDim=4] -> r:[nJ, rDim=4] zeropad

    def normalize_x(self, key):
        val = getattr(self, key)
        if key + "_m" in self.ms_dict:
            m = self.ms_dict[key + "_m"].to(device=val.device)
            s = self.ms_dict[key + "_s"].to(device=val.device)
            val = (val - m) / s
        return val

    @property
    def skel_x(self):
        assert len(self.skel_cfg) > 0, "skel_cfg is not set"
        return torch.hstack([self.normalize_x(var) for var in self.skel_cfg])

    @property
    def edge_index_bidirection(self):
        return torch.hstack((self.edge_index, self.edge_index[[1, 0]]))

    @property
    def r(self):
        r_ = (
            self.ms_dict["r_m"]
            .repeat(self.q.shape[0], 1)
            .to(dtype=self.r_nopad.dtype, device=self.r_nopad.device)
        )
        if hasattr(self, "ptr"):
            # batched graph
            # r_: [sum(nJ), rDim]
            # self.r_nopad: [nB, rDim]
            r_[self.ptr[:-1]] = self.r_nopad
        else:
            # single graph
            # r_: [nJ, rDim]
            # self.r_nopad: [1, rDim]
            r_[0] = self.r_nopad
        return r_

    @property
    def pose_x(self):
        return torch.hstack([self.normalize_x(var) for var in self.pose_cfg])

    @property
    def src_x(self):
        return torch.hstack((self.skel_x, self.pose_x))

    @property
    def tgt_x(self):
        return self.skel_x

    @property
    def mask(self):
        if hasattr(self, "V_mask"):
            return self.V_mask
        else:
            nV, dtype, device = self.lo.shape[0], torch.bool, self.lo.device
            return torch.zeros(nV, dtype=dtype, device=device)

    @mask.setter
    def mask(self, mask):
        self.V_mask = mask

    @property
    def edge_mask_bidirection(self):
        # assume E_i = [parent_i, child_i] (except [0,0] for the root), thus nV == nE
        edge_mask_bidirection_ = self.mask.repeat(2)
        # first edge of a graph is [0,0] (typically edge=[parent,child], but root has no parent so self-edge is created)
        # self-edge will be added for all nodes inside GATConv layer, so rather mask one here to avoid redundant edges
        ptr = self.ptr[:-1] if hasattr(self, "ptr") else [0]
        edge_mask_bidirection_[ptr] = True
        edge_mask_bidirection_[self.edge_index.shape[1] + ptr] = True
        return edge_mask_bidirection_

    @property
    def skel_depth(self):
        return self.edge_feature[:, 0]


def rnd_mask(B_skel, consq_n, mask_prob=0.5, edge_thres=4, demo=None):
    # mask single frame and repeat (to avoid flickering masks for the same joints among consecutive frames)
    device = B_skel.lo.device
    nV_sf = int(B_skel.lo.shape[0] / consq_n)
    mask_sf = torch.zeros((nV_sf,), device=device, dtype=torch.bool)

    if demo == "no_mask":
        return mask_sf.repeat(consq_n)

    nB = B_skel.batch.max() + 1
    nB_sf = int(nB / consq_n)
    edge_batch = B_skel.batch[B_skel.edge_index[0]]
    edge_index_sf = B_skel.edge_index[:, edge_batch < nB_sf]

    # find end-effector
    ee = find_ee(B_skel)
    n_limb = 5
    assert len(ee) == n_limb * nB
    ee_sf = ee[: nB_sf * n_limb].reshape(nB_sf, n_limb)

    # randomly select to mask or not
    do_mask = torch.rand(nB_sf) < mask_prob  # 50 %

    # randomly select one limb per skeleton (and the corresponding end-effectors)
    rnd_ith_limb = torch.randint(0, n_limb - 1, (nB_sf, 1)).to(device=device)
    rnd_ee_sf = torch.gather(ee_sf, 1, rnd_ith_limb).flatten()

    # randomly select mask depth
    # e.g.) 0: just end-effector, 1: end-effector and its parent, 2: end-effector, parent, and grandparent, ...
    mask_ee_reach = torch.randint(0, edge_thres, (len(rnd_ee_sf),))

    # find all joints to be masked
    mask_joints = []
    ee_ascend = rnd_ee_sf
    for i in range(max(mask_ee_reach) + 1):
        mask_joints.append(ee_ascend[do_mask & (i <= mask_ee_reach)])
        ee_ascend = edge_index_sf[0, ee_ascend]  # ee-> up to parent
    mask_joints = torch.concat(mask_joints)
    mask_sf[mask_joints] = True
    return mask_sf.repeat(consq_n)


def find_ee(skel_graph):
    edge_feature, edge_index = skel_graph.edge_feature, skel_graph.edge_index
    return edge_index[1, edge_feature[:, 1] == 0]


def find_feet(skel_graph):
    # CAUTION; this function assumes a single skel
    ee = find_ee(skel_graph)
    go = skel_graph.go
    left_foot, right_foot = None, None
    for foot in ee[torch.argsort(go[ee, 1])[:2]]:
        if go[foot, 0] > 0:
            left_foot = foot
        else:
            right_foot = foot
    assert left_foot != right_foot
    assert (left_foot is not None) and (right_foot is not None)
    return left_foot, right_foot
