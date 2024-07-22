import torch, os
from mypath import RESULT_DIR
from same.skel_pose_graph import rep_dim
from utils.file_io import load_model_cfg

# my_gat_conv: a modified version of `torch_geometric.nn.conv.gat_conv` to enable edge masking
from same.my_gat_conv import GATConv
from torch_geometric.nn import global_max_pool


class GATEnc(torch.nn.Module):
    def __init__(self, rep_cfg, z_dim, enc_cfg):
        super(GATEnc, self).__init__()

        skel_dim = sum([rep_dim[k] for k in rep_cfg["skel"]])
        pose_dim = sum([rep_dim[k] for k in rep_cfg["pose"]])
        input_dim = skel_dim + pose_dim

        hid_lyrs = enc_cfg["hid_lyrs"]
        heads_num = enc_cfg["heads_num"]

        e_Fs = [input_dim] + hid_lyrs + [z_dim]
        self.convs = []
        for i, (fi_prev, fi) in enumerate(zip(e_Fs[:-1], e_Fs[1:])):
            if i != 0:
                fi_prev *= heads_num
            if i != len(e_Fs) - 2:
                heads = heads_num
            else:
                heads = 1
            self.convs.append(
                GATConv(fi_prev, fi, heads=heads, add_self_loops=True, fill_value=0)
            )
        self.convs = torch.nn.ModuleList(self.convs)

        self.rep_cfg = rep_cfg

    def forward(self, src_graph):
        x = src_graph.src_x
        edge_index_bi = src_graph.edge_index_bidirection
        edge_mask_bi = src_graph.edge_mask_bidirection
        batch_id = src_graph.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index_bi, edge_mask_bi)

            if (i + 1) != len(self.convs):
                x = torch.nn.ReLU()(x)

        V_mask = src_graph.mask
        if V_mask.sum() > 0:
            pool_z_x = global_max_pool(x[~V_mask], batch_id[~V_mask])
        else:
            pool_z_x = global_max_pool(x, batch_id)

        return pool_z_x


class GATDec(torch.nn.Module):
    def __init__(self, rep_cfg, z_dim, dec_cfg):
        super(GATDec, self).__init__()

        skel_dim = sum([rep_dim[k] for k in rep_cfg["skel"]])
        out_dim = sum([rep_dim[k] for k in rep_cfg["out"]])

        hid_lyrs, heads_num, tgt_all_lyr = (
            dec_cfg["hid_lyrs"],
            dec_cfg["heads_num"],
            dec_cfg["tgt_all_lyr"],
        )
        d_Fs = [skel_dim + z_dim] + hid_lyrs + [out_dim]
        self.deconvs = []
        for i, (fi_prev, fi) in enumerate(zip(d_Fs[:-1], d_Fs[1:])):
            if i != 0:
                fi_prev *= heads_num
            if i != len(d_Fs) - 2:
                heads = heads_num
            else:
                heads = 1

            if tgt_all_lyr and i != 0:
                fi_prev += skel_dim

            self.deconvs.append(
                GATConv(fi_prev, fi, heads=heads, add_self_loops=True, fill_value=0)
            )
        self.deconvs = torch.nn.ModuleList(self.deconvs)
        self.tgt_all_lyr = tgt_all_lyr

    def forward(self, src_z, tgt_graph):
        dec_x = src_z[tgt_graph.batch]
        tgt_x = tgt_graph.tgt_x

        edge_index_bi = tgt_graph.edge_index_bidirection
        edge_mask_bi = tgt_graph.edge_mask_bidirection

        for i, conv in enumerate(self.deconvs):
            if self.tgt_all_lyr or i == 0:
                dec_x = torch.hstack((dec_x, tgt_x))
            dec_x = conv(dec_x, edge_index_bi, edge_mask_bi)

            if (i + 1) != len(self.deconvs):
                dec_x = torch.nn.ReLU()(dec_x)

        return dec_x


class Model(torch.nn.Module):
    def __init__(self, model_cfg, rep_cfg):
        super(Model, self).__init__()
        z_dim = model_cfg["z_dim"]
        enc_cfg = model_cfg["Encoder"]

        self.encoder = GATEnc(rep_cfg, z_dim, enc_cfg)
        self.encoder = torch.nn.ModuleList(
            [self.encoder]
        )  # Legacy, preserved to run pretrained models;

        dec_cfg = model_cfg["Decoder"]
        self.decoder = GATDec(rep_cfg, z_dim, dec_cfg)
        self.decoder = torch.nn.ModuleList(
            [self.decoder]
        )  # Legacy, preserved to run pretrained models;

        self.rep_cfg = rep_cfg
        self.z_dim = z_dim

        if "load" in model_cfg:
            for load_i in model_cfg["load"]:
                self.load_params(
                    load_i["dir"], load_i["epoch"], load_i["prefix"], load_i["freeze"]
                )

        print(self)

    def forward(self, src_graph, tgt_graph):
        z = self.encoder[0](src_graph)
        hatD = self.decoder[0](z, tgt_graph)
        return z, hatD

    @property
    def device(self):
        return next(self.parameters()).device

    def load_params(self, dir, epoch=None, prefix="", freeze=False):
        load_model_name = (
            "last_model.pt" if epoch is None else "model_{}.pt".format(epoch)
        )
        load_path = os.path.join(RESULT_DIR, dir, load_model_name)

        saved = torch.load(load_path, map_location=self.device)
        model_dict = self.state_dict()

        pretrained_dict = {
            k: v for k, v in saved["model"].items() if k.startswith(prefix)
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        # for key in model_dict.keys():
        #     if key not in dict(model.named_parameters()).keys():
        #         print(key)
        if freeze:
            for name, param in self.named_parameters():
                if name in pretrained_dict:
                    param.requires_grad = False
        print("load model from : ", load_path, " DONE")
        print("prefix : ", prefix, ", freeze:", freeze, "\n")


def make_load_model(model_epoch, device="cuda"):
    if (len(model_epoch.split("/")) == 2) and (model_epoch.split("/")[-1].isdigit()):
        load_epoch = model_epoch.split("/")[-1]
        load_model = model_epoch[: model_epoch.find(load_epoch) - 1]
        load_epoch = int(load_epoch)
    else:
        load_epoch = None
        load_model = model_epoch
        print(model_epoch)
    config = load_model_cfg(load_model)

    model = Model(config["model"], config["representation"]).to(device)
    model.load_params(load_model, load_epoch)

    return model, config


######################## model fwd result post-processing functions ########################
from utils import tensor_utils
from fairmotion.utils import constants


# decompose hatD into each element (e.g: r,q,c)
def parse_hatD(hatD, root_ids, out_rep_cfg, ms_dict):
    out = {}
    k_start = 0
    for k in out_rep_cfg:
        k_dim = rep_dim[k]
        val = hatD[:, k_start : k_start + k_dim]
        if k + "_s" in ms_dict:
            device = val.device
            out[k + "_n"] = val
            out[k] = val * ms_dict[k + "_s"].to(device=device) + ms_dict[k + "_m"].to(
                device=device
            )
        else:
            out[k] = val

        if k == "r":
            out[k + "_n"] = out[k + "_n"][root_ids]
            out[k] = out[k][root_ids]
        k_start += k_dim  # increment element start index
    return out


# helper functions: reshape [nC*nB, ...] -> [nC, nB, ...]
def reshape_consq(v, consq_n):
    new_shape = tuple([consq_n, -1] + list(v.shape[1:]))
    return v.reshape(*new_shape)


def reshape_dict_consq(result_dict, consq_n):
    for k, v in result_dict.items():
        result_dict[k] = reshape_consq(v, consq_n)
    return result_dict


# forward-kinematics (batch by depth)
def FK(lo, qR, r, root_ids, skel_depth, skel_edge_index):

    # lo    [sumJ, 3]
    # qR    [sumJ, 3, 3]
    # r     [B, 4]
    # root_ids          [B]
    # skel_depth        [sumJ]
    # skel_edge_index   [2, sumJ]

    joint_T = tensor_utils.Tensor(constants.eye_T())[None, ...].repeat(
        lo.shape[0], 1, 1
    )
    joint_T[..., :3, :3] = qR
    joint_T[..., :3, 3] = lo
    joint_T[root_ids, 1, 3] = r[..., 3].flatten()  # height?
    joint_T[..., 3, 3] = 1

    # CAUTION:
    # we assign skel_depth = -1 for non parent-child edges (e.g. end-effector edges) to avoid wrong FK

    for i in range(skel_depth.max() + 1):
        depth_i_idx = torch.where(skel_depth == i)[0]
        depth_i_edge = skel_edge_index[:, depth_i_idx]
        depth_i_parent, depth_i_child = depth_i_edge
        joint_T[depth_i_child] = (
            joint_T[depth_i_parent].clone() @ joint_T[depth_i_child].clone()
        )

    return joint_T


# accumulate delta root(r=(dx, dz, dtheta, h)) temporally to get a full global root trajectory (except height)
def accum_root(r, consq_n, apply_height=False):
    rT = tensor_utils.tensor_r_to_rT(r)  # [T, B, 4, 4]
    rT_accum = rT.clone()
    for i in range(1, consq_n):
        rT_accum[i] = rT_accum[i - 1].clone() @ rT_accum[i].clone()
    if apply_height:
        rT_accum[..., 1, 3] = r[..., 3]
    return rT_accum


def compute_pa_pv_ra(r, p, consq_n, batch):
    rT_accum = accum_root(
        r, consq_n, apply_height=False
    )  # [T, B, 4, 4] # do not apply height here; height is already reflected in positions('p')
    rT_accum_sel_reshaped = rT_accum.reshape(-1, 4, 4)[batch].reshape(
        consq_n, -1, 4, 4
    )  # [T, sumJ, 4, 4]
    Ta = rT_accum_sel_reshaped @ tensor_utils.tensor_p2T(p)  # [T, sumJ, 4, 4]
    pa = Ta[..., :3, 3]  # [consq_n, sumJ, 3]
    pv = pa[1:] - pa[:-1]
    ra = torch.stack(
        (
            rT_accum[..., 0, 0],
            rT_accum[..., 0, 2],
            rT_accum[..., 0, 3],
            rT_accum[..., 2, 3],
        ),
        dim=-1,
    )

    return pa, pv, ra


def out_post_fwd(out, tgt_batch, ms_dict, out_rep_cfg, consq_n):
    ###### OUT ######
    # decompose hatD -> q,r,c
    tgt_root_ids = tgt_batch.ptr[:-1]
    out.update(parse_hatD(out["hatD"], tgt_root_ids, out_rep_cfg, ms_dict))
    # q (6d representation) -> qR (rotation matrix)
    out["qR"] = tensor_utils.tensor_q2qR(out["q"])
    # FK
    out["fk_T"] = FK(
        tgt_batch.lo,
        out["qR"],
        out["r"],
        tgt_root_ids,
        tgt_batch.skel_depth,
        tgt_batch.edge_index,
    )
    # reshape [nC*nB, ...] -> [nC, nB, ...]
    reshape_dict_consq(out, consq_n)
    out["p"] = out["fk_T"][..., :3, 3]
    # accumulate temporally, compute velocity
    out["pa"], out["pv"], out["ra"] = compute_pa_pv_ra(
        out["r"], out["p"], consq_n, tgt_batch.batch
    )

    ###### GROUND_TRUTH ######
    gt = {"r": tgt_batch.r_nopad, "p": tgt_batch.p, "c": tgt_batch.c}
    gt["qR"] = tensor_utils.tensor_q2qR(tgt_batch.q)
    r_m = ms_dict["r_m"].to(device=gt["r"].device)
    r_s = ms_dict["r_s"].to(device=gt["r"].device)
    gt["r_n"] = (gt["r"] - r_m) / r_s
    gt = reshape_dict_consq(gt, consq_n)
    gt["pa"], gt["pv"], gt["ra"] = compute_pa_pv_ra(
        gt["r"], gt["p"], consq_n, tgt_batch.batch
    )
    gt["batch"] = tgt_batch.batch
    # qb_single_frame
    qbi = tgt_batch.qb
    if "mask" in tgt_batch:
        qbi = qbi & (~tgt_batch.mask)
    if consq_n > 1:
        repeat_n = out["q"].shape[0]
        gt["qbi"] = qbi.reshape(repeat_n, -1)[0]

    return out, gt
