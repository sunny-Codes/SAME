import torch
from utils.tensor_utils import cdn
from fairmotion.ops import conversions
from fairmotion.core import motion as motion_class
from same.skel_pose_graph import find_feet


def graph_2_skel(graph, batch_num):
    lo = graph.lo
    qb = graph.qb
    batch = graph.batch
    edge_index = graph.edge_index

    skel_list = []
    for bi in range(batch_num):
        lo_i = lo[batch == bi]
        qb_i = qb[batch == bi]
        edge_index_i = edge_index[:, batch[edge_index][0, :] == bi]
        J = lo_i.shape[0]  # including masked joints (will be skipped later in for loop)
        if "V_mask" in graph:
            mask_i = graph.mask[batch == bi]
            use_mask = True
            valid_ids = torch.cumsum(~mask_i, 0) - 1
        else:
            use_mask = False
            valid_ids = torch.arange(J)

        skel = motion_class.Skeleton()
        new_joint = motion_class.Joint(dof=6)
        skel.add_joint(new_joint, None)

        for j_idx in range(J):
            if use_mask and mask_i[j_idx]:
                continue
            for pid, jid in edge_index_i.transpose(1, 0):
                if pid == jid:
                    continue  # exception (dummy for root joint)
                if jid == j_idx:
                    dof = 3 if qb_i[jid] else 0
                    new_joint = motion_class.Joint(
                        dof=dof, xform_from_parent_joint=conversions.p2T(cdn(lo_i[jid]))
                    )

                    p_valid_id = valid_ids[pid]
                    new_joint.set_parent_joint(skel.joints[p_valid_id])
                    skel.add_joint(new_joint, skel.joints[p_valid_id])
                    break

        skel_list.append(skel)
    return skel_list


from utils.motion_utils import make_motion
from utils.tensor_utils import tensor_q2qR
from same.mymodel import parse_hatD, reshape_dict_consq, accum_root


def batch_graph_qrc_to_motion(
    skel_graph,
    q,
    r,
    c,
    consq_n,
    cids=None,
    first_frame_zero=False,
    contact_cleanup=False,
):
    """
    skel_graph  SkelPoseGraph(batched, may or may not include pose part)
    q           Tensor [T*sumJ, q_dim(6)]
    r           Tensor [T*B, r_dim(4)]
    c           Tensor [T*B, 1]
    batch       Tensor [0, 0, ... , B*T-1]
    """

    b = skel_graph.batch
    if skel_graph.mask.any():
        mask = skel_graph.mask
        q = q[~mask]
        c = c[~mask]
        b = b[~mask]

    qrc = {"q": q, "r": r, "c": c}
    qrc = reshape_dict_consq(qrc, consq_n)
    qR = cdn(tensor_q2qR(qrc["q"]))  # [T, sumJ, 3, 3]
    ra_T = cdn(accum_root(qrc["r"], consq_n, apply_height=True))  # [T, B, 4, 4]
    c = cdn(qrc["c"])  # [T, sumJ, 1]

    nB = (b.max() + 1) // consq_n
    skel_list = graph_2_skel(skel_graph, nB)
    if cids is not None:
        assert len(cids) == nB * 4
        cids = cdn(cids).reshape(nB, 4)

    motion_list = []
    contact_list = []

    for si, skel in enumerate(skel_list):
        Ji = cdn(torch.argwhere(b == si).flatten())
        if cids is not None:
            cid_i = cids[si]
        elif (cids is None) and contact_cleanup:
            # a bit adhoc; this asserts skel_graph has only one skel.. #TODO:FIX
            cid_i = list(find_feet(skel_graph))
        else:
            cid_i = None
        try:
            motion, contact = make_motion(
                skel,
                qR[:, Ji],
                ra_T[:, si],
                c[:, Ji],
                first_frame_zero,
                contact_cleanup,
                cid_i,
            )
        except:
            print("batch_graph_qrc_to_motion failed")
            from IPython import embed

            embed()
            exit()
        motion_list.append(motion)
        contact_list.append(contact)
    return motion_list, contact_list


def gt_recon_motion(graph_batch, consq_n):
    return batch_graph_qrc_to_motion(
        graph_batch, graph_batch.q, graph_batch.r_nopad, graph_batch.c, consq_n
    )


def hatD_recon_motion(hatD, tgt_batch, out_rep_cfg, ms_dict, consq_n):
    tgt_root_ids = tgt_batch.ptr[:-1] if hasattr(tgt_batch, "ptr") else [0]
    out = parse_hatD(hatD, tgt_root_ids, out_rep_cfg, ms_dict)
    return batch_graph_qrc_to_motion(tgt_batch, out["q"], out["r"], out["c"], consq_n)
