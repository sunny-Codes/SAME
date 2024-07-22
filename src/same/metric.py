import torch
from torch_scatter import scatter
from utils import tensor_utils


def compute_p_metric(out, gt):
    gt_p, out_p = gt["p"], out["p"]
    p_dist = torch.linalg.norm((out_p - gt_p), axis=-1)  # [T, sumJ]

    sumJ = gt_p.shape[1]
    b_single_frame = gt["batch"][:sumJ]

    if "mask" in gt:
        mask = gt["mask"].reshape(gt_p.shape[0], -1)[0]
        p_dist = p_dist[:, ~mask]
        b_single_frame = b_single_frame[~mask]

    mean_p_dist = scatter(torch.mean(p_dist, 0), b_single_frame, reduce="mean")  # [B]
    mean_p_dist = torch.mean(mean_p_dist)
    return mean_p_dist


def compute_pa_metric(out, gt):
    gt_pa, out_pa = gt["pa"], out["pa"]
    pa_diff = (out_pa - gt_pa) ** 2  # [T, sumJ, 3]
    pa_dist = torch.sqrt(torch.sum(pa_diff, axis=-1))  # [T, sumJ]

    sumJ = gt_pa.shape[1]
    b_single_frame = gt["batch"][:sumJ]

    if "mask" in gt:
        mask = gt["mask"].reshape(out_pa.shape[0], -1)[0]
        pa_dist = pa_dist[:, ~mask]
        b_single_frame = b_single_frame[~mask]

    mean_pa_dist = scatter(torch.mean(pa_dist, 0), b_single_frame, reduce="mean")
    mean_pa_dist = torch.mean(mean_pa_dist)
    return mean_pa_dist


def compute_ra_xz_metric(out, gt):
    gt_ra_xz, out_ra_xz = gt["ra"][..., [2, 3]], out["ra"][..., [2, 3]]
    ra_xz_dist = torch.linalg.norm((gt_ra_xz - out_ra_xz), axis=-1)
    return torch.mean(ra_xz_dist)


def compute_ra_theta_metric(out, gt):
    gt_ra_dt, out_ra_dt = gt["ra"][..., [0, 1]], out["ra"][..., [0, 1]]
    ra_dt_dist = torch.linalg.norm((gt_ra_dt - out_ra_dt), axis=-1)
    return torch.mean(ra_dt_dist)


def compute_rtheta_metric(out, gt):
    gt_r_theta, out_r_theta = gt["r"][..., 0], out["r"][..., 0]
    r_theta_diff = torch.abs(gt_r_theta - out_r_theta).mean()
    return r_theta_diff


def compute_rdx_metric(out, gt):
    gt_r_dx, out_r_dx = gt["r"][..., 1], out["r"][..., 1]
    r_dx_diff = torch.abs(gt_r_dx - out_r_dx).mean()
    return r_dx_diff


def compute_rdz_metric(out, gt):
    gt_r_dz, out_r_dz = gt["r"][..., 2], out["r"][..., 2]
    r_dz_diff = torch.abs(gt_r_dz - out_r_dz).mean()
    return r_dz_diff


def compute_rh_metric(out, gt):
    gt_r_h, out_r_h = gt["r"][..., 3], out["r"][..., 3]
    r_h_diff = torch.abs(gt_r_h - out_r_h).mean()
    return r_h_diff


def compute_pen_metric(out, gt):
    out_pa = out["pa"]

    if "mask" in gt:
        mask = gt["mask"].reshape(out_pa.shape[0], -1)[0]
        out_pa = out_pa[:, ~mask]

    penetration = torch.mean(torch.min(out_pa[..., 1], tensor_utils.Tensor([0])))
    return penetration


def compute_jerk_metric(out, gt, fps=30):
    """positional jitter (3rd derivative): smootheness of motion.
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    https://github.com/Xinyu-Yi/TransPose/blob/main/articulate/evaluator.py#L327
    """
    CM2KM = 0.01 * 0.001
    out_pa = out["pa"]  # [T, sumJ, 3]
    if "mask" in gt:
        mask = gt["mask"].reshape(out_pa.shape[0], -1)[0]
        out_pa = out_pa[:, ~mask]

    jerk = (out_pa[3:] - 3 * out_pa[2:-1] + 3 * out_pa[1:-2] - out_pa[:-3]) * (fps**3)
    jerk = jerk.norm(dim=-1).mean() * CM2KM
    return jerk


def compute_slide_metric(out, gt):
    """foot skating
    Mode-Adaptive Neural Networks for Quadruped Motion Control [https://homepages.inf.ed.ac.uk/tkomura/dog.pdf]
    """
    H = 3  # 2.5 TODO change it as a parameter
    out_pa = out["pa"]
    if "mask" in gt:
        mask = gt["mask"].reshape(out_pa.shape[0], -1)[0]
        out_pa = out_pa[:, ~mask]

    pv = out_pa[1:] - out_pa[:-1]
    h = out_pa[1:, :, 1]
    contact = torch.clamp(2 - 2 ** (h / H), 0, 1).unsqueeze(-1)
    slide = (pv * contact).norm(dim=-1).mean()
    return slide


def compute_pa_metric_skel_aware_version(out, gt):
    gt_pa, out_pa = gt["pa"], out["pa"]
    err = (gt_pa - out_pa) * (gt_pa - out_pa)
    err /= (gt["height"] ** 2)[..., None, None]
    err = err.mean() * 1000
    return err


def compute_p_metric_skel_aware_version(out, gt):
    gt_pa, out_pa = gt["p"], out["p"]
    err = (gt_pa - out_pa) * (gt_pa - out_pa)
    err /= (gt["height"] ** 2)[..., None, None]
    err = err.mean() * 1000
    return err


def compute_ra_metric_skel_aware_version(out, gt):
    gt_ra_xz, out_ra_xz = gt["ra"][..., [2, 3]], out["ra"][..., [2, 3]]
    err = (gt_ra_xz - out_ra_xz) * (gt_ra_xz - out_ra_xz)
    err /= (gt["height"] ** 2)[..., None]
    err = err.mean() * 1000
    return err


def compute_qR_metric(out, gt):
    gt_qR, out_qR = gt["qR"], out["qR"]
    qbi = gt["qbi"]
    qR_diff = gt_qR[:, qbi].transpose(2, 3) @ out_qR[:, qbi]
    qR_diff_aa = tensor_utils.matrix_to_axis_angle(qR_diff)
    qR_diff_aa_norm = torch.linalg.norm(qR_diff_aa, dim=-1)
    return qR_diff_aa_norm.mean()


_metric_matching_ = {
    "p": compute_p_metric,
    "pa": compute_pa_metric,
    "ra_xz": compute_ra_xz_metric,
    "ra_theta": compute_ra_theta_metric,
    "r_theta": compute_rtheta_metric,
    "r_dx": compute_rdx_metric,
    "r_dz": compute_rdz_metric,
    "r_h": compute_rh_metric,
    "pen": compute_pen_metric,
    "jerk": compute_jerk_metric,
    "slide": compute_slide_metric,
    "pa_skel_aware": compute_pa_metric_skel_aware_version,
    "p_skel_aware": compute_p_metric_skel_aware_version,
    "ra_skel_aware": compute_ra_metric_skel_aware_version,
    "qR": compute_qR_metric,
}


def get_metric_function(ltype):
    return _metric_matching_[ltype]


def get_metric_names():
    return list(_metric_matching_.keys())


def compute_metric(metric_cfg, out, gt):
    metrics = dict()
    for key in metric_cfg:
        ftn = get_metric_function(key)
        metrics[key] = ftn(out, gt)
    return metrics
