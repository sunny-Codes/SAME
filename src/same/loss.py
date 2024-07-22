import torch
from functools import partial
from utils import tensor_utils


def default_mse_loss(loss_key, out, gt):
    y_gt, y_out = gt[loss_key], out[loss_key]

    if ("mask" in gt) and (loss_key != "r") and (loss_key != "ra"):
        mask = gt["mask"].reshape(-1, y_gt.shape[1])[0]
        y_gt, y_out = y_gt[:, ~mask], y_out[:, ~mask]

    if y_gt.dtype == torch.bool:
        y_gt = y_gt.float()
    if y_out.dtype == torch.bool:
        y_out = y_out.float()

    return torch.nn.MSELoss()(y_gt, y_out)


def compute_q_loss(out, gt):
    qbi = gt["qbi"]
    return torch.nn.MSELoss()(gt["qR"][:, qbi], out["qR"][:, qbi])


def compute_cv_loss(out, gt):
    gt_c = gt["c"][:, :, 0]
    pv_norm = torch.norm(out["pv"], dim=-1)
    return torch.mean(gt_c[1:] * pv_norm)


def compute_pen_loss(out, gt):
    out_pa = out["pa"]
    if "mask" in gt:
        mask = gt["mask"].reshape(out_pa.shape[0], -1)[0]
        out_pa = out_pa[:, ~mask]
    return torch.mean(torch.min(out_pa[..., 1], tensor_utils.Tensor([0])) ** 2)


def compute_jerk_loss(out, gt, fps=30):
    CM2KM = 0.01 * 0.001
    out_pa = out["pa"]

    if "mask" in gt:
        mask = gt["mask"].reshape(out_pa.shape[0], -1)[0]
        out_pa = out_pa[:, ~mask]

    jerk = (out_pa[3:] - 3 * out_pa[2:-1] + 3 * out_pa[1:-2] - out_pa[:-3]) * (fps**3)
    jerk = jerk.norm(dim=-1) * CM2KM
    jerk_loss = (jerk**2).mean()
    return jerk_loss


def compute_slide_loss(out, gt):
    H = 3  # 2.5 TODO change it as a parameter
    out_pa = out["pa"]
    if "mask" in gt:
        mask = gt["mask"].reshape(out_pa.shape[0], -1)[0]
        out_pa = out_pa[:, ~mask]

    pv = out_pa[1:] - out_pa[:-1]
    h = out_pa[1:, :, 1]
    # contact = torch.clamp(2- 2**(h/H), 0, 1).unsqueeze(-1)
    contact = torch.clamp(1 - h / H, 0, 1).unsqueeze(-1)
    slide = (pv * contact).norm(dim=-1)
    slide_loss = (slide**2).mean()
    return slide_loss


def compute_z_loss(out, gt):
    z_loss = torch.nn.MSELoss()(out["z"], out["z_tgt"])
    return z_loss


# def compute_kernel(x, y):
#     x_size = x.size(0)
#     y_size = y.size(0)
#     dim = x.size(1)
#     tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
#     tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
#     kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
#     return torch.exp(-kernel_input) # (x_size, y_size)

# def compute_mmd(x, y):
#     x_kernel = compute_kernel(x, x)
#     y_kernel = compute_kernel(y, y)
#     xy_kernel = compute_kernel(x, y)
#     mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
#     return mmd

# def mmd_loss(z):
#     prior_samples = torch.randn_like(z)
#     return compute_mmd(z, prior_samples)


def compute_z_gauss_loss(out, gt):
    z = out["z"]
    mu = z.mean(dim=0)
    std = z.std(dim=0)
    return torch.mean(mu**2 + (std - 1) ** 2)


_loss_matching_ = {
    "p": partial(default_mse_loss, "p"),
    "r": partial(default_mse_loss, "r_n"),
    "c": partial(default_mse_loss, "c"),
    "pv": partial(default_mse_loss, "pv"),
    "z": compute_z_loss,
    "q": compute_q_loss,
    "cv": compute_cv_loss,
    "pen": compute_pen_loss,
    "jerk": compute_jerk_loss,
    "slide": compute_slide_loss,
    "z_gauss": compute_z_gauss_loss,
}


def get_loss_function(ltype):
    return _loss_matching_[ltype]


def get_loss_names():
    return list(_loss_matching_.keys())


def compute_loss(loss_cfg, out, gt):
    losses = dict()
    loss_sum = 0.0
    for key, weight in loss_cfg.items():
        if weight > 1e-8:
            ftn = get_loss_function(key)
            try:
                loss = ftn(out, gt)
            except:
                print(key, "error occured")
                from IPython import embed

                embed()
                exit()
            if torch.isnan(loss):
                print(key, "NaN occured")
                from IPython import embed

                embed()
                exit()
            weighted_loss = loss * weight
            losses[key] = weighted_loss
            loss_sum += weighted_loss
    losses["total"] = loss_sum
    return losses
