import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

import GPUtil, gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

use_cuda = True
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
Tensor = FloatTensor


def set_device(device):
    globals()["use_cuda"] = device != "cpu"
    globals()["FloatTensor"] = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    globals()["LongTensor"] = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    globals()["IntTensor"] = torch.cuda.IntTensor if use_cuda else torch.IntTensor
    globals()["ByteTensor"] = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
    globals()["BoolTensor"] = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
    globals()["Tensor"] = FloatTensor

    torch_device = torch.device(device if (torch.cuda.is_available()) else "cpu")
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.set_device(torch_device)


def print_gpu_usage(gpu_no=0):
    GPUtil.showUtilization()
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(gpu_no)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f"total    : {info.total}")
    print(f"free     : {info.free}")
    print(f"used     : {info.used}")
    print(torch.cuda.memory_summary())
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(type(obj), obj.size())
        except:
            pass


# def proj_tensor(u, v):
#     # u, v: [B, ..., D]
#     # projet v to u
#     B, D = u.shape[0], u.shape[-1]
#     uv = torch.sum(u * v, axis=-1)
#     uu = torch.sum(u * u, axis=-1)
#     a = (uv / uu).unsqueeze(dim=-1)
#     repeat_dim = [1] * len(u.shape)
#     repeat_dim[-1] = D
#     a = a.repeat(repeat_dim)
#     return a * u


def tensor_q2qR(q):
    """
    input q: [..., T, q_dim(=6)]
    output qR: [..., T, 3, 3]
    """
    q_shape = q.shape
    q_reshape = tuple(list(q_shape[:-1]) + [2, 3])
    q_ = q.reshape(q_reshape)  # [..., T, 2, 3]

    v1 = q_[..., 0, :]
    v2 = q_[..., 1, :]
    e1 = torch.nn.functional.normalize(v1, dim=-1)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1  # v2 - proj_tensor(v1, v2)
    e2 = torch.nn.functional.normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)

    # a1, a2 = d6[..., :3], d6[..., 3:]
    # b1 = torch.nn.functional.normalize(a1, dim=-1)
    # b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    # b2 = torch.nn.functional.normalize(b2, dim=-1)
    # b3 = torch.cross(b1, b2, dim=-1)
    # return torch.stack((b1, b2, b3), dim=-2)


from fairmotion.utils import constants


def tensor_r_to_rT(r_dn, apply_height=False):
    """
    input r_dn  :   [..., r_dim]
    output rT   :   [..., 4, 4]
    """

    dtheta, dx, dz, h = r_dn[..., 0], r_dn[..., 1], r_dn[..., 2], r_dn[..., 3]
    dcos, dsin = torch.cos(dtheta), torch.sin(dtheta)

    repeat_shape = tuple(list(r_dn.shape[:-1]) + [1, 1])

    root_T = Tensor(np.tile(constants.eye_T(), repeat_shape))
    root_T[..., 0, 0] = dcos
    root_T[..., 0, 2] = dsin
    root_T[..., 0, 3] = dx
    root_T[..., 2, 0] = -dsin
    root_T[..., 2, 2] = dcos
    root_T[..., 2, 3] = dz

    if apply_height:
        root_T[..., 1, 3] = h

    return root_T


def tensor_p2T(p):
    reshape = tuple(list(p.shape[:-1]) + [4, 4])
    T = Tensor(constants.eye_T()).expand(*reshape).clone()
    T[..., :3, 3] = p
    return T


def cdn(torch_tensor):
    return torch_tensor.cpu().detach().numpy()


# below are rotation_conversions code copied from pytorch3d
# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))
