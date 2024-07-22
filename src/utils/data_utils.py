import copy
import numpy as np


def safe_normalize_to_one(np_vec, eps=1e-4):
    vec_norm = np.linalg.norm(np_vec)
    if vec_norm > eps:
        return np_vec / vec_norm
    else:
        return np_vec


def safe_normalize(data, axis=0, ignore_std=1e-4):
    mean, std = data.mean(axis=axis), data.std(axis=axis)
    for i in range(len(std)):
        if std[i] < ignore_std:
            std[i] = 1.0
    return (data - mean) / std, (mean, std)


def safe_normalize_pre_stat(data, mean, std, ignore_std=1e-4):
    for i in range(len(std)):
        if std[i] < ignore_std:
            std[i] = 1.0
    return (data - mean) / std


def gather_safe_normalize(data, axis=-1, ignore_std=1e-4, real=None):
    """
    safe normalize data of shape [A, B, C, D, E] (for example) with axis = 2
    it means, to squeeze all axis execpt 2, get safe_mean_std, and then reshape data back to its original data
    real must have shape of [A,B,D,E] in this case.
    """

    data_gathered_by_axis = copy.deepcopy(data)
    dim = data.shape[axis]

    # (if axis != -1) : before flattening except target axis, first transpose target axis with the last axis
    if axis != -1:
        transpose_order = list(range(len(data.shape)))
        transpose_order[-1] = axis
        transpose_order[axis] = -1
        data_gathered_by_axis = np.transpose(data_gathered_by_axis, transpose_order)

    data_reordered_dim = data_gathered_by_axis.shape
    data_gathered_by_axis = data_gathered_by_axis.reshape(-1, dim)

    # filter real only if needed
    if real is not None:
        assert data_reordered_dim[:-1] == real.shape
        real_flatten = real.flatten()
        data_gathered_by_axis_real_only = data_gathered_by_axis[real_flatten]
    else:
        data_gathered_by_axis_real_only = data_gathered_by_axis

    # get mean,std using flattened, filtered data
    mean, std = data_gathered_by_axis_real_only.mean(
        axis=0
    ), data_gathered_by_axis_real_only.std(axis=0)
    for i in range(len(std)):
        if std[i] < ignore_std:
            std[i] = 1.0

    # normalize flattened (not filtered: in order to preserve the shape when reshaping back) data
    data_gathered_by_axis = (data_gathered_by_axis - mean) / std

    # reshape back
    data_reshape_back = data_gathered_by_axis.reshape(data_reordered_dim)
    if axis != -1:
        data_reshape_back = np.transpose(data_reshape_back, transpose_order)

    return data_reshape_back, (mean, std)
