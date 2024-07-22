import numpy as np
import torch


def argmin_nd(a):
    return np.unravel_index(np.argmin(a, axis=None), a.shape)


def argmax_nd(a):
    return np.unravel_index(np.argmax(a, axis=None), a.shape)


def argsort_nd(x):
    return np.unravel_index(np.argsort(x, axis=None), x.shape)


""" recursive dict """


def is_valid_rec_keys(recursive_dict, keys):
    if recursive_dict is None:
        return False

    dict_i = recursive_dict
    for key in keys:
        if key in dict_i.keys():
            dict_i = dict_i[key]
        else:
            return False
    return True  # dict_i


def get_recursive_keys_value(recursive_dict, keys):
    dict_i = recursive_dict
    try:
        for key in keys:
            dict_i = dict_i[key]
    except:
        assert False, f"Error in get_recursive_keys_value: {keys}"
    return dict_i


def add_rec_key_value(recursive_dict, keys, value):
    dict_i = recursive_dict
    for key in keys[:-1]:
        if key not in dict_i.keys():
            dict_i[key] = dict()
        dict_i = dict_i[key]
    dict_i[keys[-1]] = value


def recursive_print_dict(data, indent=0):
    for k, v in data.items():
        if torch.is_tensor(v):
            print("  " * (indent), k, "\t", v.shape, "\t", v.device)
        elif isinstance(v, np.ndarray):
            print("  " * (indent), k, "\t", v.shape)
        elif isinstance(v, dict):
            print(
                "  " * indent,
                k,
                "\t",
            )
            recursive_print_dict(v, indent + 1)
        else:
            print("  " * (indent), k, "\t", v)


def set_default_if_absent(dict_i, key, default_value):
    if key not in dict_i:
        dict_i[key] = default_value


# compute instance size
import sys


def get_tensor_size(tensor):
    return tensor.element_size() * tensor.numel()


# dataclass
def get_dataclass_size(dc):
    total_size = sys.getsizeof(dc)  # Basic size of the dataclass instance
    for field in dc.__dataclass_fields__.values():
        attr = getattr(dc, field.name)
        if isinstance(attr, torch.Tensor):
            total_size += get_tensor_size(attr)
        else:
            total_size += sys.getsizeof(attr)
    return total_size


def get_torch_geometric_data_size(data):
    # Estimate the size of the Data object
    # # Example data
    # from torch_geometric.data import Data
    # x = torch.randn(10, 3)  # 10 nodes with 3 features each
    # edge_index = torch.randint(0, 10, (2, 30))  # 30 edges
    # data = Data(x=x, edge_index=edge_index)

    # Basic object overhead
    data_size = sys.getsizeof(data)
    # Adding up sizes of all attributes stored as tensors
    for key, item in data:
        if torch.is_tensor(item):
            data_size += get_tensor_size(item)  # Add the size of each tensor
    return data_size
