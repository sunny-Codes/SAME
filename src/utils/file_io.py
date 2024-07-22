"""File input / output."""

import os
import yaml
from typing import Dict
from mypath import *


def get_config_path(cfg_name):
    return os.path.join(CFG_DIR, cfg_name + ".yml")


def load_yaml_file(path) -> Dict:
    with open(path) as f:
        d = yaml.load(f, Loader=yaml.SafeLoader)
    return d


def is_valid_cfg(cfg, key_list):
    cur_cfg = cfg
    for key in key_list:
        if key in cur_cfg:
            cur_cfg = cur_cfg[key]
        else:
            return False
    return True


def fillin_default_value(obj, var_name, var_default):
    if not hasattr(obj, var_name):
        setattr(obj, var_name, var_default)


def filllin_missing_value(cfg, key_list, default_val):
    if is_valid_cfg(cfg, key_list):
        return

    cur_cfg = cfg
    for i, key in enumerate(key_list):
        if key in cfg:
            cur_cfg = cur_cfg[key]
        elif i + 1 == len(key_list):
            cur_cfg[key] = default_val
        else:
            cur_cfg[key] = dict()
            cur_cfg = cur_cfg[key]


def nested_fillin_missing(cfg, default_cfg):
    """Updates a nested dictionary `cfg` with new values `default_cfg` if the key is missing"""
    for k, v in default_cfg.items():
        if isinstance(v, dict):
            cfg[k] = nested_fillin_missing(cfg.get(k, {}), v)
        elif k not in cfg:
            cfg[k] = v
    return cfg


def update_yaml_None(config_dict):
    for key in config_dict:
        if config_dict[key] == "None":
            config_dict[key] = None
        if type(config_dict[key]) is dict:
            update_yaml_None(config_dict[key])


def load_cfg(cfg_name):
    config = load_yaml_file(get_config_path(cfg_name))
    update_yaml_None(config)
    return config


def fillin_default(cfg):
    def_cfg = load_cfg("superset_default")
    nested_fillin_missing(cfg, def_cfg)
    return cfg


def get_cfg(cfg, key_list):
    if not is_valid_cfg(cfg, key_list):
        return None

    cur_cfg = cfg
    for key in key_list:
        cur_cfg = cur_cfg[key]
    return cur_cfg


def set_cfg_recursive(cfg, key_list, val):
    if len(key_list) == 0:
        return
    key = key_list.pop(0)
    if len(key_list) == 0:
        cfg[key] = val
    else:
        set_cfg_recursive(cfg.get(key, {}), key_list, val)


def load_model_cfg(load_model):
    model_dir = os.path.join(RESULT_DIR, load_model)
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        update_yaml_None(config)
    return config


def create_dir_if_not_exists(file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def uniquify_log_dir(log_dir_in) -> str:
    log_dir_out = log_dir_in
    if os.path.exists(log_dir_out):
        counter = 0
        while os.path.exists(log_dir_out):
            log_dir_out = log_dir_in + "_" + str(counter)
            counter += 1
        print(
            f"FileIO: since folder {log_dir_in} already exist, changed to {log_dir_out}."
        )
    return log_dir_out


def get_unique_filepath(folder, postfix):
    """create filename based on what files already exist in folder (don't overwrite)."""
    name = os.path.join(folder, "{:02d}_" + postfix)
    counter = 0
    while os.path.isfile(name.format(counter)):
        counter += 1
    return name.format(counter)


def get_filepath(folder, postfix, counter=0):
    return os.path.join(folder, "{:02d}_" + postfix).format(counter)


def get_all_files(directory, ending: str):
    """Returns list of all file paths with specific ending e.g. glb. also in subdirectories."""
    all_files = []
    for root, subdirs, files in os.walk(directory):
        files.sort()
        for name in files:
            if name.endswith(ending):
                all_files.append(os.path.join(root, name))

    if len(all_files) == 0:
        raise ValueError(f"No files in directory: {directory}")
    return all_files


def override_by_args(config, args):
    args_cfg = {key: val for key, val in vars(args).items() if val is not None}
    config.update(args_cfg)


# def parser_add_def_args(parser):
#     def_cfg = load_cfg('superset_default')
#     parser_recursive_def_args(parser, def_cfg, '')
# def parser_recursive_def_args(parser, cfg, prefix):
#     for key, val in cfg.items():
#         if isinstance(val, dict):
#             parser_recursive_def_args(parser, val, prefix+key+"/")
#         else:
#             from IPython import embed; embed()
#             parser.add_argument('--'+prefix+key, type=type(val), default='1')
#             cfg[key] = val


# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# import json
# import pickle
# import logging
# from pathlib import Path

# def load_config(cfg_name):
#     """Creates a single config dictionary, possibly composed of multiple yaml files."""
#     cfg_path = get_config_path(cfg_name)
#     cfg = load_yaml_file(cfg_path)
#     cfg_expanded = copy.deepcopy(cfg)
#     expand_yaml_string_with_dict(cfg, cfg_expanded)
#     cfg_out = update_nested_dict(cfg_expanded, cfg)
#     remove_yaml_key(cfg_out)
#     return cfg_out

# def expand_yaml_string_with_dict(cfg, cfg_expanded):
#     """
#     Searches for strings that end in .yaml and replaces that
#     key with a dictionary generated from the yaml file.
#     """
#     for key, value in cfg.items():
#         if isinstance(value, str):
#             if value.endswith(".yaml"):
#                 cfg_yaml = load_yaml_file(get_config_path(f"partials/{value}"))  # set dictionary as just the yaml file.
#                 cfg_expanded.update(cfg_yaml)

#         if isinstance(value, dict):
#             expand_yaml_string_with_dict(cfg[key], cfg_expanded[key])

# def update_nested_dict(d, u):
#     """Updates a nested dictionary d with new values u."""
#     for k, v in u.items():
#         if isinstance(v, dict):
#             d[k] = update_nested_dict(d.get(k, {}), v)
#         else:
#             d[k] = v
#     return d

# def remove_yaml_key(d):
#     for k, v in d.items():
#         if k == "_default":
#             del d[k]

# def load_training_cfg(trial_folder):
#     """Loads config that has already been used to train a policy."""
#     cfg_path = os.path.join(trial_folder, "config.pkl")
#     with open(cfg_path, "rb") as f:
#         cfg = pickle.load(f)
#     logging.debug(f"Config:\n {json.dumps(cfg, indent=4)}")
#     return cfg

# def get_latest_model_path(trial_folder, iteration=None):
#     try:
#         latest_model = [x for x in sorted(os.listdir(trial_folder)) if x.startswith("model")][-1]
#     except IndexError:
#         logging.error(f"No pytorch model saved in dir {trial_folder}.")
#         raise
#     model = latest_model if iteration is None else "model_{:06d}.pt".format(iteration)
#     path = trial_folder + "/" + model
#     return path

# def get_full_paths(trial_folder, motion_clips, type):
#     """type is either "sim" or "ref" """
#     return [trial_folder + "log_" + clip + "_" + type + ".mmo" for clip in motion_clips]
