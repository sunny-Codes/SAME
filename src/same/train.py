import argparse, sys, yaml, gc, shutil
from tqdm import tqdm
import numpy as np
from functools import partial
from IPython import embed

from mypath import *
from utils import file_io, tensor_utils, network_utils
from same.mydataset import PairedDataset, get_paired_data_loader
from same.mymodel import Model, out_post_fwd
from same.skel_pose_graph import SkelPoseGraph
from same.loss import compute_loss
from same.metric import compute_metric


def load_trainer(load_cfg, model, optimizer, scheduler, device, log_path):
    if (load_cfg is None) or (load_cfg["dir"] is None):
        return

    load_dir, load_epoch = load_cfg["dir"], load_cfg["epoch"]
    load_model_name = (
        "last_model.pt" if load_epoch is None else "model_{}.pt".format(load_epoch)
    )
    load_abs_dir = os.path.join(RESULT_DIR, load_dir)
    load_path = os.path.join(load_abs_dir, load_model_name)

    saved = torch.load(load_path, map_location=device)

    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    if "scheduler" in saved and scheduler:
        scheduler.load_state_dict(saved["scheduler"])
    epoch_cnt = saved["epoch"]

    prev_log_path = os.path.join(load_abs_dir, "logs")
    if not os.path.samefile(prev_log_path, log_path):
        import shutil

        shutil.rmtree(log_path)
        shutil.copytree(prev_log_path, log_path)
    print("continue training from ", prev_log_path)
    return epoch_cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--cfg", type=str, default="mycfg")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    cfg = file_io.load_cfg(args.cfg)

    ## seed, device, printoptions
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tensor_utils.set_device(args.device)
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_cfg = cfg["train"]
    if "copy_orig_contact" in cfg["train"]:
        PairedDataset.copy_orig_contact = cfg["train"]["copy_orig_contact"]

    ## Dataset
    data_dir = os.path.join(DATA_DIR, cfg["train_data"]["dir"])
    dl = get_paired_data_loader(
        data_dir,
        train_cfg["batch_size"],
        train_cfg["consq_n"],
        shuffle=True,
        mask_option=cfg["train_data"]["mask"],
        device=args.device,
    )

    ## Model, Optimizer, Scheduler
    model = Model(cfg["model"], cfg["representation"]).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = network_utils.get_scheduler(
        optimizer, train_cfg["lr_schedule"], train_cfg["epoch_num"]
    )

    ## Log setup
    from torch.utils.tensorboard import SummaryWriter

    save_dir = os.path.join(RESULT_DIR, args.exp)
    log_path = os.path.join(save_dir, "logs")
    writer = SummaryWriter(log_path)

    with open(os.path.join(save_dir, "para.txt"), "w") as para_file:
        para_file.write(" ".join(sys.argv))
    with open(os.path.join(save_dir, "config.yaml"), "w") as config_file:
        yaml.dump(cfg, config_file)  # save all cfg (not only cfg(==cfg['train]))
    print("SAVE DIR: ", save_dir)

    # copy training ms_dict to model directory (so that it loads correctly during test time)
    shutil.copyfile(
        os.path.join(data_dir, "ms_dict.pt"), os.path.join(save_dir, "ms_dict.pt")
    )
    ms_dict = torch.load(os.path.join(data_dir, "ms_dict.pt"))

    # set SkelPoseGraph class variables
    SkelPoseGraph.skel_cfg = cfg["representation"]["skel"]
    SkelPoseGraph.pose_cfg = cfg["representation"]["pose"]
    SkelPoseGraph.ms_dict = ms_dict

    epoch_init = 0
    if "load" in train_cfg:
        epoch_init = load_trainer(
            train_cfg["load"], model, optimizer, scheduler, args.device, log_path
        )

    ## debugging
    # torch.autograd.set_detect_anomaly(True)

    print("======================= READY TO TRAIN ===================== ")
    ## Train Loop
    model.train()  # set train mode
    for epoch_cnt in tqdm(range(epoch_init, train_cfg["epoch_num"])):
        epoch_loss = {loss: 0 for loss in list(train_cfg["loss"].keys()) + ["total"]}
        epoch_metric = {metric: 0 for metric in train_cfg["metric"]}
        for bi, (src_batch, tgt_batch) in enumerate(dl):
            optimizer.zero_grad()

            ## forward
            z, hatD = model(src_batch, tgt_batch)
            z_tgt = model.encoder[0](tgt_batch)
            out = {"hatD": hatD, "z": z, "z_tgt": z_tgt}

            out, gt = out_post_fwd(
                out,
                tgt_batch,
                ms_dict,
                cfg["representation"]["out"],
                train_cfg["consq_n"],
            )
            loss = compute_loss(train_cfg["loss"], out, gt)
            metric = compute_metric(train_cfg["metric"], out, gt)

            ## backward
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg["grad_max_norm"]
            )
            optimizer.step()

            ## sanity check / debugging
            # for name, param in model.named_parameters():
            #     if not param.requires_grad: continue
            #     elif torch.isnan(param).any() or not torch.isfinite(param).any():
            #         print(name, param); embed()
            #     else:
            #         grad_norm = param.grad.norm()
            #         if torch.isnan(grad_norm).any() or not torch.isfinite(grad_norm).any():
            #             print(name, param); embed()

            ## log
            for k, v in loss.items():
                epoch_loss[k] += tensor_utils.cdn(v)
            for k, v in metric.items():
                epoch_metric[k] += tensor_utils.cdn(v)

            del src_batch, tgt_batch, out, loss, metric

        ## Write Log
        for k, v in epoch_loss.items():
            writer.add_scalar("loss/" + k, v / len(dl), epoch_cnt)
        for k, v in epoch_metric.items():
            writer.add_scalar("metric/" + k, v / len(dl), epoch_cnt)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_cnt)

        ## Save
        if (epoch_cnt % train_cfg["save_per"] == 0) or (
            epoch_cnt + 1 == train_cfg["epoch_num"]
        ):
            save_state = {
                "epoch": epoch_cnt,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(
                save_state, os.path.join(save_dir, "model_{}.pt".format(epoch_cnt))
            )
            torch.save(save_state, os.path.join(save_dir, "last_model.pt"))
            print("save done: ", epoch_cnt)

        if scheduler is not None:
            scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()
