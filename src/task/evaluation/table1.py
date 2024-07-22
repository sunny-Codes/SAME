import os, argparse, tqdm, torch, gc
from mypath import *
from same.test import prepare_model_test
from same.mydataset import PairedDataset, get_mi_src_tgt_all_graph
from same.mymodel import out_post_fwd

from same.metric import compute_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_epoch", type=str, default="ckpt0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="test/motion/processed")
    args = parser.parse_args()

    model, cfg, ms_dict = prepare_model_test(args.model_epoch, args.device)

    # Dataset
    ds = PairedDataset()
    data_dir = os.path.join(DATA_DIR, args.data_dir)
    ds.load_data_dir_pairs(data_dir)

    metric_key = ["qR", "ra_xz", "pa", "slide", "pen"]
    metric = {key: 0.0 for key in metric_key}
    N = len(ds.mi_ri_2_fi)
    for mi in tqdm.tqdm(range(N)):
        (src_batch, tgt_batch), consq_n = get_mi_src_tgt_all_graph(
            ds, mi, 0, 0, args.device
        )
        z, hatD = model(src_batch, tgt_batch)
        out, gt = out_post_fwd(
            {"hatD": hatD, "z": z},
            tgt_batch,
            ms_dict,
            cfg["representation"]["out"],
            consq_n,
        )
        mi_metric = compute_metric(metric_key, out, gt)
        for key in metric_key:
            metric[key] += mi_metric[key].detach().item()

        del src_batch, tgt_batch, z
        gc.collect()
        torch.cuda.empty_cache()

    for key in metric_key:
        print(f"{key}: {metric[key]/N:.4f}", end="\t")

    metric_fp = os.path.join(PRJ_DIR, "table1.txt")
    if not os.path.exists(metric_fp):
        with open(metric_fp, "w") as f:
            line = "model_epoch,\t"
            for key in metric_key:
                line += f"{key},\t"
            f.write(line + "\n")
    with open(metric_fp, "a") as f:
        line = f"{args.model_epoch},\t"
        for key in metric_key:
            line += f"{metric[key]/N:.4f},\t"
        f.write(line + "\n")
