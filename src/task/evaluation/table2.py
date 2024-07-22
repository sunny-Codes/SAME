import os, argparse, tqdm, time
from mypath import *
from same.test import prepare_model_test, retarget
from same.mydataset import PairedDataset, get_mi_src_tgt_all_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_epoch", type=str, default="ckpt0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="evaluation/motion/processed")
    args = parser.parse_args()

    model, cfg, ms_dict = prepare_model_test(args.model_epoch, args.device)

    # Dataset
    ds = PairedDataset()
    data_dir = os.path.join(DATA_DIR, args.data_dir)
    ds.load_data_dir_pairs(data_dir)

    int_err, cross_err = 0.0, 0.0
    int_cnt, cross_cnt = 0, 0

    def compute_err(mi, src_ri, tgt_ri):
        # Metric from <Skeleton-Aware Networks for Deep Motion Retargeting>
        # https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/retargeting/get_error.py#L47-L55
        (src_batch, tgt_batch), consq_n = get_mi_src_tgt_all_graph(
            dataset=ds, mi=mi, src_ri=src_ri, tgt_ri=tgt_ri, device=args.device
        )
        src_motion, tgt_motion, out_motion = retarget(
            model,
            src_batch,
            tgt_batch,
            ms_dict,
            out_rep_cfg=cfg["representation"]["out"],
            consq_n=consq_n,
        )
        # character height: Max(joints' height at t-pose)
        height = tgt_batch.go[:, 1].max().item()
        pos_ref = tgt_motion.positions(local=False)
        pos = out_motion.positions(local=False)
        err = (pos - pos_ref) * (pos - pos_ref)
        err /= height**2
        return err.mean() * 1000

    st = time.time()
    for mi in tqdm.tqdm(range(len(ds.mi_ri_2_fi))):
        R = len(ds.mi_ri_2_fi[mi])
        # sanity check
        # filepaths = [ds.filepaths[ds.mi_ri_2_fi[mi][ri]] for ri in range(R)]
        # characters = [os.path.basename(os.path.dirname(fp)) for fp in filepaths]
        # assert characters == ['BigVegas', 'Goblin_m', 'Mousey_m', 'Mremireh_m', 'Vampire_m']

        # Comparison with <Skeleton-Aware Networks for Deep Motion Retargeting>
        # cross: BigVegas -> Goblin_m, Mousey_m, Mremireh_m, Vampire_m
        # internal: Goblin_m, Mousey_m, Mremireh_m, Vampire_m <->
        # https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/retargeting/test.py
        for tgt_ri in range(1, R):
            cross_err += compute_err(mi, 0, tgt_ri)
            cross_cnt += 1
        for src_ri in range(1, R):
            for tgt_ri in range(1, R):
                int_err += compute_err(mi, src_ri, tgt_ri)
                int_cnt += 1

    print(f"internal: {int_err/int_cnt:.4f}, cross: {cross_err/cross_cnt:.4f}")

    metric_fp = os.path.join(PRJ_DIR, "table2.txt")
    if not os.path.exists(metric_fp):
        with open(metric_fp, "w") as f:
            f.write("model_epoch,\tinternal,\tcross\n")
    with open(metric_fp, "a") as f:
        f.write(
            f"{args.model_epoch},\t{int_err/int_cnt:.5f},\t{cross_err/cross_cnt:.5f}\n"
        )
