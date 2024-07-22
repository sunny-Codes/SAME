import os, argparse
from mypath import *
from utils.skel_gen_utils import create_random_skel
from default_veiwer import get_default_viewer
from same.test import prepare_model_test, bvh_2_graph_z, decode_z_skel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_epoch", type=str, default="ckpt0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="arithmetic/base/")
    parser.add_argument(
        "--m_plus", type=str, default="arithmetic/jackie_waving_one_slow.bvh"
    )
    parser.add_argument("--m_minus", type=str, default="arithmetic/Idle_legzero.bvh")
    parser.add_argument("--scale", type=float, default=1.0)

    args = parser.parse_args()
    model, cfg, ms_dict = prepare_model_test(args.model_epoch, args.device)

    base_data_dir = os.path.join(DATA_DIR, args.data_dir)
    motions, graphs, zs, filenames = [], [], [], []
    for filename in os.listdir(base_data_dir):
        bvh_fp = os.path.join(base_data_dir, filename)
        if not bvh_fp.endswith(".bvh"):
            continue
        motion, graph_batch, z = bvh_2_graph_z(model, bvh_fp)
        motions.append(motion)
        graphs.append(graph_batch)
        zs.append(z)
        filenames.append(filename)

    motion_plus, graph_plus, z_plus = bvh_2_graph_z(
        model, os.path.join(DATA_DIR, args.m_plus)
    )
    motion_minus, graph_minus, z_minus = bvh_2_graph_z(
        model, os.path.join(DATA_DIR, args.m_minus)
    )

    def arithmetic_mi(mi):
        # print(filenames[mi])
        z_i = zs[mi]
        m_len = min(len(z_i), len(z_plus), len(z_minus))
        # arithmetic operation on z-space
        z_edited = z_i[:m_len] + (z_plus[:m_len] - z_minus[:m_len]) * args.scale
        # decode edited z to random skeleton
        rnd_tgt_skel = create_random_skel()
        out_motion, _ = decode_z_skel(model, z_edited, rnd_tgt_skel, ms_dict)
        viewer.update_motions(
            [motions[mi], motion_plus, motion_minus, out_motion], 100, linear=True
        )
        viewer.mi = mi

    from default_veiwer import get_default_viewer

    viewer = get_default_viewer(argparse.Namespace(imgui=False))

    arithmetic_mi(0)

    def extra_key_callback(key):
        if key == b"m":
            next_mi = (viewer.mi + 1) % len(motions)
            arithmetic_mi(next_mi)
            return True
        return False

    viewer.extra_key_callback = extra_key_callback
    viewer.run()

# task/arithmetic/new_main.py
