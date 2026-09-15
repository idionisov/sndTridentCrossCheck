#!/usr/bin/env python3

import os
import sys
import glob
import time
import argparse
import ROOT
from snd import DataManager, print_yield_header, print_yield_summary

def main():
    parser = argparse.ArgumentParser(
        description="Calculate physical event yields for trident processes from Monte Carlo ROOT files without writing any files to disk."
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_pattern",
        nargs="?",
        default="/eos/user/i/idioniso/sndMuTri/out/trimuon_all_*.root",
        help="Input ROOT file path or wildcard pattern (default: %(default)s)"
    )
    parser.add_argument(
        "--lumi", "--L-lhc",
        dest="lumi_target",
        type=float,
        default=1.0,
        help="Target integrated luminosity to normalize yields to in fb^-1 (default: 1.0)"
    )
    parser.add_argument(
        "--L-mc",
        dest="lumi_mc",
        type=float,
        default=0.025, # 1/40 fb^-1
        help="Total integrated luminosity to which MC dataset corresponds in fb^-1 (default: 0.025 = 1/40 fb^-1)"
    )
    parser.add_argument(
        "--z-range",
        dest="target_z_range",
        nargs=2,
        type=float,
        default=[260.0, 355.0],
        metavar=("Z_MIN", "Z_MAX"),
        help="Target region Z boundaries [Z_MIN, Z_MAX] in cm (default: 260.0 355.0)"
    )
    parser.add_argument(
        "-j", "--threads",
        dest="num_threads",
        type=int,
        default=8,
        help="Number of worker threads for RDataFrame (default: %(default)s)"
    )

    args = parser.parse_args()

    target_z_min, target_z_max = args.target_z_range

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    data = DataManager(args.input_pattern, num_threads=args.num_threads)
    weight_scale = args.lumi_target / args.lumi_mc if args.lumi_mc > 0 else 1.0

    print_yield_header(
        data=data,
        target_z_min=target_z_min,
        target_z_max=target_z_max,
        lumi_mc=args.lumi_mc,
        lumi_target=args.lumi_target,
        num_threads=args.num_threads,
        input_pattern=args.input_pattern,
    )

    has_precomputed = data.has_branch("mc_weight") and data.has_branch("proc_type")

    t0 = time.time()
    df = data.rdf()

    if has_precomputed:
        if data.has_branch("is_signal"):
            df_sig = df.Filter("is_signal == true")
        else:
            df_sig = df

        df_sig = df_sig.Define("scaled_weight", f"mc_weight * {weight_scale}")

        if data.has_branch("vtx_z"):
            df_sig = df_sig.Define("dyn_region_type", f"vtx_z < {target_z_min} ? 1 : (vtx_z < {target_z_max} ? 2 : 3)")
        else:
            df_sig = df_sig.Define("dyn_region_type", "region_type")
    else:
        config = ROOT.snd.trident.TridentTruthConfig()
        config.targetZMin = target_z_min
        config.targetZMax = target_z_max
        config.weightScale = weight_scale
        config.zMin = -99999.0
        config.zMax = 99999.0
        config.useFiducial = False
        config.processMask = 7

        processor = ROOT.snd.trident.TridentTruthProcessor(config)

        df_sig = (
            df.Define("truth", processor, ["MCTrack"])
              .Filter("truth.hasCandidate")
              .Define("proc_type", "truth.procType")
              .Define("dyn_region_type", "truth.regionType")
              .Define("scaled_weight", "truth.scaledWeight")
        )

    actions = {}
    actions["total_count"] = df_sig.Count()
    actions["total_yield"] = df_sig.Sum("scaled_weight")

    processes = [
        (0, "Genuine Trident (mu -> 3mu)"),
        (1, "Photon Conversion (gamma -> 2mu)"),
        (2, "Positron Annihilation (e+ e- -> 2mu)")
    ]

    regions = [
        (1, f"Upstream Rock (Z < {target_z_min:.1f} cm)"),
        (2, f"Target Region [{target_z_min:.1f}, {target_z_max:.1f}] cm"),
        (3, f"Muon System (Z >= {target_z_max:.1f} cm)")
    ]

    for p_id, p_name in processes:
        df_p = df_sig.Filter(f"proc_type == {p_id}")
        actions[f"p_{p_id}_cnt"] = df_p.Count()
        actions[f"p_{p_id}_yld"] = df_p.Sum("scaled_weight")

    for r_id, r_name in regions:
        df_r = df_sig.Filter(f"dyn_region_type == {r_id}")
        actions[f"r_{r_id}_cnt"] = df_r.Count()
        actions[f"r_{r_id}_yld"] = df_r.Sum("scaled_weight")

    # 2D Cross matrix (Process x Region)
    for p_id, _ in processes:
        for r_id, _ in regions:
            df_pr = df_sig.Filter(f"proc_type == {p_id} && dyn_region_type == {r_id}")
            actions[f"pr_{p_id}_{r_id}_cnt"] = df_pr.Count()
            actions[f"pr_{p_id}_{r_id}_yld"] = df_pr.Sum("scaled_weight")

    results = {k: v.GetValue() for k, v in actions.items()}
    elapsed = time.time() - t0

    print_yield_summary(
        results=results,
        processes=processes,
        regions=regions,
        lumi_target=args.lumi_target,
        elapsed_time=elapsed,
    )

if __name__ == "__main__":
    main()
