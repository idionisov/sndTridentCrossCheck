#!/usr/bin/env python3
"""
calculate_trident_yields.py
---------------------------
High-throughput calculation script for estimating physical trident event yields
(MuonToMuonPair, GammaToMuPair, Positron Annihilation) for a specified target
integrated luminosity (e.g. 1.0 fb^-1 or 38.0 fb^-1).

Features:
- Fast multi-threaded processing using ROOT RDataFrame.
- Reads filtered trimuon files or raw MC simulation files.
- Computes comprehensive yields:
    * Total yields and raw event counts
    * Process breakdown (Genuine, Gamma Conversion, Positron Annihilation, Secondary)
    * Spatial region breakdown (Upstream Rock, Target Region, Muon System)
    * 2D Process x Region cross-tabulation matrix
- Clean, readable tabular terminal outputs.
- Does NOT create or write any output ROOT files to disk.
"""

import os
import sys
import glob
import time
import argparse
import ROOT

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
        "-b", "--boost", "--boost-factor",
        dest="boost_factor",
        type=float,
        default=100.0,
        help="Geant4 cross section boost factor if analyzing raw unboosted files (default: 100.0)"
    )
    parser.add_argument(
        "-j", "--threads",
        dest="num_threads",
        type=int,
        default=8,
        help="Number of worker threads for RDataFrame (default: %(default)s)"
    )

    args = parser.parse_args()

    # Unpack target Z boundaries
    target_z_min, target_z_max = args.target_z_range

    if args.num_threads > 0:
        ROOT.EnableImplicitMT(args.num_threads)

    matched_files = sorted(glob.glob(args.input_pattern))
    if not matched_files:
        if os.path.exists(args.input_pattern):
            matched_files = [args.input_pattern]
        else:
            print(f"Error: No files found matching pattern: {args.input_pattern}")
            sys.exit(1)

    weight_scale = args.lumi_target / args.lumi_mc if args.lumi_mc > 0 else 1.0

    print("=" * 70)
    print("SND@LHC Trident Physical Yield Calculator")
    print("=" * 70)
    print(f"Input Pattern     : {args.input_pattern}")
    print(f"Files Found       : {len(matched_files):,}")
    print(f"Target Z Region   : [{target_z_min:.1f}, {target_z_max:.1f}] cm")
    print(f"Dataset MC Lumi   : {args.lumi_mc:.4f} fb^-1")
    print(f"Target Lumi       : {args.lumi_target:.2f} fb^-1")
    print(f"Lumi Multiplier   : {weight_scale:.4f}x")
    print(f"Worker Threads    : {args.num_threads}")
    print("=" * 70)

    # Build TChain
    chain = ROOT.TChain("cbmsim")
    for fpath in matched_files:
        chain.Add(fpath)

    # Inspect tree branches
    t_first = ROOT.TFile.Open(matched_files[0], "READ")
    tree = t_first.Get("cbmsim")
    branch_names = [b.GetName() for b in tree.GetListOfBranches()]
    t_first.Close()

    has_precomputed = "mc_weight" in branch_names and "proc_type" in branch_names

    t0 = time.time()
    df = ROOT.RDataFrame(chain)

    if has_precomputed:
        # Check if 'is_signal' column exists; if so, filter by signal
        if "is_signal" in branch_names:
            df_sig = df.Filter("is_signal == true")
        else:
            df_sig = df

        # Define scaled event weight
        df_sig = df_sig.Define("scaled_weight", f"mc_weight * {weight_scale}")

        # Dynamically define region_type based on target_z_min and target_z_max if vtx_z exists
        if "vtx_z" in branch_names:
            df_sig = df_sig.Define("dyn_region_type", f"vtx_z < {target_z_min} ? 1 : (vtx_z < {target_z_max} ? 2 : 3)")
        else:
            df_sig = df_sig.Define("dyn_region_type", "region_type")
    else:
        # On raw MC files, declare JIT evaluator
        sndsw_path = os.environ.get("SNDSW_ROOT", "")
        if sndsw_path:
            ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndSciFiTools.h"')
            ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndTchainGetter.h"')
            ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndGeometryGetter.h"')

        cxx_code = f"""
        #ifndef YIELD_TRIDENT_CALC_H
        #define YIELD_TRIDENT_CALC_H
        #include "ShipMCTrack.h"
        #include <TClonesArray.h>
        #include <cmath>

        namespace YieldCalc {{
            double g_target_z_min = {target_z_min};
            double g_target_z_max = {target_z_max};
            double g_boost = {args.boost_factor};
            double g_scale = {weight_scale};

            struct RawTrident {{
                bool is_signal;
                int proc_type;
                int region_type;
                double scaled_w;
            }};

            RawTrident evaluate(const TClonesArray& mc_tracks) {{
                int n_tracks = mc_tracks.GetEntriesFast();
                for (int i = 0; i < n_tracks; ++i) {{
                    auto* tr1 = static_cast<ShipMCTrack*>(mc_tracks.At(i));
                    if (!tr1) continue;
                    int p1 = tr1->GetProcID();
                    if (p1 != 5 && p1 != 10 && p1 != 11 && p1 != 12) continue;
                    int pdg1 = tr1->GetPdgCode();
                    if (std::abs(pdg1) != 13) continue;

                    for (int j = i + 1; j < n_tracks; ++j) {{
                        auto* tr2 = static_cast<ShipMCTrack*>(mc_tracks.At(j));
                        if (!tr2) continue;
                        if (tr2->GetProcID() == p1 && tr2->GetMotherId() == tr1->GetMotherId()) {{
                            int pdg2 = tr2->GetPdgCode();
                            if ((pdg1 == 13 && pdg2 == -13) || (pdg1 == -13 && pdg2 == 13)) {{
                                int m_id = tr1->GetMotherId();
                                int proc_type = -1;
                                if (m_id == -1) proc_type = 0;
                                else if (m_id >= 0 && m_id < n_tracks) {{
                                    auto* mother = static_cast<ShipMCTrack*>(mc_tracks.At(m_id));
                                    if (mother) {{
                                        int m_pdg = mother->GetPdgCode();
                                        if (std::abs(m_pdg) == 13) proc_type = 0;
                                        else if (m_pdg == 22) proc_type = 1;
                                        else if (std::abs(m_pdg) == 11) proc_type = 2;
                                    }}
                                }}
                                double z_v = tr1->GetStartZ();
                                int r_type = 0;
                                if (z_v < g_target_z_min) r_type = 1;
                                else if (z_v < g_target_z_max) r_type = 2;
                                else r_type = 3;

                                double raw_w = tr1->GetWeight();
                                if (proc_type == 0 && g_boost > 0.0) raw_w *= g_boost;
                                double scaled_w = raw_w * g_scale;
                                return {{true, proc_type, r_type, scaled_w}};
                            }}
                        }}
                    }}
                }}
                return {{false, -1, 0, 0.0}};
            }}

            thread_local const TClonesArray* tls_t = nullptr;
            thread_local int tls_s = -1;
            thread_local RawTrident tls_c;

            inline RawTrident get_raw(const TClonesArray& t) {{
                if (&t == tls_t && t.GetEntriesFast() == tls_s) return tls_c;
                tls_t = &t;
                tls_s = t.GetEntriesFast();
                tls_c = evaluate(t);
                return tls_c;
            }}

            bool is_sig(const TClonesArray& t) {{ return get_raw(t).is_signal; }}
            int proc(const TClonesArray& t) {{ return get_raw(t).proc_type; }}
            int region(const TClonesArray& t) {{ return get_raw(t).region_type; }}
            double weight(const TClonesArray& t) {{ return get_raw(t).scaled_w; }}
        }}
        #endif
        """
        ROOT.gInterpreter.Declare(cxx_code)
        df_sig = (
            df.Filter("YieldCalc::is_sig(MCTrack)")
              .Define("proc_type", "YieldCalc::proc(MCTrack)")
              .Define("dyn_region_type", "YieldCalc::region(MCTrack)")
              .Define("scaled_weight", "YieldCalc::weight(MCTrack)")
        )

    # Book all actions simultaneously for a single event loop
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

    # Evaluate results
    tot_cnt = actions["total_count"].GetValue()
    tot_yld = actions["total_yield"].GetValue() if tot_cnt > 0 else 0.0
    elapsed = time.time() - t0

    print(f"\nCalculation completed in {elapsed:.2f} seconds.")
    print("\n" + "=" * 70)
    print(f"PHYSICAL YIELD SUMMARY (Target Lumi: {args.lumi_target:.2f} fb^-1)")
    print("=" * 70)
    print(f"Total Raw Signal Events Selected : {tot_cnt:,}")
    print(f"Total Physical Expected Yield   : {tot_yld:,.2f} events")
    print("-" * 70)

    # Process breakdown table
    print("\n[1] BREAKDOWN BY PROCESS:")
    print(f"{'Process':<36} | {'Raw MC Events':>13} | {'Raw %':>7} | {'Yield (' + str(args.lumi_target) + ' fb^-1)':>18} | {'Yield %':>7}")
    print("-" * 92)
    for p_id, p_name in processes:
        pcnt = actions[f"p_{p_id}_cnt"].GetValue()
        pyld = actions[f"p_{p_id}_yld"].GetValue() if pcnt > 0 else 0.0
        p_pct_raw = (pcnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
        p_pct_yld = (pyld / tot_yld * 100.0) if tot_yld > 0 else 0.0
        print(f"{p_name:<36} | {pcnt:>13,} | {p_pct_raw:>6.2f}% | {pyld:>18.2f} | {p_pct_yld:>6.2f}%")
    print("-" * 92)
    print(f"{'Total All Processes':<36} | {tot_cnt:>13,} | 100.00% | {tot_yld:>18.2f} | 100.00%")

    # Spatial region breakdown table
    print("\n[2] BREAKDOWN BY SPATIAL REGION:")
    print(f"{'Spatial Region':<36} | {'Raw MC Events':>13} | {'Raw %':>7} | {'Yield (' + str(args.lumi_target) + ' fb^-1)':>18} | {'Yield %':>7}")
    print("-" * 92)
    for r_id, r_name in regions:
        rcnt = actions[f"r_{r_id}_cnt"].GetValue()
        ryld = actions[f"r_{r_id}_yld"].GetValue() if rcnt > 0 else 0.0
        r_pct_raw = (rcnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
        r_pct_yld = (ryld / tot_yld * 100.0) if tot_yld > 0 else 0.0
        print(f"{r_name:<36} | {rcnt:>13,} | {r_pct_raw:>6.2f}% | {ryld:>18.2f} | {r_pct_yld:>6.2f}%")
    print("-" * 92)
    print(f"{'Total All Regions':<36} | {tot_cnt:>13,} | 100.00% | {tot_yld:>18.2f} | 100.00%")

    # 2D Cross Table (Process x Region)
    print("\n[3] CROSS MATRIX: PROCESS x REGION YIELDS:")
    col_w = 20
    header_str = f"{'Process':<28} | " + " | ".join([f"{r_name[:18]:^{col_w}}" for _, r_name in regions]) + f" | {'Total':^{col_w}}"
    print(header_str)
    print("-" * len(header_str))

    for p_id, p_name in processes:
        row_str = f"{p_name[:28]:<28} | "
        p_row_tot = 0.0
        for r_id, _ in regions:
            pr_cnt = actions[f"pr_{p_id}_{r_id}_cnt"].GetValue()
            pr_yld = actions[f"pr_{p_id}_{r_id}_yld"].GetValue() if pr_cnt > 0 else 0.0
            p_row_tot += pr_yld
            row_str += f"{pr_yld:^{col_w}.2f} | "
        row_str += f"{p_row_tot:^{col_w}.2f}"
        print(row_str)

    print("-" * len(header_str))
    tot_row_str = f"{'Total':<28} | "
    for r_id, _ in regions:
        r_col_tot = actions[f"r_{r_id}_yld"].GetValue()
        tot_row_str += f"{r_col_tot:^{col_w}.2f} | "
    tot_row_str += f"{tot_yld:^{col_w}.2f}"
    print(tot_row_str)
    print("=" * len(header_str) + "\n")

if __name__ == "__main__":
    main()
