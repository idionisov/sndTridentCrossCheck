#!/usr/bin/env python3

from __future__ import annotations

import argparse
from array import array
import os
import sys
import time
from typing import List

import ROOT
from snd import load_trident_libraries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and export clean passing muon events with full truth values."
    )
    parser.add_argument(
        "--mc-input",
        type=str,
        default="/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root",
        help="Input MC ROOT file path.",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        default="cbmsim",
        help="Input ROOT TTree name (default: cbmsim).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="/eos/user/i/idioniso/sndMuTri/out/passing_muons_filtered.root",
        help="Destination output ROOT file path.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="Limit number of processed events (-1 = process all).",
    )
    parser.add_argument(
        "--shower-threshold",
        type=float,
        default=2.0,
        help="EM shower secondary kinetic energy threshold [GeV] (default: 2.0).",
    )
    parser.add_argument(
        "--hadronic-threshold",
        type=float,
        default=0.5,
        help="Hadronic interaction secondary energy threshold [GeV] (default: 0.5).",
    )
    parser.add_argument(
        "--fiducial-margin",
        type=float,
        default=1.5,
        help="Fiducial margin inside SciFi boundaries [cm] (default: 1.5).",
    )
    parser.add_argument(
        "--chi2-max",
        type=float,
        default=20.0,
        help="Global maximum track chi2/ndf (default: 5.0).",
    )
    parser.add_argument(
        "--chi2-max-scifi",
        type=float,
        default=None,
        help="Maximum SciFi track chi2/ndf (default: inherits --chi2-max).",
    )
    parser.add_argument(
        "--chi2-max-ds",
        type=float,
        default=None,
        help="Maximum Downstream track chi2/ndf (default: inherits --chi2-max).",
    )
    parser.add_argument(
        "--max-slope",
        type=float,
        default=0.05,
        help="Maximum track slope in xz/yz relative to z-axis [rad] (default: 0.05).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25000,
        help="Progress report interval (default: 25,000 events).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("=" * 76)
    print(" SND@LHC Passing Muon Event Extraction & Truth Snapshot")
    print("=" * 76)

    # 1. Load C++ trident shared libraries
    load_trident_libraries(".")
    ROOT.gROOT.SetBatch(True)

    input_path = args.mc_input
    local_scratch = "/tmp/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root"
    if os.path.exists(local_scratch) and os.path.getsize(local_scratch) > 1024 * 1024 * 100:
        print(f"[*] Detected local scratch copy: {local_scratch}")
        input_path = local_scratch

    print(f"[*] Input File   : {input_path}")
    print(f"[*] Output Target: {args.output}")
    print(f"[*] Tree Name    : {args.tree_name}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"[!] Warning: Could not create output directory {out_dir}: {e}")

    # 2. Open input ROOT file and TTree
    f_in = ROOT.TFile.Open(input_path)
    if not f_in or f_in.IsZombie():
        raise RuntimeError(f"Could not open input ROOT file: {input_path}")
    t_in = f_in.Get(args.tree_name)
    if not t_in:
        raise RuntimeError(f"Could not find tree '{args.tree_name}' in {input_path}")

    total_tree_entries = t_in.GetEntries()
    n_to_process = total_tree_entries if args.max_events <= 0 else min(args.max_events, total_tree_entries)
    print(f"[*] Available Events: {total_tree_entries:,} | To Process: {n_to_process:,}")

    # 3. Setup Processors matching passing_muon_selection.py
    truth_cfg = ROOT.snd.trident.PassingMuonTruthConfig()
    truth_cfg.shower_energy_threshold = args.shower_threshold
    truth_cfg.hadronic_energy_threshold = args.hadronic_threshold
    truth_cfg.fiducial_margin = args.fiducial_margin
    truth_proc = ROOT.snd.trident.PassingMuonTruthProcessor(truth_cfg)

    calib_cfg = ROOT.snd.trident.MuonCalibrationConfig()
    calib_cfg.chi2_max = args.chi2_max
    calib_cfg.chi2_max_scifi = args.chi2_max_scifi if args.chi2_max_scifi is not None else args.chi2_max
    calib_cfg.chi2_max_ds = args.chi2_max_ds if args.chi2_max_ds is not None else args.chi2_max
    calib_cfg.max_slope = args.max_slope
    calib_cfg.fiducial_margin = args.fiducial_margin
    calib_proc = ROOT.snd.trident.MuonCalibrationProcessor(calib_cfg)
    print(f"[*] Calibration Cuts: Chi2 Max (SciFi: {calib_cfg.chi2_max_scifi}, DS: {calib_cfg.chi2_max_ds}) | Max Slope: {calib_cfg.max_slope} rad")

    # 4. Prepare Output File & CloneTree for Single-Pass Streaming
    f_out = ROOT.TFile.Open(args.output, "RECREATE")
    if not f_out or f_out.IsZombie():
        raise RuntimeError(f"Could not create output ROOT file: {args.output}")

    t_out = t_in.CloneTree(0)

    # Truth branch storage buffers
    b_truth_cat_id         = array("i", [0])
    b_truth_cat_name       = ROOT.std.string()
    b_truth_is_signal      = array("i", [0])
    b_truth_in_acc         = array("i", [0])
    b_truth_fid_truth      = array("i", [0])
    b_truth_weight         = array("d", [0.0])
    b_truth_mu_id          = array("i", [0])
    b_truth_mu_pdg         = array("i", [0])
    b_truth_mu_p           = array("d", [0.0])
    b_truth_mu_pt          = array("d", [0.0])
    b_truth_mu_pz          = array("d", [0.0])
    b_truth_mu_eta         = array("d", [0.0])
    b_truth_mu_x           = array("d", [0.0])
    b_truth_mu_y           = array("d", [0.0])
    b_truth_mu_z           = array("d", [0.0])
    b_truth_mu_sxz         = array("d", [0.0])
    b_truth_mu_syz         = array("d", [0.0])
    b_truth_n_mu_sf        = array("i", [0])
    b_truth_n_mu_ds        = array("i", [0])
    b_truth_n_sf_pts       = array("i", [0])
    b_truth_n_mf_pts       = array("i", [0])
    b_truth_n_mu_sf_pts    = array("i", [0])
    b_truth_n_nonmu_sf_pts = array("i", [0])
    b_truth_nonmu_frac     = array("d", [0.0])
    b_truth_n_sec          = array("i", [0])
    b_truth_max_sec_e      = array("d", [0.0])
    b_truth_dom_proc       = ROOT.std.string()

    # Reco branch storage buffers
    b_reco_clean           = array("i", [0])
    b_reco_chi2_ndf        = array("f", [0.0])
    b_reco_sf_nhits        = array("i", [0])
    b_reco_ds_nhits        = array("i", [0])
    b_reco_veto_nhits      = array("i", [0])
    b_reco_us_nhits        = array("i", [0])
    b_reco_slope_xz        = array("f", [0.0])
    b_reco_slope_yz        = array("f", [0.0])
    b_reco_start_x         = array("f", [0.0])
    b_reco_start_y         = array("f", [0.0])
    b_reco_start_z         = array("f", [0.0])

    # Register truth branches
    t_out.Branch("truth_category_id",           b_truth_cat_id,         "truth_category_id/I")
    t_out.Branch("truth_category_name",         b_truth_cat_name)
    t_out.Branch("truth_is_signal",             b_truth_is_signal,      "truth_is_signal/I")
    t_out.Branch("truth_is_in_acceptance",      b_truth_in_acc,         "truth_is_in_acceptance/I")
    t_out.Branch("truth_is_fiducial_truth",     b_truth_fid_truth,      "truth_is_fiducial_truth/I")
    t_out.Branch("truth_mc_weight",             b_truth_weight,         "truth_mc_weight/D")
    t_out.Branch("truth_primary_muon_track_id", b_truth_mu_id,          "truth_primary_muon_track_id/I")
    t_out.Branch("truth_primary_muon_pdg",      b_truth_mu_pdg,         "truth_primary_muon_pdg/I")
    t_out.Branch("truth_primary_muon_p",        b_truth_mu_p,           "truth_primary_muon_p/D")
    t_out.Branch("truth_primary_muon_pt",       b_truth_mu_pt,          "truth_primary_muon_pt/D")
    t_out.Branch("truth_primary_muon_pz",       b_truth_mu_pz,          "truth_primary_muon_pz/D")
    t_out.Branch("truth_primary_muon_eta",      b_truth_mu_eta,         "truth_primary_muon_eta/D")
    t_out.Branch("truth_primary_muon_start_x",  b_truth_mu_x,           "truth_primary_muon_start_x/D")
    t_out.Branch("truth_primary_muon_start_y",  b_truth_mu_y,           "truth_primary_muon_start_y/D")
    t_out.Branch("truth_primary_muon_start_z",  b_truth_mu_z,           "truth_primary_muon_start_z/D")
    t_out.Branch("truth_primary_muon_slope_xz", b_truth_mu_sxz,         "truth_primary_muon_slope_xz/D")
    t_out.Branch("truth_primary_muon_slope_yz", b_truth_mu_syz,         "truth_primary_muon_slope_yz/D")
    t_out.Branch("truth_n_muons_in_scifi",      b_truth_n_mu_sf,        "truth_n_muons_in_scifi/I")
    t_out.Branch("truth_n_muons_in_ds",         b_truth_n_mu_ds,        "truth_n_muons_in_ds/I")
    t_out.Branch("truth_n_scifi_points",        b_truth_n_sf_pts,       "truth_n_scifi_points/I")
    t_out.Branch("truth_n_mufi_points",         b_truth_n_mf_pts,       "truth_n_mufi_points/I")
    t_out.Branch("truth_n_muon_scifi_points",   b_truth_n_mu_sf_pts,    "truth_n_muon_scifi_points/I")
    t_out.Branch("truth_n_non_muon_scifi_points", b_truth_n_nonmu_sf_pts, "truth_n_non_muon_scifi_points/I")
    t_out.Branch("truth_non_muon_point_fraction", b_truth_nonmu_frac,   "truth_non_muon_point_fraction/D")
    t_out.Branch("truth_n_secondaries_in_detector", b_truth_n_sec,      "truth_n_secondaries_in_detector/I")
    t_out.Branch("truth_max_secondary_energy",  b_truth_max_sec_e,      "truth_max_secondary_energy/D")
    t_out.Branch("truth_dominant_process",      b_truth_dom_proc)

    # Register reco branches
    t_out.Branch("reco_is_clean",               b_reco_clean,           "reco_is_clean/I")
    t_out.Branch("reco_chi2_ndf",               b_reco_chi2_ndf,        "reco_chi2_ndf/F")
    t_out.Branch("reco_sf_nhits",               b_reco_sf_nhits,        "reco_sf_nhits/I")
    t_out.Branch("reco_ds_nhits",               b_reco_ds_nhits,        "reco_ds_nhits/I")
    t_out.Branch("reco_veto_nhits",             b_reco_veto_nhits,      "reco_veto_nhits/I")
    t_out.Branch("reco_us_nhits",               b_reco_us_nhits,        "reco_us_nhits/I")
    t_out.Branch("reco_slope_xz",               b_reco_slope_xz,        "reco_slope_xz/F")
    t_out.Branch("reco_slope_yz",               b_reco_slope_yz,        "reco_slope_yz/F")
    t_out.Branch("reco_start_x",                b_reco_start_x,         "reco_start_x/F")
    t_out.Branch("reco_start_y",                b_reco_start_y,         "reco_start_y/F")
    t_out.Branch("reco_start_z",                b_reco_start_z,         "reco_start_z/F")

    # Get specific branch pointers for fast staged reading
    b_scifi_hits = t_in.GetBranch("Digi_ScifiHits")
    b_mctrack    = t_in.GetBranch("MCTrack")
    b_scifipoint = t_in.GetBranch("ScifiPoint")
    b_mufipoint  = t_in.GetBranch("MuFilterPoint")
    b_recotracks = t_in.GetBranch("Reco_MuonTracks")
    b_mufihits   = t_in.GetBranch("Digi_MuFilterHits")

    # Cutflow counters
    c_raw = 0
    c_min_scifi = 0
    c_ip1 = 0
    c_single_trk = 0
    c_ds_track = 0
    c_chi2 = 0
    c_slope = 0
    c_fiducial = 0
    c_ds_match = 0
    n_passing = 0

    print("\n[*] Processing events in single-pass streaming mode...")
    sys.stdout.flush()

    t_scan_start = time.time()
    for i in range(n_to_process):
        c_raw += 1

        # Cut 0: SciFi hits >= 3
        b_scifi_hits.GetEntry(i)
        if t_in.Digi_ScifiHits.GetEntries() < 3:
            continue
        c_min_scifi += 1

        # Cut 1: Acceptance (IP1 truth)
        b_mctrack.GetEntry(i)
        b_scifipoint.GetEntry(i)
        b_mufipoint.GetEntry(i)
        truth = truth_proc.process(t_in.MCTrack, t_in.ScifiPoint, t_in.MuFilterPoint)
        if not truth.is_in_acceptance:
            continue
        c_ip1 += 1

        # Reco cuts
        b_recotracks.GetEntry(i)
        b_mufihits.GetEntry(i)
        reco = calib_proc.process(t_in.Reco_MuonTracks, t_in.Digi_ScifiHits, t_in.Digi_MuFilterHits)

        # Cut 2: Single SciFi Track
        if not reco.pass_cut_single_scifi_track:
            continue
        c_single_trk += 1

        # Cut 3: DS Penetration Track
        if not reco.pass_cut_ds_track:
            continue
        c_ds_track += 1

        # Cut 4: Chi2 Quality Cut
        if not reco.pass_cut_chi2:
            continue
        c_chi2 += 1

        # Cut 5: Angular Slope Cut
        if not reco.pass_cut_slope:
            continue
        c_slope += 1

        # Cut 6: Fiducial plane 430
        if not reco.pass_cut_fiducial_430:
            continue
        c_fiducial += 1

        # Cut 7: SciFi - DS Matching
        if not reco.pass_cut_ds_match:
            continue
        c_ds_match += 1

        # -------------------------------------------------------------
        # Event passed all cuts! Read all original branches for storage
        # -------------------------------------------------------------
        t_in.GetEntry(i)

        # Populate truth branch buffers
        b_truth_cat_id[0]         = int(truth.category_id)
        b_truth_cat_name.replace(0, ROOT.std.string.npos, str(truth.category_name))
        b_truth_is_signal[0]      = int(truth.is_signal)
        b_truth_in_acc[0]         = int(truth.is_in_acceptance)
        b_truth_fid_truth[0]      = int(truth.is_fiducial_truth)
        b_truth_weight[0]         = float(truth.mc_weight)
        b_truth_mu_id[0]          = int(truth.primary_muon_track_id)
        b_truth_mu_pdg[0]         = int(truth.primary_muon_pdg)
        b_truth_mu_p[0]           = float(truth.primary_muon_p)
        b_truth_mu_pt[0]          = float(truth.primary_muon_pt)
        b_truth_mu_pz[0]          = float(truth.primary_muon_pz)
        b_truth_mu_eta[0]         = float(truth.primary_muon_eta)
        b_truth_mu_x[0]           = float(truth.primary_muon_start_x)
        b_truth_mu_y[0]           = float(truth.primary_muon_start_y)
        b_truth_mu_z[0]           = float(truth.primary_muon_start_z)
        b_truth_mu_sxz[0]         = float(truth.primary_muon_slope_xz)
        b_truth_mu_syz[0]         = float(truth.primary_muon_slope_yz)
        b_truth_n_mu_sf[0]        = int(truth.n_muons_in_scifi)
        b_truth_n_mu_ds[0]        = int(truth.n_muons_in_ds)
        b_truth_n_sf_pts[0]       = int(truth.n_scifi_points)
        b_truth_n_mf_pts[0]       = int(truth.n_mufi_points)
        b_truth_n_mu_sf_pts[0]    = int(truth.n_muon_scifi_points)
        b_truth_n_nonmu_sf_pts[0] = int(truth.n_non_muon_scifi_points)
        b_truth_nonmu_frac[0]     = float(truth.non_muon_point_fraction)
        b_truth_n_sec[0]          = int(truth.n_secondaries_in_detector)
        b_truth_max_sec_e[0]      = float(truth.max_secondary_energy)
        b_truth_dom_proc.replace(0, ROOT.std.string.npos, str(truth.dominant_process))

        # Populate reco branch buffers
        b_reco_clean[0]           = int(reco.is_clean)
        b_reco_chi2_ndf[0]        = float(reco.track_chi2_ndf)
        b_reco_sf_nhits[0]        = int(reco.scifi_nhits)
        b_reco_ds_nhits[0]        = int(reco.ds_nhits)
        b_reco_veto_nhits[0]      = int(reco.veto_nhits)
        b_reco_us_nhits[0]        = int(reco.us_nhits)
        b_reco_slope_xz[0]        = float(reco.track_slope_xz)
        b_reco_slope_yz[0]        = float(reco.track_slope_yz)
        b_reco_start_x[0]         = float(reco.track_start_x)
        b_reco_start_y[0]         = float(reco.track_start_y)
        b_reco_start_z[0]         = float(reco.track_start_z)

        t_out.Fill()
        n_passing += 1

        if (i + 1) % args.batch_size == 0 or (i + 1) == n_to_process:
            t_out.AutoSave("SaveSelf")
            elapsed = time.time() - t_scan_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    --> Processed {i + 1:>8,} / {n_to_process:,} events ({rate:>7.1f} evt/s) | Clean passing muons: {n_passing:,}", flush=True)

    t_scan_elapsed = time.time() - t_scan_start
    print(f"\n[✓] Processing completed in {t_scan_elapsed:.2f} s. Extracted {n_passing:,} clean passing events.")

    # 5. Flush and close output tree
    print("[*] Writing output file to disk...")
    sys.stdout.flush()
    t_out.Write("", ROOT.TObject.kOverwrite)
    f_out.Close()
    f_in.Close()

    total_elapsed = time.time() - start_time

    # 6. Cutflow Summary Table
    print("\n" + "=" * 76)
    print(" Passing Muon Selection Cutflow Summary")
    print("=" * 76)
    print(f"Total events analyzed       : {c_raw:>8,}")
    print(f"Cut 0: Digi SciFi Hits >= 3 : {c_min_scifi:>8,} ({c_min_scifi / c_raw * 100:>6.2f}%)")
    print(f"Cut 1: In Acceptance (IP1)  : {c_ip1:>8,} ({c_ip1 / c_raw * 100:>6.2f}%)")
    print(f"Cut 2: Single SciFi Track   : {c_single_trk:>8,} ({c_single_trk / c_raw * 100:>6.2f}%)")
    print(f"Cut 3: DS Penetration Track : {c_ds_track:>8,} ({c_ds_track / c_raw * 100:>6.2f}%)")
    print(f"Cut 4: Track Quality Chi2   : {c_chi2:>8,} ({c_chi2 / c_raw * 100:>6.2f}%)")
    print(f"Cut 5: Angular Slope Cut    : {c_slope:>8,} ({c_slope / c_raw * 100:>6.2f}%)")
    print(f"Cut 6: Fiducial Plane 430   : {c_fiducial:>8,} ({c_fiducial / c_raw * 100:>6.2f}%)")
    print(f"Cut 7: SciFi - DS Matching  : {c_ds_match:>8,} ({c_ds_match / c_raw * 100:>6.2f}%)")
    print("=" * 76)

    # 7. Verification of Output File
    if os.path.exists(args.output):
        out_bytes = os.path.getsize(args.output)
        f_check = ROOT.TFile.Open(args.output)
        if f_check and not f_check.IsZombie():
            t_check = f_check.Get(args.tree_name)
            n_entries = t_check.GetEntries() if t_check else 0
            n_branches = len(t_check.GetListOfBranches()) if t_check else 0
            f_check.Close()
            print(f"\n[✓] Successfully created skim file: {args.output}")
            print(f"    Total Events Stored : {n_entries:,}")
            print(f"    Total Branches      : {n_branches} (all raw + 38 truth/reco branches)")
            print(f"    Output File Size    : {out_bytes / (1024 * 1024):.2f} MB")
            print(f"    Average Processing Rate : {n_to_process / t_scan_elapsed:.1f} evt/s")
            print(f"    Total Elapsed Time      : {total_elapsed:.2f} s")
    else:
        print(f"[!] Error: Output file {args.output} was not created.")


if __name__ == "__main__":
    main()
