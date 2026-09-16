#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import ROOT
from snd import DataManager, load_trident_libraries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SciFi and DS track chi2/ndf distributions after Cut 3."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root",
        help="Input ROOT file path.",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        default="cbmsim",
        help="Input TTree name.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="/eos/user/i/idioniso/sndMuTri/out/calibration",
        help="Output directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="track_chi2_distribution_cut3",
        help="Base name for output plots.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="Max events to process.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--sf-bins",
        type=int,
        default=80,
        help="Number of bins for SciFi chi2/ndf.",
    )
    parser.add_argument(
        "--sf-max",
        type=float,
        default=30.0,
        help="Max range for SciFi chi2/ndf axis.",
    )
    parser.add_argument(
        "--ds-bins",
        type=int,
        default=80,
        help="Number of bins for DS chi2/ndf.",
    )
    parser.add_argument(
        "--ds-max",
        type=float,
        default=15.0,
        help="Max range for DS chi2/ndf axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_trident_libraries(".")
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat("emruo")

    input_path = args.input
    local_scratch = "/tmp/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root"
    if os.path.exists(local_scratch) and os.path.getsize(local_scratch) > 1024 * 1024 * 100:
        input_path = local_scratch

    os.makedirs(args.output_dir, exist_ok=True)

    dm = DataManager(input_path, tree_name=args.tree_name, num_threads=args.num_threads)
    df = dm.rdf()

    if args.max_events > 0:
        df = df.Range(args.max_events)

    column_names = [str(c) for c in df.GetColumnNames()]
    has_truth = "MCTrack" in column_names and "ScifiPoint" in column_names

    calib_cfg = ROOT.snd.trident.MuonCalibrationConfig()
    calib_proc = ROOT.snd.trident.MuonCalibrationProcessor(calib_cfg)

    df = (
        df
        .Filter("Digi_ScifiHits.GetEntries() >= 3")
        .Define("reco", calib_proc, ["Reco_MuonTracks", "Digi_ScifiHits", "Digi_MuFilterHits"])
    )

    if has_truth:
        truth_cfg = ROOT.snd.trident.PassingMuonTruthConfig()
        truth_proc = ROOT.snd.trident.PassingMuonTruthProcessor(truth_cfg)
        df = (
            df
            .Define("truth", truth_proc, ["MCTrack", "ScifiPoint", "MuFilterPoint"])
            .Define("c1_ip1", "truth.is_in_acceptance")
            .Define("c2_one_scifi_trk", "c1_ip1 && reco.pass_cut_single_scifi_track")
            .Define("c3_ds_track", "c2_one_scifi_trk && reco.pass_cut_ds_track")
            .Define("is_signal", "truth.is_signal")
        )
    else:
        df = (
            df
            .Define("c1_ip1", "EventHeader.isIP1()")
            .Define("c2_one_scifi_trk", "c1_ip1 && reco.pass_cut_single_scifi_track")
            .Define("c3_ds_track", "c2_one_scifi_trk && reco.pass_cut_ds_track")
            .Define("is_signal", "1")
        )

    df_cut3 = (
        df
        .Filter("c3_ds_track")
        .Define("scifi_chi2_ndf", "reco.track_chi2_ndf")
        .Define("ds_chi2_ndf", "reco.ds_track_chi2_ndf")
    )

    h_sf_all = df_cut3.Histo1D(
        ("h_sf_chi2_all", "SciFi Track #chi^{2}/ndf;SciFi Track #chi^{2}/ndf;Entries / bin", args.sf_bins, 0.0, args.sf_max),
        "scifi_chi2_ndf"
    )
    h_ds_all = df_cut3.Histo1D(
        ("h_ds_chi2_all", "Downstream Track #chi^{2}/ndf;DS Track #chi^{2}/ndf;Entries / bin", args.ds_bins, 0.0, args.ds_max),
        "ds_chi2_ndf"
    )

    if has_truth:
        h_sf_sig = df_cut3.Filter("is_signal == 1").Histo1D(
            ("h_sf_chi2_sig", "SciFi Track #chi^{2}/ndf (Signal);SciFi Track #chi^{2}/ndf;Entries / bin", args.sf_bins, 0.0, args.sf_max),
            "scifi_chi2_ndf"
        )
        h_sf_bkg = df_cut3.Filter("is_signal == 0").Histo1D(
            ("h_sf_chi2_bkg", "SciFi Track #chi^{2}/ndf (Background);SciFi Track #chi^{2}/ndf;Entries / bin", args.sf_bins, 0.0, args.sf_max),
            "scifi_chi2_ndf"
        )
        h_ds_sig = df_cut3.Filter("is_signal == 1").Histo1D(
            ("h_ds_chi2_sig", "DS Track #chi^{2}/ndf (Signal);DS Track #chi^{2}/ndf;Entries / bin", args.ds_bins, 0.0, args.ds_max),
            "ds_chi2_ndf"
        )
        h_ds_bkg = df_cut3.Filter("is_signal == 0").Histo1D(
            ("h_ds_chi2_bkg", "DS Track #chi^{2}/ndf (Background);DS Track #chi^{2}/ndf;Entries / bin", args.ds_bins, 0.0, args.ds_max),
            "ds_chi2_ndf"
        )

    h2_chi2 = df_cut3.Histo2D(
        ("h2_chi2_correlation", "Track Fit Quality Correlation;SciFi Track #chi^{2}/ndf;DS Track #chi^{2}/ndf",
         args.sf_bins, 0.0, args.sf_max, args.ds_bins, 0.0, args.ds_max),
        "scifi_chi2_ndf", "ds_chi2_ndf"
    )

    canvas = ROOT.TCanvas("c_track_chi2_cut3", "Track Fit Quality after Cut 3", 1600, 700)
    canvas.Divide(2, 1)

    pad1 = canvas.cd(1)
    pad1.SetLeftMargin(0.13)
    pad1.SetRightMargin(0.05)
    pad1.SetTopMargin(0.09)
    pad1.SetBottomMargin(0.12)
    pad1.SetGridx(True)
    pad1.SetGridy(True)

    h_sf = h_sf_all.GetValue()
    h_sf.SetLineWidth(3)
    h_sf.SetLineColor(ROOT.kBlack)
    h_sf.GetXaxis().SetTitleSize(0.045)
    h_sf.GetYaxis().SetTitleSize(0.045)
    h_sf.GetXaxis().SetTitleOffset(1.1)
    h_sf.GetYaxis().SetTitleOffset(1.3)
    h_sf.SetTitle("SciFi Track #chi^{2}/ndf (After Cut 3)")

    max_sf = h_sf.GetMaximum()
    h_sf.SetMaximum(max_sf * 1.30 if max_sf > 0 else 10.0)
    h_sf.Draw("HIST")

    leg1 = ROOT.TLegend(0.50, 0.65, 0.92, 0.88)
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetTextFont(42)
    leg1.SetTextSize(0.038)
    leg1.AddEntry(h_sf, f"All Events (N = {int(h_sf.GetEntries())})", "l")

    if has_truth:
        h_s_sf = h_sf_sig.GetValue()
        h_s_sf.SetLineWidth(2)
        h_s_sf.SetLineColor(ROOT.kAzure + 1)
        h_s_sf.SetFillColorAlpha(ROOT.kAzure + 1, 0.35)
        h_s_sf.Draw("HIST SAME")
        leg1.AddEntry(h_s_sf, f"Signal Muons (N = {int(h_s_sf.GetEntries())})", "f")

        h_b_sf = h_sf_bkg.GetValue()
        h_b_sf.SetLineWidth(2)
        h_b_sf.SetLineColor(ROOT.kRed + 1)
        h_b_sf.SetLineStyle(2)
        h_b_sf.Draw("HIST SAME")
        leg1.AddEntry(h_b_sf, f"Background (N = {int(h_b_sf.GetEntries())})", "l")

    leg1.Draw()

    pad2 = canvas.cd(2)
    pad2.SetLeftMargin(0.13)
    pad2.SetRightMargin(0.05)
    pad2.SetTopMargin(0.09)
    pad2.SetBottomMargin(0.12)
    pad2.SetGridx(True)
    pad2.SetGridy(True)

    h_ds = h_ds_all.GetValue()
    h_ds.SetLineWidth(3)
    h_ds.SetLineColor(ROOT.kBlack)
    h_ds.GetXaxis().SetTitleSize(0.045)
    h_ds.GetYaxis().SetTitleSize(0.045)
    h_ds.GetXaxis().SetTitleOffset(1.1)
    h_ds.GetYaxis().SetTitleOffset(1.3)
    h_ds.SetTitle("Downstream Track #chi^{2}/ndf (After Cut 3)")

    max_ds = h_ds.GetMaximum()
    h_ds.SetMaximum(max_ds * 1.30 if max_ds > 0 else 10.0)
    h_ds.Draw("HIST")

    leg2 = ROOT.TLegend(0.50, 0.65, 0.92, 0.88)
    leg2.SetBorderSize(0)
    leg2.SetFillStyle(0)
    leg2.SetTextFont(42)
    leg2.SetTextSize(0.038)
    leg2.AddEntry(h_ds, f"All Events (N = {int(h_ds.GetEntries())})", "l")

    if has_truth:
        h_s_ds = h_ds_sig.GetValue()
        h_s_ds.SetLineWidth(2)
        h_s_ds.SetLineColor(ROOT.kAzure + 1)
        h_s_ds.SetFillColorAlpha(ROOT.kAzure + 1, 0.35)
        h_s_ds.Draw("HIST SAME")
        leg2.AddEntry(h_s_ds, f"Signal Muons (N = {int(h_s_ds.GetEntries())})", "f")

        h_b_ds = h_ds_bkg.GetValue()
        h_b_ds.SetLineWidth(2)
        h_b_ds.SetLineColor(ROOT.kRed + 1)
        h_b_ds.SetLineStyle(2)
        h_b_ds.Draw("HIST SAME")
        leg2.AddEntry(h_b_ds, f"Background (N = {int(h_b_ds.GetEntries())})", "l")

    leg2.Draw()

    pdf_path = os.path.join(args.output_dir, f"{args.output_name}.pdf")
    png_path = os.path.join(args.output_dir, f"{args.output_name}.png")
    root_path = os.path.join(args.output_dir, f"{args.output_name}.root")

    canvas.SaveAs(pdf_path)
    canvas.SaveAs(png_path)

    f_out = ROOT.TFile.Open(root_path, "RECREATE")
    h_sf.Write("h_scifi_chi2_all")
    h_ds.Write("h_ds_chi2_all")
    if has_truth:
        h_sf_sig.GetValue().Write("h_scifi_chi2_signal")
        h_sf_bkg.GetValue().Write("h_scifi_chi2_background")
        h_ds_sig.GetValue().Write("h_ds_chi2_signal")
        h_ds_bkg.GetValue().Write("h_ds_chi2_background")
    h2_chi2.GetValue().Write("h2_chi2_scifi_vs_ds")
    canvas.Write("c_track_chi2_cut3")
    f_out.Close()

    print(f"Output PDF : {pdf_path}")
    print(f"Output PNG : {png_path}")
    print(f"Output ROOT: {root_path}")


if __name__ == "__main__":
    main()
