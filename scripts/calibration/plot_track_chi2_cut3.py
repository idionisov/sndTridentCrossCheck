#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import ROOT
from snd import DataManager, load_trident_libraries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SciFi and DS track chi2/ndf, NDF, and corrected chi2/(ndf-5) distributions after Cut 3."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root",
        help="Input MC ROOT file path.",
    )
    parser.add_argument(
        "--data",
        "-d",
        "--input-data",
        type=str,
        default="/eos/user/i/idioniso/1_Data/Tracks/run_006640/muonReco_run6640_f0.0.0.root",
        help="Input collision data ROOT file path.",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        default="cbmsim",
        help="Input MC TTree name.",
    )
    parser.add_argument(
        "--tree-data",
        type=str,
        default="cbmsim",
        help="Input Data TTree name.",
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
        "-n",
        type=int,
        default=-1,
        help="Max events to process.",
    )
    parser.add_argument(
        "--num-threads",
        "-j",
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


def setup_pad(pad: ROOT.TVirtualPad) -> None:
    pad.SetLeftMargin(0.13)
    pad.SetRightMargin(0.05)
    pad.SetTopMargin(0.09)
    pad.SetBottomMargin(0.12)
    pad.SetGridx(True)
    pad.SetGridy(True)


def main() -> None:
    args = parse_args()

    load_trident_libraries(".")
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    input_path = args.input
    local_scratch = "/tmp/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root"
    if os.path.exists(local_scratch) and os.path.getsize(local_scratch) > 1024 * 1024 * 100:
        input_path = local_scratch

    os.makedirs(args.output_dir, exist_ok=True)

    dm_mc = DataManager(input_path, tree_name=args.tree_name, num_threads=args.num_threads)
    df_mc = dm_mc.rdf()
    if args.max_events > 0:
        df_mc = df_mc.Range(args.max_events)

    calib_cfg = ROOT.snd.trident.MuonCalibrationConfig()
    calib_proc = ROOT.snd.trident.MuonCalibrationProcessor(calib_cfg)
    truth_cfg = ROOT.snd.trident.PassingMuonTruthConfig()
    truth_proc = ROOT.snd.trident.PassingMuonTruthProcessor(truth_cfg)

    df_mc = (
        df_mc
        .Filter("Digi_ScifiHits.GetEntries() >= 3")
        .Define("reco", calib_proc, ["Reco_MuonTracks", "Digi_ScifiHits", "Digi_MuFilterHits"])
        .Define("truth", truth_proc, ["MCTrack", "ScifiPoint", "MuFilterPoint"])
        .Define("c1_ip1", "truth.is_in_acceptance")
        .Define("c2_one_scifi_trk", "c1_ip1 && reco.pass_cut_single_scifi_track")
        .Define("c3_ds_track", "c2_one_scifi_trk && reco.pass_cut_ds_track")
        .Define("is_signal", "truth.is_signal")
        .Filter("c3_ds_track")
        .Define("scifi_chi2_ndf", "reco.track_chi2_ndf")
        .Define("ds_chi2_ndf", "reco.ds_track_chi2_ndf")
        .Define("scifi_ndf", "reco.track_ndf")
        .Define("ds_ndf", "reco.ds_track_ndf")
        .Define("scifi_chi2_corr", "reco.track_ndf > 5 ? reco.track_chi2 / (reco.track_ndf - 5) : -1.0")
        .Define("ds_chi2_corr", "reco.ds_track_ndf > 5 ? reco.ds_track_chi2 / (reco.ds_track_ndf - 5) : -1.0")
    )

    h_sf_all = df_mc.Histo1D(
        ("h_sf_chi2_all", "SciFi Track #chi^{2}/ndf;SciFi Track #chi^{2}/ndf;Entries / bin", args.sf_bins, 0.0, args.sf_max),
        "scifi_chi2_ndf"
    )
    h_sf_sig = df_mc.Filter("is_signal == 1").Histo1D(
        ("h_sf_chi2_sig", "SciFi Track #chi^{2}/ndf (Signal);SciFi Track #chi^{2}/ndf;Entries / bin", args.sf_bins, 0.0, args.sf_max),
        "scifi_chi2_ndf"
    )
    h_sf_bkg = df_mc.Filter("is_signal == 0").Histo1D(
        ("h_sf_chi2_bkg", "SciFi Track #chi^{2}/ndf (Background);SciFi Track #chi^{2}/ndf;Entries / bin", args.sf_bins, 0.0, args.sf_max),
        "scifi_chi2_ndf"
    )

    h_ds_all = df_mc.Histo1D(
        ("h_ds_chi2_all", "Downstream Track #chi^{2}/ndf;DS Track #chi^{2}/ndf;Entries / bin", args.ds_bins, 0.0, args.ds_max),
        "ds_chi2_ndf"
    )
    h_ds_sig = df_mc.Filter("is_signal == 1").Histo1D(
        ("h_ds_chi2_sig", "DS Track #chi^{2}/ndf (Signal);DS Track #chi^{2}/ndf;Entries / bin", args.ds_bins, 0.0, args.ds_max),
        "ds_chi2_ndf"
    )
    h_ds_bkg = df_mc.Filter("is_signal == 0").Histo1D(
        ("h_ds_chi2_bkg", "DS Track #chi^{2}/ndf (Background);DS Track #chi^{2}/ndf;Entries / bin", args.ds_bins, 0.0, args.ds_max),
        "ds_chi2_ndf"
    )

    h_sf_ndf_all = df_mc.Histo1D(
        ("h_sf_ndf_all", "SciFi Track NDF;SciFi Track NDF;Entries", 35, -0.5, 34.5),
        "scifi_ndf"
    )
    h_sf_ndf_sig = df_mc.Filter("is_signal == 1").Histo1D(
        ("h_sf_ndf_sig", "SciFi Track NDF (Signal);SciFi Track NDF;Entries", 35, -0.5, 34.5),
        "scifi_ndf"
    )
    h_sf_ndf_bkg = df_mc.Filter("is_signal == 0").Histo1D(
        ("h_sf_ndf_bkg", "SciFi Track NDF (Background);SciFi Track NDF;Entries", 35, -0.5, 34.5),
        "scifi_ndf"
    )

    h_ds_ndf_all = df_mc.Histo1D(
        ("h_ds_ndf_all", "DS Track NDF;DS Track NDF;Entries", 15, -0.5, 14.5),
        "ds_ndf"
    )
    h_ds_ndf_sig = df_mc.Filter("is_signal == 1").Histo1D(
        ("h_ds_ndf_sig", "DS Track NDF (Signal);DS Track NDF;Entries", 15, -0.5, 14.5),
        "ds_ndf"
    )
    h_ds_ndf_bkg = df_mc.Filter("is_signal == 0").Histo1D(
        ("h_ds_ndf_bkg", "DS Track NDF (Background);DS Track NDF;Entries", 15, -0.5, 14.5),
        "ds_ndf"
    )

    h_sf_corr_all = df_mc.Filter("scifi_chi2_corr >= 0").Histo1D(
        ("h_sf_corr_all", "SciFi #chi^{2}/(ndf - 5);SciFi #chi^{2}/(ndf - 5);Entries / bin", args.sf_bins, 0.0, args.sf_max),
        "scifi_chi2_corr"
    )
    h_sf_corr_sig = df_mc.Filter("is_signal == 1 && scifi_chi2_corr >= 0").Histo1D(
        ("h_sf_corr_sig", "SciFi #chi^{2}/(ndf - 5) (Signal);SciFi #chi^{2}/(ndf - 5);Entries / bin", args.sf_bins, 0.0, args.sf_max),
        "scifi_chi2_corr"
    )
    h_sf_corr_bkg = df_mc.Filter("is_signal == 0 && scifi_chi2_corr >= 0").Histo1D(
        ("h_sf_corr_bkg", "SciFi #chi^{2}/(ndf - 5) (Background);SciFi #chi^{2}/(ndf - 5);Entries / bin", args.sf_bins, 0.0, args.sf_max),
        "scifi_chi2_corr"
    )

    h_ds_corr_all = df_mc.Filter("ds_chi2_corr >= 0").Histo1D(
        ("h_ds_corr_all", "DS #chi^{2}/(ndf - 5) (ndf > 5);DS #chi^{2}/(ndf - 5);Entries / bin", args.ds_bins, 0.0, args.ds_max),
        "ds_chi2_corr"
    )
    h_ds_corr_sig = df_mc.Filter("is_signal == 1 && ds_chi2_corr >= 0").Histo1D(
        ("h_ds_corr_sig", "DS #chi^{2}/(ndf - 5) (Signal, ndf > 5);DS #chi^{2}/(ndf - 5);Entries / bin", args.ds_bins, 0.0, args.ds_max),
        "ds_chi2_corr"
    )
    h_ds_corr_bkg = df_mc.Filter("is_signal == 0 && ds_chi2_corr >= 0").Histo1D(
        ("h_ds_corr_bkg", "DS #chi^{2}/(ndf - 5) (Background, ndf > 5);DS #chi^{2}/(ndf - 5);Entries / bin", args.ds_bins, 0.0, args.ds_max),
        "ds_chi2_corr"
    )

    h2_chi2 = df_mc.Histo2D(
        ("h2_chi2_correlation", "Track Fit Quality Correlation;SciFi Track #chi^{2}/ndf;DS Track #chi^{2}/ndf",
         args.sf_bins, 0.0, args.sf_max, args.ds_bins, 0.0, args.ds_max),
        "scifi_chi2_ndf", "ds_chi2_ndf"
    )

    h_dt_sf = None
    h_dt_ds = None
    h_dt_sf_ndf = None
    h_dt_ds_ndf = None
    h_dt_sf_corr = None
    h_dt_ds_corr = None

    if args.data and os.path.exists(args.data):
        dm_data = DataManager(args.data, tree_name=args.tree_data, num_threads=args.num_threads)
        df_data = dm_data.rdf()
        if args.max_events > 0:
            df_data = df_data.Range(args.max_events)
        cols_data = [str(c) for c in df_data.GetColumnNames()]
        header_branch = "EventHeader." if "EventHeader." in cols_data else "EventHeader"
        ip1_checker = ROOT.snd.trident.IP1Filter()

        df_data = (
            df_data
            .Filter("Digi_ScifiHits.GetEntries() >= 3")
            .Define("reco", calib_proc, ["Reco_MuonTracks", "Digi_ScifiHits", "Digi_MuFilterHits"])
            .Define("c1_ip1", ip1_checker, [header_branch])
            .Define("c2_one_scifi_trk", "c1_ip1 && reco.pass_cut_single_scifi_track")
            .Define("c3_ds_track", "c2_one_scifi_trk && reco.pass_cut_ds_track")
            .Filter("c3_ds_track")
            .Define("scifi_chi2_ndf", "reco.track_chi2_ndf")
            .Define("ds_chi2_ndf", "reco.ds_track_chi2_ndf")
            .Define("scifi_ndf", "reco.track_ndf")
            .Define("ds_ndf", "reco.ds_track_ndf")
            .Define("scifi_chi2_corr", "reco.track_ndf > 5 ? reco.track_chi2 / (reco.track_ndf - 5) : -1.0")
            .Define("ds_chi2_corr", "reco.ds_track_ndf > 5 ? reco.ds_track_chi2 / (reco.ds_track_ndf - 5) : -1.0")
        )

        h_dt_sf = df_data.Histo1D(
            ("h_dt_sf_chi2", "Data SciFi #chi^{2}/ndf;SciFi Track #chi^{2}/ndf;Entries / bin", args.sf_bins, 0.0, args.sf_max),
            "scifi_chi2_ndf"
        )
        h_dt_ds = df_data.Histo1D(
            ("h_dt_ds_chi2", "Data DS #chi^{2}/ndf;DS Track #chi^{2}/ndf;Entries / bin", args.ds_bins, 0.0, args.ds_max),
            "ds_chi2_ndf"
        )
        h_dt_sf_ndf = df_data.Histo1D(
            ("h_dt_sf_ndf", "Data SciFi NDF;SciFi Track NDF;Entries", 35, -0.5, 34.5),
            "scifi_ndf"
        )
        h_dt_ds_ndf = df_data.Histo1D(
            ("h_dt_ds_ndf", "Data DS NDF;DS Track NDF;Entries", 15, -0.5, 14.5),
            "ds_ndf"
        )
        h_dt_sf_corr = df_data.Filter("scifi_chi2_corr >= 0").Histo1D(
            ("h_dt_sf_corr", "Data SciFi #chi^{2}/(ndf - 5);SciFi #chi^{2}/(ndf - 5);Entries / bin", args.sf_bins, 0.0, args.sf_max),
            "scifi_chi2_corr"
        )
        h_dt_ds_corr = df_data.Filter("ds_chi2_corr >= 0").Histo1D(
            ("h_dt_ds_corr", "Data DS #chi^{2}/(ndf - 5);DS #chi^{2}/(ndf - 5);Entries / bin", args.ds_bins, 0.0, args.ds_max),
            "ds_chi2_corr"
        )

    # -------------------------------------------------------------
    # Canvas 1: Track chi2 / ndf
    # -------------------------------------------------------------
    c1 = ROOT.TCanvas("c_track_chi2_cut3", "Track Fit Quality after Cut 3", 1600, 700)
    c1.Divide(2, 1)

    setup_pad(c1.cd(1))
    h1_sf = h_sf_all.GetValue()
    h1_sf.SetLineWidth(3)
    h1_sf.SetLineColor(ROOT.kBlack)
    h1_sf.GetXaxis().SetTitleSize(0.045)
    h1_sf.GetYaxis().SetTitleSize(0.045)
    h1_sf.GetXaxis().SetTitleOffset(1.1)
    h1_sf.GetYaxis().SetTitleOffset(1.3)
    h1_sf.SetTitle("SciFi Track #chi^{2}/ndf (After Cut 3)")

    max1_sf = h1_sf.GetMaximum()
    h1_sf.SetMaximum(max1_sf * 1.35 if max1_sf > 0 else 10.0)
    h1_sf.Draw("HIST")

    leg1_sf = ROOT.TLegend(0.48, 0.65, 0.92, 0.88)
    leg1_sf.SetBorderSize(0)
    leg1_sf.SetFillStyle(0)
    leg1_sf.SetTextFont(42)
    leg1_sf.SetTextSize(0.035)
    leg1_sf.AddEntry(h1_sf, f"MC All (N = {int(h1_sf.GetEntries())})", "l")

    h1_s_sf = h_sf_sig.GetValue()
    h1_s_sf.SetLineWidth(2)
    h1_s_sf.SetLineColor(ROOT.kAzure + 1)
    h1_s_sf.SetFillColorAlpha(ROOT.kAzure + 1, 0.35)
    h1_s_sf.Draw("HIST SAME")
    leg1_sf.AddEntry(h1_s_sf, f"MC Signal (N = {int(h1_s_sf.GetEntries())})", "f")

    h1_b_sf = h_sf_bkg.GetValue()
    h1_b_sf.SetLineWidth(2)
    h1_b_sf.SetLineColor(ROOT.kRed + 1)
    h1_b_sf.SetLineStyle(2)
    h1_b_sf.Draw("HIST SAME")
    leg1_sf.AddEntry(h1_b_sf, f"MC Background (N = {int(h1_b_sf.GetEntries())})", "l")

    if h_dt_sf is not None:
        h1_dt_sf = h_dt_sf.GetValue()
        h1_dt_sf.SetMarkerStyle(20)
        h1_dt_sf.SetMarkerSize(0.9)
        h1_dt_sf.SetMarkerColor(ROOT.kBlack)
        h1_dt_sf.SetLineColor(ROOT.kBlack)
        if h1_dt_sf.Integral() > 0 and h1_sf.Integral() > 0:
            h1_dt_sf.Scale(h1_sf.Integral() / h1_dt_sf.Integral())
        h1_dt_sf.Draw("E1 SAME")
        leg1_sf.AddEntry(h1_dt_sf, f"Data (scaled, N = {int(h_dt_sf.GetEntries())})", "lep")

    leg1_sf.Draw()

    setup_pad(c1.cd(2))
    h1_ds = h_ds_all.GetValue()
    h1_ds.SetLineWidth(3)
    h1_ds.SetLineColor(ROOT.kBlack)
    h1_ds.GetXaxis().SetTitleSize(0.045)
    h1_ds.GetYaxis().SetTitleSize(0.045)
    h1_ds.GetXaxis().SetTitleOffset(1.1)
    h1_ds.GetYaxis().SetTitleOffset(1.3)
    h1_ds.SetTitle("Downstream Track #chi^{2}/ndf (After Cut 3)")

    max1_ds = h1_ds.GetMaximum()
    h1_ds.SetMaximum(max1_ds * 1.35 if max1_ds > 0 else 10.0)
    h1_ds.Draw("HIST")

    leg1_ds = ROOT.TLegend(0.48, 0.65, 0.92, 0.88)
    leg1_ds.SetBorderSize(0)
    leg1_ds.SetFillStyle(0)
    leg1_ds.SetTextFont(42)
    leg1_ds.SetTextSize(0.035)
    leg1_ds.AddEntry(h1_ds, f"MC All (N = {int(h1_ds.GetEntries())})", "l")

    h1_s_ds = h_ds_sig.GetValue()
    h1_s_ds.SetLineWidth(2)
    h1_s_ds.SetLineColor(ROOT.kAzure + 1)
    h1_s_ds.SetFillColorAlpha(ROOT.kAzure + 1, 0.35)
    h1_s_ds.Draw("HIST SAME")
    leg1_ds.AddEntry(h1_s_ds, f"MC Signal (N = {int(h1_s_ds.GetEntries())})", "f")

    h1_b_ds = h_ds_bkg.GetValue()
    h1_b_ds.SetLineWidth(2)
    h1_b_ds.SetLineColor(ROOT.kRed + 1)
    h1_b_ds.SetLineStyle(2)
    h1_b_ds.Draw("HIST SAME")
    leg1_ds.AddEntry(h1_b_ds, f"MC Background (N = {int(h1_b_ds.GetEntries())})", "l")

    if h_dt_ds is not None:
        h1_dt_ds = h_dt_ds.GetValue()
        h1_dt_ds.SetMarkerStyle(20)
        h1_dt_ds.SetMarkerSize(0.9)
        h1_dt_ds.SetMarkerColor(ROOT.kBlack)
        h1_dt_ds.SetLineColor(ROOT.kBlack)
        if h1_dt_ds.Integral() > 0 and h1_ds.Integral() > 0:
            h1_dt_ds.Scale(h1_ds.Integral() / h1_dt_ds.Integral())
        h1_dt_ds.Draw("E1 SAME")
        leg1_ds.AddEntry(h1_dt_ds, f"Data (scaled, N = {int(h_dt_ds.GetEntries())})", "lep")

    leg1_ds.Draw()

    # -------------------------------------------------------------
    # Canvas 2: Track NDF
    # -------------------------------------------------------------
    c2 = ROOT.TCanvas("c_track_ndf_cut3", "Track NDF after Cut 3", 1600, 700)
    c2.Divide(2, 1)

    setup_pad(c2.cd(1))
    h2_sf = h_sf_ndf_all.GetValue()
    h2_sf.SetLineWidth(3)
    h2_sf.SetLineColor(ROOT.kBlack)
    h2_sf.GetXaxis().SetTitleSize(0.045)
    h2_sf.GetYaxis().SetTitleSize(0.045)
    h2_sf.GetXaxis().SetTitleOffset(1.1)
    h2_sf.GetYaxis().SetTitleOffset(1.3)
    h2_sf.SetTitle("SciFi Track NDF (After Cut 3)")

    max2_sf = h2_sf.GetMaximum()
    h2_sf.SetMaximum(max2_sf * 1.35 if max2_sf > 0 else 10.0)
    h2_sf.Draw("HIST")

    leg2_sf = ROOT.TLegend(0.48, 0.65, 0.92, 0.88)
    leg2_sf.SetBorderSize(0)
    leg2_sf.SetFillStyle(0)
    leg2_sf.SetTextFont(42)
    leg2_sf.SetTextSize(0.035)
    leg2_sf.AddEntry(h2_sf, f"MC All (N = {int(h2_sf.GetEntries())})", "l")

    h2_s_sf = h_sf_ndf_sig.GetValue()
    h2_s_sf.SetLineWidth(2)
    h2_s_sf.SetLineColor(ROOT.kAzure + 1)
    h2_s_sf.SetFillColorAlpha(ROOT.kAzure + 1, 0.35)
    h2_s_sf.Draw("HIST SAME")
    leg2_sf.AddEntry(h2_s_sf, f"MC Signal (N = {int(h2_s_sf.GetEntries())})", "f")

    h2_b_sf = h_sf_ndf_bkg.GetValue()
    h2_b_sf.SetLineWidth(2)
    h2_b_sf.SetLineColor(ROOT.kRed + 1)
    h2_b_sf.SetLineStyle(2)
    h2_b_sf.Draw("HIST SAME")
    leg2_sf.AddEntry(h2_b_sf, f"MC Background (N = {int(h2_b_sf.GetEntries())})", "l")

    if h_dt_sf_ndf is not None:
        h2_dt_sf = h_dt_sf_ndf.GetValue()
        h2_dt_sf.SetMarkerStyle(20)
        h2_dt_sf.SetMarkerSize(0.9)
        h2_dt_sf.SetMarkerColor(ROOT.kBlack)
        h2_dt_sf.SetLineColor(ROOT.kBlack)
        if h2_dt_sf.Integral() > 0 and h2_sf.Integral() > 0:
            h2_dt_sf.Scale(h2_sf.Integral() / h2_dt_sf.Integral())
        h2_dt_sf.Draw("E1 SAME")
        leg2_sf.AddEntry(h2_dt_sf, f"Data (scaled, N = {int(h_dt_sf_ndf.GetEntries())})", "lep")

    leg2_sf.Draw()

    setup_pad(c2.cd(2))
    h2_ds = h_ds_ndf_all.GetValue()
    h2_ds.SetLineWidth(3)
    h2_ds.SetLineColor(ROOT.kBlack)
    h2_ds.GetXaxis().SetTitleSize(0.045)
    h2_ds.GetYaxis().SetTitleSize(0.045)
    h2_ds.GetXaxis().SetTitleOffset(1.1)
    h2_ds.GetYaxis().SetTitleOffset(1.3)
    h2_ds.SetTitle("Downstream Track NDF (After Cut 3)")

    max2_ds = h2_ds.GetMaximum()
    h2_ds.SetMaximum(max2_ds * 1.35 if max2_ds > 0 else 10.0)
    h2_ds.Draw("HIST")

    leg2_ds = ROOT.TLegend(0.48, 0.65, 0.92, 0.88)
    leg2_ds.SetBorderSize(0)
    leg2_ds.SetFillStyle(0)
    leg2_ds.SetTextFont(42)
    leg2_ds.SetTextSize(0.035)
    leg2_ds.AddEntry(h2_ds, f"MC All (N = {int(h2_ds.GetEntries())})", "l")

    h2_s_ds = h_ds_ndf_sig.GetValue()
    h2_s_ds.SetLineWidth(2)
    h2_s_ds.SetLineColor(ROOT.kAzure + 1)
    h2_s_ds.SetFillColorAlpha(ROOT.kAzure + 1, 0.35)
    h2_s_ds.Draw("HIST SAME")
    leg2_ds.AddEntry(h2_s_ds, f"MC Signal (N = {int(h2_s_ds.GetEntries())})", "f")

    h2_b_ds = h_ds_ndf_bkg.GetValue()
    h2_b_ds.SetLineWidth(2)
    h2_b_ds.SetLineColor(ROOT.kRed + 1)
    h2_b_ds.SetLineStyle(2)
    h2_b_ds.Draw("HIST SAME")
    leg2_ds.AddEntry(h2_b_ds, f"MC Background (N = {int(h2_b_ds.GetEntries())})", "l")

    if h_dt_ds_ndf is not None:
        h2_dt_ds = h_dt_ds_ndf.GetValue()
        h2_dt_ds.SetMarkerStyle(20)
        h2_dt_ds.SetMarkerSize(0.9)
        h2_dt_ds.SetMarkerColor(ROOT.kBlack)
        h2_dt_ds.SetLineColor(ROOT.kBlack)
        if h2_dt_ds.Integral() > 0 and h2_ds.Integral() > 0:
            h2_dt_ds.Scale(h2_ds.Integral() / h2_dt_ds.Integral())
        h2_dt_ds.Draw("E1 SAME")
        leg2_ds.AddEntry(h2_dt_ds, f"Data (scaled, N = {int(h_dt_ds_ndf.GetEntries())})", "lep")

    leg2_ds.Draw()

    # -------------------------------------------------------------
    # Canvas 3: Corrected chi2 / (ndf - 5)
    # -------------------------------------------------------------
    c3 = ROOT.TCanvas("c_track_chi2_corr_cut3", "Track Corrected Fit Quality after Cut 3", 1600, 700)
    c3.Divide(2, 1)

    setup_pad(c3.cd(1))
    h3_sf = h_sf_corr_all.GetValue()
    h3_sf.SetLineWidth(3)
    h3_sf.SetLineColor(ROOT.kBlack)
    h3_sf.GetXaxis().SetTitleSize(0.045)
    h3_sf.GetYaxis().SetTitleSize(0.045)
    h3_sf.GetXaxis().SetTitleOffset(1.1)
    h3_sf.GetYaxis().SetTitleOffset(1.3)
    h3_sf.SetTitle("SciFi Track #chi^{2}/(ndf - 5) (After Cut 3)")

    max3_sf = h3_sf.GetMaximum()
    h3_sf.SetMaximum(max3_sf * 1.35 if max3_sf > 0 else 10.0)
    h3_sf.Draw("HIST")

    leg3_sf = ROOT.TLegend(0.48, 0.65, 0.92, 0.88)
    leg3_sf.SetBorderSize(0)
    leg3_sf.SetFillStyle(0)
    leg3_sf.SetTextFont(42)
    leg3_sf.SetTextSize(0.035)
    leg3_sf.AddEntry(h3_sf, f"MC All (N = {int(h3_sf.GetEntries())})", "l")

    h3_s_sf = h_sf_corr_sig.GetValue()
    h3_s_sf.SetLineWidth(2)
    h3_s_sf.SetLineColor(ROOT.kAzure + 1)
    h3_s_sf.SetFillColorAlpha(ROOT.kAzure + 1, 0.35)
    h3_s_sf.Draw("HIST SAME")
    leg3_sf.AddEntry(h3_s_sf, f"MC Signal (N = {int(h3_s_sf.GetEntries())})", "f")

    h3_b_sf = h_sf_corr_bkg.GetValue()
    h3_b_sf.SetLineWidth(2)
    h3_b_sf.SetLineColor(ROOT.kRed + 1)
    h3_b_sf.SetLineStyle(2)
    h3_b_sf.Draw("HIST SAME")
    leg3_sf.AddEntry(h3_b_sf, f"MC Background (N = {int(h3_b_sf.GetEntries())})", "l")

    if h_dt_sf_corr is not None:
        h3_dt_sf = h_dt_sf_corr.GetValue()
        h3_dt_sf.SetMarkerStyle(20)
        h3_dt_sf.SetMarkerSize(0.9)
        h3_dt_sf.SetMarkerColor(ROOT.kBlack)
        h3_dt_sf.SetLineColor(ROOT.kBlack)
        if h3_dt_sf.Integral() > 0 and h3_sf.Integral() > 0:
            h3_dt_sf.Scale(h3_sf.Integral() / h3_dt_sf.Integral())
        h3_dt_sf.Draw("E1 SAME")
        leg3_sf.AddEntry(h3_dt_sf, f"Data (scaled, N = {int(h_dt_sf_corr.GetEntries())})", "lep")

    leg3_sf.Draw()

    setup_pad(c3.cd(2))
    h3_ds = h_ds_corr_all.GetValue()
    h3_ds.SetLineWidth(3)
    h3_ds.SetLineColor(ROOT.kBlack)
    h3_ds.GetXaxis().SetTitleSize(0.045)
    h3_ds.GetYaxis().SetTitleSize(0.045)
    h3_ds.GetXaxis().SetTitleOffset(1.1)
    h3_ds.GetYaxis().SetTitleOffset(1.3)
    h3_ds.SetTitle("DS Track #chi^{2}/(ndf - 5) [ndf > 5] (After Cut 3)")

    max3_ds = h3_ds.GetMaximum()
    h3_ds.SetMaximum(max3_ds * 1.35 if max3_ds > 0 else 10.0)
    h3_ds.Draw("HIST")

    leg3_ds = ROOT.TLegend(0.48, 0.65, 0.92, 0.88)
    leg3_ds.SetBorderSize(0)
    leg3_ds.SetFillStyle(0)
    leg3_ds.SetTextFont(42)
    leg3_ds.SetTextSize(0.035)
    leg3_ds.AddEntry(h3_ds, f"MC All (N = {int(h3_ds.GetEntries())})", "l")

    h3_s_ds = h_ds_corr_sig.GetValue()
    h3_s_ds.SetLineWidth(2)
    h3_s_ds.SetLineColor(ROOT.kAzure + 1)
    h3_s_ds.SetFillColorAlpha(ROOT.kAzure + 1, 0.35)
    h3_s_ds.Draw("HIST SAME")
    leg3_ds.AddEntry(h3_s_ds, f"MC Signal (N = {int(h3_s_ds.GetEntries())})", "f")

    h3_b_ds = h_ds_corr_bkg.GetValue()
    h3_b_ds.SetLineWidth(2)
    h3_b_ds.SetLineColor(ROOT.kRed + 1)
    h3_b_ds.SetLineStyle(2)
    h3_b_ds.Draw("HIST SAME")
    leg3_ds.AddEntry(h3_b_ds, f"MC Background (N = {int(h3_b_ds.GetEntries())})", "l")

    if h_dt_ds_corr is not None:
        h3_dt_ds = h_dt_ds_corr.GetValue()
        h3_dt_ds.SetMarkerStyle(20)
        h3_dt_ds.SetMarkerSize(0.9)
        h3_dt_ds.SetMarkerColor(ROOT.kBlack)
        h3_dt_ds.SetLineColor(ROOT.kBlack)
        if h3_dt_ds.Integral() > 0 and h3_ds.Integral() > 0:
            h3_dt_ds.Scale(h3_ds.Integral() / h3_dt_ds.Integral())
        h3_dt_ds.Draw("E1 SAME")
        leg3_ds.AddEntry(h3_dt_ds, f"Data (scaled, N = {int(h_dt_ds_corr.GetEntries())})", "lep")

    leg3_ds.Draw()

    pdf_multi = os.path.join(args.output_dir, f"{args.output_name}.pdf")
    c1.SaveAs(f"{pdf_multi}(")
    c2.SaveAs(f"{pdf_multi}")
    c3.SaveAs(f"{pdf_multi})")

    c1.SaveAs(os.path.join(args.output_dir, f"{args.output_name}_chi2_ndf.png"))
    c2.SaveAs(os.path.join(args.output_dir, f"{args.output_name}_ndf.png"))
    c3.SaveAs(os.path.join(args.output_dir, f"{args.output_name}_chi2_corr.png"))

    root_path = os.path.join(args.output_dir, f"{args.output_name}.root")
    f_out = ROOT.TFile.Open(root_path, "RECREATE")

    h1_sf.Write("h_scifi_chi2_all")
    h1_s_sf.Write("h_scifi_chi2_signal")
    h1_b_sf.Write("h_scifi_chi2_background")
    h1_ds.Write("h_ds_chi2_all")
    h1_s_ds.Write("h_ds_chi2_signal")
    h1_b_ds.Write("h_ds_chi2_background")

    h2_sf.Write("h_scifi_ndf_all")
    h2_s_sf.Write("h_scifi_ndf_signal")
    h2_b_sf.Write("h_scifi_ndf_background")
    h2_ds.Write("h_ds_ndf_all")
    h2_s_ds.Write("h_ds_ndf_signal")
    h2_b_ds.Write("h_ds_ndf_background")

    h3_sf.Write("h_scifi_chi2_corr_all")
    h3_s_sf.Write("h_scifi_chi2_corr_signal")
    h3_b_sf.Write("h_scifi_chi2_corr_background")
    h3_ds.Write("h_ds_chi2_corr_all")
    h3_s_ds.Write("h_ds_chi2_corr_signal")
    h3_b_ds.Write("h_ds_chi2_corr_background")

    if h_dt_sf is not None:
        h1_dt_sf.Write("h_data_scifi_chi2")
        h1_dt_ds.Write("h_data_ds_chi2")
        h2_dt_sf.Write("h_data_scifi_ndf")
        h2_dt_ds.Write("h_data_ds_ndf")
        h3_dt_sf.Write("h_data_scifi_chi2_corr")
        h3_dt_ds.Write("h_data_ds_chi2_corr")

    h2_chi2.GetValue().Write("h2_chi2_scifi_vs_ds")

    c1.Write("c_track_chi2_cut3")
    c2.Write("c_track_ndf_cut3")
    c3.Write("c_track_chi2_corr_cut3")
    f_out.Close()

    print(f"Output multi-page PDF: {pdf_multi}")
    print(f"Output ROOT file     : {root_path}")


if __name__ == "__main__":
    main()
