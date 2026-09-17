import ROOT
from snd import DataManager, load_trident_libraries

load_trident_libraries(".")
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetOptStat(0)

f_out = ROOT.TFile("/eos/user/i/idioniso/sndMuTri/out/exploration/scifi_anisotropy.root", "recreate")

mc_path = "/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root"
geo_path = "/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/geofile_full.Ntuple-TGeant4.root"

dm = DataManager(mc_path, geo_path=geo_path, init_geo=True, num_threads=1)
scifi_det = dm.scifi
mufi_det = dm.mufilter
rdf = dm.rdf()

truth_cfg  = ROOT.snd.trident.PassingMuonTruthConfig()
truth_processor = ROOT.snd.trident.PassingMuonTruthProcessor(truth_cfg)
scifi_anisotropy_calculator = ROOT.snd.trident.SciFiAnisotropyCalculator(scifi_det)
us_anisotropy_calculator = ROOT.snd.trident.MuFilterAnisotropyCalculator(mufi_det, 2)

rdf = (
    rdf
    .Define("truth", truth_processor, ["MCTrack", "ScifiPoint", "MuFilterPoint"])
    .Define("truth_category_id", "truth.category_id")
    .Define("truth_category_name", "truth.category_name")
    .Define("truth_mc_weight", "truth.mc_weight")
    .Define("anisotropy_sf", scifi_anisotropy_calculator, ["Digi_ScifiHits"])
    .Define("anisotropy_us", us_anisotropy_calculator, ["Digi_MuFilterHits"])
)

categories = [
    (1, "Clean Passing Muon", ROOT.kAzure + 2, 1, 3),
    (2, "Catastrophic EM Shower", ROOT.kRed + 1, 2, 2),
    (3, "Hadronic Interaction", ROOT.kOrange + 1, 1, 2),
    (4, "Multi-Muon Bundle", ROOT.kTeal + 2, 4, 2),
    (5, "Halo / Shower (No Muon)", ROOT.kGray + 1, 3, 2),
    (6, "Stopping / Scattered Muon", ROOT.kMagenta + 2, 2, 2),
]


hist_models = {"sf": {}, "us": {}}
for cat_id, _, _, _, _ in categories:
    sub_rdf = rdf.Filter(f"truth_category_id == {cat_id}")

    hist_models["sf"][cat_id] = sub_rdf.Histo1D(
        (f"h_sf_aniso_cat{cat_id}", "", 100, 0.0, 1.0),
        "anisotropy_sf", "truth_mc_weight"
    )
    hist_models["us"][cat_id] = sub_rdf.Histo1D(
        (f"h_us_aniso_cat{cat_id}", "", 50, 0.0, 1.0),
        "anisotropy_us", "truth_mc_weight"
    )

def generate_anisotropy_canvas(subsystem_key, title_label, x_axis_title):
    canvas = ROOT.TCanvas(f"c_aniso_{subsystem_key}", f"{title_label} Spatial Anisotropy", 900, 950)

    pad1 = ROOT.TPad(f"pad1_{subsystem_key}", "Distribution", 0.0, 0.38, 1.0, 1.0)
    pad1.SetBottomMargin(0.02)
    pad1.SetTopMargin(0.08)
    pad1.SetLeftMargin(0.14)
    pad1.SetRightMargin(0.05)
    pad1.SetLogy(True)
    pad1.SetGridx(True)
    pad1.SetGridy(True)
    pad1.Draw()

    pad2 = ROOT.TPad(f"pad2_{subsystem_key}", "Efficiency", 0.0, 0.0, 1.0, 0.38)
    pad2.SetTopMargin(0.03)
    pad2.SetBottomMargin(0.25)
    pad2.SetLeftMargin(0.14)
    pad2.SetRightMargin(0.05)
    pad2.SetGridx(True)
    pad2.SetGridy(True)
    pad2.Draw()

    h_norm = {}
    h_cum = {}

    for cat_id, label, color, style, width in categories:
        h = hist_models[subsystem_key][cat_id].GetValue()
        if h.GetEntries() < 5:
            continue

        h_n = h.Clone(f"{h.GetName()}_norm")
        if h_n.Integral() > 0:
            h_n.Scale(1.0 / h_n.Integral())
        h_n.SetLineColor(color)
        h_n.SetLineStyle(style)
        h_n.SetLineWidth(width)
        h_norm[cat_id] = h_n

        h_c = h.Clone(f"{h.GetName()}_cum")
        total_w = h.Integral(0, h.GetNbinsX() + 1)
        for b in range(1, h.GetNbinsX() + 1):
            eff = h.Integral(b, h.GetNbinsX() + 1) / total_w if total_w > 0 else 0.0
            h_c.SetBinContent(b, eff)
            h_c.SetBinError(b, 0.0)

        h_c.SetLineColor(color)
        h_c.SetLineStyle(style)
        h_c.SetLineWidth(width)
        h_cum[cat_id] = h_c

    pad1.cd()
    legend = ROOT.TLegend(0.17, 0.55, 0.65, 0.89)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.040)

    first = True
    for cat_id, label, _, _, _ in categories:
        if cat_id not in h_norm:
            continue
        h = h_norm[cat_id]
        h.SetMaximum(2.0)
        h.SetMinimum(5e-4)
        h.GetYaxis().SetTitle("Normalized / bin")
        h.GetYaxis().SetTitleSize(0.05)
        h.GetYaxis().SetTitleOffset(1.25)
        h.GetYaxis().SetLabelSize(0.042)
        h.GetXaxis().SetLabelSize(0)

        h.Draw("HIST" if first else "HIST SAME")
        n_raw = int(hist_models[subsystem_key][cat_id].GetEntries())
        legend.AddEntry(h, f"{label} (N = {n_raw})", "l")
        first = False

    legend.Draw()

    pad2.cd()
    first = True
    for cat_id, _, _, _, _ in categories:
        if cat_id not in h_cum:
            continue
        h = h_cum[cat_id]
        h.SetMaximum(1.08)
        h.SetMinimum(0.0)
        h.GetXaxis().SetTitle(x_axis_title)
        h.GetXaxis().SetTitleSize(0.08)
        h.GetXaxis().SetTitleOffset(1.2)
        h.GetXaxis().SetLabelSize(0.07)
        h.GetYaxis().SetTitle("Efficiency (#geq Cut)")
        h.GetYaxis().SetTitleSize(0.08)
        h.GetYaxis().SetTitleOffset(0.78)
        h.GetYaxis().SetLabelSize(0.07)
        h.GetYaxis().SetNdivisions(505)

        h.Draw("HIST" if first else "HIST SAME")
        first = False

    line = ROOT.TLine(0.0, 0.95, 1.0, 0.95)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineStyle(7)
    line.SetLineWidth(1)
    line.Draw()

    canvas.Update()
    return canvas

c_sf = generate_anisotropy_canvas(
    "sf", "SciFi", "SciFi Spatial Anisotropy Cut (#lambda_{1} / #Sigma#lambda > Cut)"
)
c_us = generate_anisotropy_canvas(
    "us", "Upstream", "US MuFilter Spatial Anisotropy Cut (#lambda_{1} / #Sigma#lambda > Cut)"
)

f_out.cd()
c_sf.Write()
c_us.Write()
f_out.Close()
