import ROOT
from snd import DataManager, load_trident_libraries

load_trident_libraries(".")
ROOT.gROOT.SetBatch(True)


f_out = ROOT.TFile("/eos/user/i/idioniso/sndMuTri/out/exploration/scifi_anisotropy.root", "recreate")


mc_path = "/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_Trks.root"
geo_path = "/eos/user/i/idioniso/1_Data/Monte_Carlo/passing_muons/protons2023/geofile_full.Ntuple-TGeant4.root"

dm = DataManager(mc_path, geo_path=geo_path, init_geo=True, num_threads=8)
scifi_det = dm.scifi
df = dm.rdf()

truth_cfg  = ROOT.snd.trident.PassingMuonTruthConfig()
truth_proc = ROOT.snd.trident.PassingMuonTruthProcessor(truth_cfg)
aniso_calc = ROOT.snd.trident.SciFiAnisotropyCalculator(scifi_det)

df = (
    df
    .Filter("Digi_ScifiHits.GetEntries() >= 3")
    .Define("truth", truth_proc, ["MCTrack", "ScifiPoint", "MuFilterPoint"])
    .Define("cat_id", "truth.category_id")
    .Define("cat_name", "truth.category_name")
    .Define("weight", "truth.mc_weight")
    .Define("anisotropy", aniso_calc, ["Digi_ScifiHits"])
    .Filter("anisotropy > 0")
)

categories = [
    (1, "Clean Passing Muon", ROOT.kBlue + 1, 1),
    (2, "Catastrophic EM Shower", ROOT.kViolet + 1, 2),
    (3, "Hadronic Interaction", ROOT.kOrange + 7, 1),
    (4, "Multi-Muon Bundle", ROOT.kTeal + 2, 2),
    (5, "Halo / Shower (No Muon)", ROOT.kGray + 2, 3),
    (6, "Stopping / Scattered Muon", ROOT.kMagenta + 2, 2),
]

hist_models = {}
for cat_id, label, color, style in categories:
    h_name = f"h_sf_aniso_cat{cat_id}"
    title = f"{label};sf spatial anisotropy #lambda_{{1}}/#Sigma#lambda;normalized"
    hist_models[cat_id] = df.Filter(f"cat_id == {cat_id}").Histo1D(
        (h_name, title, 80, 0.40, 1.005),
        "anisotropy", "weight"
    )

canvas = ROOT.TCanvas("c_aniso", "SciFi Spatial Anisotropy", 900, 700)
canvas.SetLogy(True)

legend = ROOT.TLegend(0.15, 0.62, 0.58, 0.88)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
legend.SetTextSize(0.032)

first = True
for cat_id, label, color, style in categories:
    h = hist_models[cat_id].GetValue()
    if h.GetEntries() < 5:
        continue

    h.SetLineColor(color)
    h.SetLineStyle(style)
    h.SetLineWidth(2)

    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

    h.SetMaximum(2.0)
    h.SetMinimum(1e-4)

    draw_opt = "HIST" if first else "HIST SAME"
    h.Draw(draw_opt)
    legend.AddEntry(h, f"{label} (N={int(h.GetEntries())})", "l")
    first = False

legend.Draw()
canvas.Update()

f_out.cd()
canvas.Write()
f_out.Close()
