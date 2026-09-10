import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MIP discriminators
    #### Exploration of values that could serve as MIP discriminator
    """)
    return


@app.cell
def _():
    import ROOT, os, json, copy, time
    from pathlib import Path
    import uproot
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.colors import Normalize

    from ddfUtils.mpl_styling import add_label
    import ddfUtils.root as ddf_root

    plt.style.use("root")

    ROOT.gROOT.SetBatch(True)
    return Path, ROOT, add_label, ddf_root, json, np, pd, plt, ticker, uproot


@app.cell
def _(Path):
    DATA_DIR = Path("/eos/user/i/idioniso/sndMuTri/out")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT = DATA_DIR / "pGun_response.root"

    PGUN_DIR = Path("/eos/user/i/idioniso/1_Data/Monte_Carlo/pGun/muons")
    MC_PATH = Path(
        "/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet"
    )

    ENERGY_DIRS = [
        "10.GeV",
        "12.5.GeV",
        "15.GeV",
        "17.5.GeV",
        "20.GeV",
        "25.GeV",
        "30.GeV",
        "35.GeV",
        "55.GeV",
        "100.GeV",
        "200.GeV",
        "300.GeV",
        "450.GeV",
        "600.GeV",
        "800.GeV",
        "1010.GeV",
    ]
    return ENERGY_DIRS, MC_PATH, OUTPUT_ROOT, PGUN_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data extraction
    """)
    return


@app.cell
def _(ENERGY_DIRS, PGUN_DIR, ROOT, np):
    raw_events_sf = {}
    raw_events_ds = {}

    for edir in ENERGY_DIRS:
      dir_path = PGUN_DIR / edir
      digi_file = dir_path / "sndLHC.PG_13-TGeant4_digCPP.root"
      print(digi_file)
      if not digi_file.exists():
        continue

      energy_val = float(edir.replace(".GeV", "").replace("GeV", ""))
      f = ROOT.TFile.Open(str(digi_file))
      if not f or f.IsZombie():
        print(f"Skipping corrupt file: {digi_file}")
        continue

      tree = f.Get("cbmsim")
      if not tree:
        f.Close()
        continue

      n_entries = tree.GetEntries()
      hits_sf_list, qdc_sf_list = [], []
      hits_ds_list, qdc_ds_list = [], []

      for i in range(n_entries):
        tree.GetEntry(i)

        # 1. SciFi Response (10 planes)
        valid_sf_signals = [
            h.GetSignal() for h in tree.Digi_ScifiHits if h.isValid()
        ]
        n_sf = len(valid_sf_signals)
        hits_sf_list.append(n_sf / 10.0)
        qdc_sf_list.append(sum(valid_sf_signals) / 10.0)

        ds_hits = [
            h for h in tree.Digi_MuFilterHits if h.GetSystem() == 3 and h.isValid()
        ]
        n_ds = len(ds_hits)
        hits_ds_list.append(n_ds / 4.0)
        tot_ds_qdc = sum(v for h in ds_hits for k, v in h.GetAllSignals())
        qdc_ds_list.append(tot_ds_qdc / 4.0)

      f.Close()

      if len(hits_sf_list) > 0:
        raw_events_sf[energy_val] = {
            "hits_plane": np.array(hits_sf_list),
            "qdc_plane": np.array(qdc_sf_list),
        }
      if len(hits_ds_list) > 0:
        raw_events_ds[energy_val] = {
            "hits_plane": np.array(hits_ds_list),
            "qdc_plane": np.array(qdc_ds_list),
        }
    return raw_events_ds, raw_events_sf


@app.cell
def _(np, raw_events_ds, raw_events_sf):
    sf_mip_hits = np.concatenate(
        [data["hits_plane"] for E, data in raw_events_sf.items() if E <= 40]
    )
    sf_mip_qdc = np.concatenate(
        [data["qdc_plane"] for E, data in raw_events_sf.items() if E <= 40]
    )

    ds_mip_hits = np.concatenate(
        [data["hits_plane"] for E, data in raw_events_ds.items() if E <= 40]
    )
    ds_mip_qdc = np.concatenate(
        [data["qdc_plane"] for E, data in raw_events_ds.items() if E <= 40]
    )

    w_bar_sf = float(np.mean(sf_mip_hits))
    qdc_bar_sf = float(np.mean(sf_mip_qdc))

    w_bar_ds = float(np.mean(ds_mip_hits))
    qdc_bar_ds = float(np.mean(ds_mip_qdc))

    print(
        f"w_bar_SF   = {w_bar_sf:.3f} chn/plane |"
        f"QDC_bar_SF = {qdc_bar_sf:.3f}"
    )
    print(
        f"w_bar_DS   = {w_bar_ds:.3f} chn/plane |"
        f"QDC_bar_DS = {qdc_bar_ds:.3f}"
    )
    return qdc_bar_ds, qdc_bar_sf, w_bar_ds, w_bar_sf


@app.cell
def _(np, pd):
    def get_derived_vars(raw_data, w_bar, qdc_bar):
      results = []
      for energy_val in sorted(raw_data.keys()):
        h_arr = raw_data[energy_val]["hits_plane"]
        q_arr = raw_data[energy_val]["qdc_plane"]

        emip_arr = 0.5 * ((h_arr / w_bar) + (q_arr / qdc_bar))

        valid_mask = (h_arr > 0) & (q_arr >= 0)
        if np.any(valid_mask):
            rhomip_arr = (q_arr[valid_mask] / qdc_bar) / (
                h_arr[valid_mask] / w_bar
            )
        else:
            rhomip_arr = np.array([0.0])


        n = len(h_arr)
        n_rho = len(rhomip_arr)

        results.append({
            "energy_gev": energy_val,
        
            "mean_hits": float(np.mean(h_arr)),
            "sem_hits": float(np.std(h_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "median_hits": float(np.median(h_arr)),
            "q16_hits": float(np.percentile(h_arr, 16)),
            "q84_hits": float(np.percentile(h_arr, 84)),

            "mean_qdc": float(np.mean(q_arr)),
            "sem_qdc": float(np.std(q_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "median_qdc": float(np.median(q_arr)),
            "q16_qdc": float(np.percentile(q_arr, 16)),
            "q84_qdc": float(np.percentile(q_arr, 84)),

            "mean_emip": float(np.mean(emip_arr)),
            "sem_emip": float(np.std(emip_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "median_emip": float(np.median(emip_arr)),
            "q16_emip": float(np.percentile(emip_arr, 16)),
            "q84_emip": float(np.percentile(emip_arr, 84)),

            "mean_rhomip": float(np.nanmean(rhomip_arr)),
            "sem_rhomip": float(np.nanstd(rhomip_arr, ddof=1) / np.sqrt(n_rho)) if n_rho > 1 else 0.0,
            "median_rhomip": float(np.nanmedian(rhomip_arr)),
            "q16_rhomip": float(np.nanpercentile(rhomip_arr, 16)),
            "q84_rhomip": float(np.nanpercentile(rhomip_arr, 84)),
            "n_events": n,
        })
      return pd.DataFrame(results)


    # df_sf = get_derived_vars(raw_events_sf, w_bar_sf, qdc_bar_sf)
    # df_ds = get_derived_vars(raw_events_ds, w_bar_ds, qdc_bar_ds)
    return


app._unparsable_cell(
    r"""
    with uproot.open(ROOT_FILE) as f_in:
          # Read MIP baselines stored in metadata
          meta = json.loads(str(f_in[\"metadata\"]))
          w_bar_sf = meta[\"w_bar_SF\"]
          qdc_bar_sf = meta[\"qdc_bar_SF\"]
          w_bar_ds = meta[\"w_bar_DS\"]
          qdc_bar_ds = meta[\"qdc_bar_DS\"]
    
          # Read raw TNtuple into a pandas DataFrame (instant load)
          df_raw = f_in[\"pgun_raw/ntuple_pgun_raw\"].arrays(library=\"pd\")
    
        print(f\"[*] Loaded {len(df_raw):,} raw events from '{ROOT_FILE}'\")
        print(
            f\"    Baselines: w_bar_SF = {w_bar_sf:.3f}, QDC_bar_SF = {qdc_bar_sf:.3f} |\"
            f\" w_bar_DS = {w_bar_ds:.3f}, QDC_bar_DS = {qdc_bar_ds:.3f}\"
        )
    
        # ==============================================================================
        # 2. RECONSTRUCT RAW EVENT DICTIONARIES
        # ==============================================================================
        raw_events_sf = {}
        raw_events_ds = {}
    
        for energy_val, grp in df_raw.groupby(\"energy_gev\"):
          raw_events_sf[energy_val] = {
              \"hits_plane\": grp[\"hits_plane_sf\"].to_numpy(),
              \"qdc_plane\": grp[\"qdc_plane_sf\"].to_numpy(),
          }
          raw_events_ds[energy_val] = {
              \"hits_plane\": grp[\"hits_plane_ds\"].to_numpy(),
              \"qdc_plane\": grp[\"qdc_plane_ds\"].to_numpy(),
          }
    
    
        # ==============================================================================
        # 3. APPLY get_derived_vars
        # ==============================================================================
        def get_derived_vars(raw_data, w_bar, qdc_bar):
          results = []
          for energy_val in sorted(raw_data.keys()):
            h_arr = raw_data[energy_val][\"hits_plane\"]
            q_arr = raw_data[energy_val][\"qdc_plane\"]
    
            # Derived variables: e_mip and rho_mip
            emip_arr = 0.5 * ((h_arr / w_bar) + (q_arr / qdc_bar))
    
            valid_mask = (h_arr > 0) & (q_arr >= 0)
            if np.any(valid_mask):
              rhomip_arr = (q_arr[valid_mask] / qdc_bar) / (h_arr[valid_mask] / w_bar)
            else:
              rhomip_arr = np.array([0.0])
    
            n = len(h_arr)
            n_rho = len(rhomip_arr)
    
            results.append({
                \"energy_gev\": energy_val,
                # Hits per plane
                \"mean_hits\": float(np.mean(h_arr)),
                \"sem_hits\": (
                    float(np.std(h_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                ),
                \"median_hits\": float(np.median(h_arr)),
                \"q16_hits\": float(np.percentile(h_arr, 16)),
                \"q84_hits\": float(np.percentile(h_arr, 84)),
                # QDC per plane
                \"mean_qdc\": float(np.mean(q_arr)),
                \"sem_qdc\": float(np.std(q_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                \"median_qdc\": float(np.median(q_arr)),
                \"q16_qdc\": float(np.percentile(q_arr, 16)),
                \"q84_qdc\": float(np.percentile(q_arr, 84)),
                # MIP-Equivalence e_mip
                \"mean_emip\": float(np.mean(emip_arr)),
                \"sem_emip\": (
                    float(np.std(emip_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                ),
                \"median_emip\": float(np.median(emip_arr)),
                \"q16_emip\": float(np.percentile(emip_arr, 16)),
                \"q84_emip\": float(np.percentile(emip_arr, 84)),
                # MIP Charge Density rho_mip
                \"mean_rhomip\": float(np.nanmean(rhomip_arr)),
                \"sem_rhomip\": (
                    float(np.nanstd(rhomip_arr, ddof=1) / np.sqrt(n_rho))
                    if n_rho > 1
                    else 0.0
                ),
                \"median_rhomip\": float(np.nanmedian(rhomip_arr)),
                \"q16_rhomip\": float(np.nanpercentile(rhomip_arr, 16)),
                \"q84_rhomip\": float(np.nanpercentile(rhomip_arr, 84)),
                \"n_events\": n,
            })
          return pd.DataFrame(results)


        # Generate the processed SciFi and Downstream DataFrames
        df_sf = get_derived_vars(raw_events_sf, w_bar_sf, qdc_bar_sf)
        df_ds = get_derived_vars(raw_events_ds, w_bar_ds, qdc_bar_ds)
    """,
    name="_"
)


@app.cell
def _(MC_PATH, np, pd):
    mc_df = pd.read_parquet(MC_PATH)
    sig_mc = mc_df[(mc_df["is_signal"] == 1) & (mc_df["region_type"] == 1)].copy()


    def apply_optimal_topology(df: pd.DataFrame) -> pd.DataFrame:
      """Applies [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc) selection."""

      def check_nc(prefix, proj, z_min, z_max):
        n = df[f"n_lines_{prefix}_{proj}"].to_numpy()
        m1, c1 = (
            np.nan_to_num(df[f"{proj}_{prefix}_m1"].to_numpy()),
            np.nan_to_num(df[f"{proj}_{prefix}_c1"].to_numpy()),
        )
        m2, c2 = (
            np.nan_to_num(df[f"{proj}_{prefix}_m2"].to_numpy()),
            np.nan_to_num(df[f"{proj}_{prefix}_c2"].to_numpy()),
        )
        m3, c3 = (
            np.nan_to_num(df[f"{proj}_{prefix}_m3"].to_numpy()),
            np.nan_to_num(df[f"{proj}_{prefix}_c3"].to_numpy()),
        )

        x1_min, x1_max = m1 * z_min + c1, m1 * z_max + c1
        x2_min, x2_max = m2 * z_min + c2, m2 * z_max + c2
        x3_min, x3_max = m3 * z_min + c3, m3 * z_max + c3

        no_cross_12 = ((x1_min - x2_min) * (x1_max - x2_max)) > 0
        no_cross_13 = ((x1_min - x3_min) * (x1_max - x3_max)) > 0
        no_cross_23 = ((x2_min - x3_min) * (x2_max - x3_max)) > 0

        pass_3 = (n >= 3) & no_cross_12 & no_cross_13 & no_cross_23
        pass_2 = (n == 2) & no_cross_12
        return pass_3 | pass_2

      nx, ny = df["n_lines_sf_xz"].to_numpy(), df["n_lines_sf_yz"].to_numpy()
      nx_ds, ny_ds = df["n_lines_ds_xz"].to_numpy(), df["n_lines_ds_yz"].to_numpy()

      sf_3_1 = ((nx >= 3) & (ny >= 1)) | ((ny >= 3) & (nx >= 1))
      ds_3_1 = ((nx_ds >= 3) & (ny_ds >= 1)) | ((ny_ds >= 3) & (nx_ds >= 1))
      mult_mask = sf_3_1 | ds_3_1

      coll_sf = check_nc("sf", "xz", 260.0, 355.0) | check_nc(
          "sf", "yz", 260.0, 355.0
      )
      coll_ds = check_nc("ds", "xz", 460.0, 590.0) | check_nc(
          "ds", "yz", 460.0, 590.0
      )
      return df[mult_mask & coll_sf & coll_ds].copy()


    sel_mc = apply_optimal_topology(sig_mc)

    m_mu = 0.10566
    e1 = np.sqrt(sel_mc["p_mu_in"] ** 2 + m_mu**2).to_numpy()
    e2 = np.sqrt(sel_mc["p_mu_minus"] ** 2 + m_mu**2).to_numpy()
    e3 = np.sqrt(sel_mc["p_mu_plus"] ** 2 + m_mu**2).to_numpy()
    e_mat = np.sort(np.vstack([e1, e2, e3]).T, axis=1)

    e_soft = e_mat[:, 0]
    e_sublead = e_mat[:, 1]
    e_lead = e_mat[:, 2]
    w_mc = sel_mc["mc_weight"].to_numpy()
    return e_lead, e_soft, e_sublead, w_mc


@app.cell
def _(
    OUTPUT_ROOT,
    ROOT,
    ddf_root,
    df_ds,
    df_sf,
    e_lead,
    e_soft,
    e_sublead,
    np,
    qdc_bar_ds,
    qdc_bar_sf,
    w_bar_ds,
    w_bar_sf,
    w_mc,
):
    def create_detector_graphs(df, det_name, n_planes, y_label_hits, y_label_qdc):
      n_pts = len(df)
      x = np.ascontiguousarray(df["energy_gev"], dtype=np.float64)
      ex = np.zeros(n_pts, dtype=np.float64)

      gr_mean_hits = ROOT.TGraphErrors(
          n_pts, x,
          np.ascontiguousarray(df["mean_hits"], dtype=np.float64),
          ex,
          np.ascontiguousarray(df["sem_hits"], dtype=np.float64),
      )
      gr_mean_hits.SetName("gr_mean_hits_plane")
      gr_mean_hits.SetTitle(
          f"{det_name};E_{{#mu}} [GeV];{y_label_hits}"
      )

      gr_median_hits = ROOT.TGraph(
          n_pts, x, np.ascontiguousarray(df["median_hits"], dtype=np.float64)
      )
      gr_median_hits.SetName("gr_median_hits_plane")
      gr_median_hits.SetTitle(
          f"{det_name};E_{{#mu}} [GeV];{y_label_hits}"
      )

      eyl_h = np.ascontiguousarray(
          df["median_hits"] - df["q16_hits"], dtype=np.float64
      )
      eyh_h = np.ascontiguousarray(
          df["q84_hits"] - df["median_hits"], dtype=np.float64
      )
      gr_band_hits = ROOT.TGraphAsymmErrors(
          n_pts, x,
          np.ascontiguousarray(df["median_hits"], dtype=np.float64),
          ex, ex, eyl_h, eyh_h,
      )
      gr_band_hits.SetName("gr_band_hits_plane")
      gr_band_hits.SetTitle(
          f"{det_name} 68% containment hits/plane;E_{{#mu}} [GeV];{y_label_hits}"
      )

      gr_mean_qdc = ROOT.TGraphErrors(
          n_pts, x,
          np.ascontiguousarray(df["mean_qdc"], dtype=np.float64),
          ex,
          np.ascontiguousarray(df["sem_qdc"], dtype=np.float64),
      )
      gr_mean_qdc.SetName("gr_mean_qdc_plane")
      gr_mean_qdc.SetTitle(
          f"{det_name};E_{{#mu}} [GeV];{y_label_qdc}"
      )

      gr_median_qdc = ROOT.TGraph(
          n_pts, x, np.ascontiguousarray(df["median_qdc"], dtype=np.float64)
      )
      gr_median_qdc.SetName("gr_median_qdc_plane")
      gr_median_qdc.SetTitle(
          f"{det_name};E_{{#mu}} [GeV];{y_label_qdc}"
      )

      eyl_q = np.ascontiguousarray(
          df["median_qdc"] - df["q16_qdc"], dtype=np.float64
      )
      eyh_q = np.ascontiguousarray(
          df["q84_qdc"] - df["median_qdc"], dtype=np.float64
      )
      gr_band_qdc = ROOT.TGraphAsymmErrors(
          n_pts, x,
          np.ascontiguousarray(df["median_qdc"], dtype=np.float64),
          ex, ex, eyl_q, eyh_q,
      )
      gr_band_qdc.SetName("gr_band_qdc_plane")
      gr_band_qdc.SetTitle(
          f"{det_name} 68% containment QDC/plane;E_{{#mu}} [GeV];{y_label_qdc}"
      )

      gr_mean_emip = ROOT.TGraphErrors(
          n_pts, x,
          np.ascontiguousarray(df["mean_emip"], dtype=np.float64),
          ex,
          np.ascontiguousarray(df["sem_emip"], dtype=np.float64),
      )
      gr_mean_emip.SetName("gr_mean_emip")
      gr_mean_emip.SetTitle(
          f"{det_name};E_{{#mu}} [GeV];#varepsilon_{{mip}}"
      )

      gr_median_emip = ROOT.TGraph(
          n_pts, x, np.ascontiguousarray(df["median_emip"], dtype=np.float64)
      )
      gr_median_emip.SetName("gr_median_emip")
      gr_median_emip.SetTitle(
          f"{det_name};E_{{#mu}} [GeV];#varepsilon_{{mip}}"
      )

      eyl_e = np.ascontiguousarray(
          df["median_emip"] - df["q16_emip"], dtype=np.float64
      )
      eyh_e = np.ascontiguousarray(
          df["q84_emip"] - df["median_emip"], dtype=np.float64
      )
      gr_band_emip = ROOT.TGraphAsymmErrors(
          n_pts, x,
          np.ascontiguousarray(df["median_emip"], dtype=np.float64),
          ex, ex, eyl_e, eyh_e,
      )
      gr_band_emip.SetName("gr_band_emip")
      gr_band_emip.SetTitle(
          f"{det_name} 68% containment MIP-equivalence;E_{{#mu}} [GeV];#varepsilon_{{mip}}"
      )

      gr_mean_rhomip = ROOT.TGraphErrors(
          n_pts, x,
          np.ascontiguousarray(df["mean_rhomip"], dtype=np.float64),
          ex,
          np.ascontiguousarray(df["sem_rhomip"], dtype=np.float64),
      )
      gr_mean_rhomip.SetName("gr_mean_rhomip")
      gr_mean_rhomip.SetTitle(
          f"{det_name} Mean MIP Charge Density;Muon Energy E_{{#mu}} [GeV];#rho_{{mip}}"
      )

      gr_median_rhomip = ROOT.TGraph(
          n_pts, x, np.ascontiguousarray(df["median_rhomip"], dtype=np.float64)
      )
      gr_median_rhomip.SetName("gr_median_rhomip")
      gr_median_rhomip.SetTitle(
          f"{det_name} Median MIP charge density;E_{{#mu}} [GeV];#rho_{{mip}}"
      )

      eyl_r = np.ascontiguousarray(
          df["median_rhomip"] - df["q16_rhomip"], dtype=np.float64
      )
      eyh_r = np.ascontiguousarray(
          df["q84_rhomip"] - df["median_rhomip"], dtype=np.float64
      )
      gr_band_rhomip = ROOT.TGraphAsymmErrors(
          n_pts, x,
          np.ascontiguousarray(df["median_rhomip"], dtype=np.float64),
          ex, ex, eyl_r, eyh_r,
      )
      gr_band_rhomip.SetName("gr_band_rhomip")
      gr_band_rhomip.SetTitle(
          f"{det_name} 68% containment MIP charge density;E_{{#mu}} [GeV];#rho_{{mip}}"
      )

      return {
          "gr_mean_hits_plane": gr_mean_hits,
          "gr_median_hits_plane": gr_median_hits,
          "gr_band_hits_plane": gr_band_hits,
          "gr_mean_qdc_plane": gr_mean_qdc,
          "gr_median_qdc_plane": gr_median_qdc,
          "gr_band_qdc_plane": gr_band_qdc,
          "gr_mean_emip": gr_mean_emip,
          "gr_median_emip": gr_median_emip,
          "gr_band_emip": gr_band_emip,
          "gr_mean_rhomip": gr_mean_rhomip,
          "gr_median_rhomip": gr_median_rhomip,
          "gr_band_rhomip": gr_band_rhomip,
      }


    sf_graphs = create_detector_graphs(
        df_sf, "SciFi", 10, "<N_{hits}^{SF} / 10>", "<QDC^{SF} / 10>"
    )
    ds_graphs = create_detector_graphs(
        df_ds, "Downstream", 4, "<N_{hits}^{DS} / 4>", "<QDC^{DS} / 4>"
    )

    bins_log = np.logspace(1.0, 3.60, 46)
    # bins_log = np.logspace(1.0, 3.01, 36)
    h_e_soft = ROOT.TH1D(
        "h_e_mu1",
        "E_{1};Muon energy [GeV];Normalized events",
        35,
        bins_log,
    )
    h_e_sublead = ROOT.TH1D(
        "h_e_mu2",
        "E_{2};Muon Energy [GeV];Normalized events",
        35,
        bins_log,
    )
    h_e_lead = ROOT.TH1D(
        "h_e_lead",
        "Leading Muon Energy E_{lead};Muon Energy [GeV];Normalized events",
        35,
        bins_log,
    )

    for val, w in zip(e_soft, w_mc):
      h_e_soft.Fill(val, w)
    for val, w in zip(e_sublead, w_mc):
      h_e_sublead.Fill(val, w)
    for val, w in zip(e_lead, w_mc):
      h_e_lead.Fill(val, w)

    c_sf_hits_qdc = ROOT.TCanvas(
        "c_scifi_hits_qdc", "SciFi Hits & QDC Response", 1200, 600
    )
    c_sf_hits_qdc.Divide(2, 1)
    c_sf_hits_qdc.cd(1).SetLogx(1)
    sf_graphs["gr_mean_hits_plane"].Draw("APL")
    c_sf_hits_qdc.cd(2).SetLogx(1)
    sf_graphs["gr_mean_qdc_plane"].Draw("APL")

    c_sf_emip_rhomip = ROOT.TCanvas(
        "c_scifi_emip_rhomip", "SciFi MIP-Equivalence & Charge Density", 1200, 600
    )
    c_sf_emip_rhomip.Divide(2, 1)
    c_sf_emip_rhomip.cd(1).SetLogx(1)
    sf_graphs["gr_mean_emip"].Draw("APL")
    c_sf_emip_rhomip.cd(2).SetLogx(1)
    sf_graphs["gr_mean_rhomip"].Draw("APL")

    c_ds_hits_qdc = ROOT.TCanvas(
        "c_ds_hits_qdc", "Downstream Hits & QDC Response", 1200, 600
    )
    c_ds_hits_qdc.Divide(2, 1)
    c_ds_hits_qdc.cd(1).SetLogx(1)
    ds_graphs["gr_mean_hits_plane"].Draw("APL")
    c_ds_hits_qdc.cd(2).SetLogx(1)
    ds_graphs["gr_mean_qdc_plane"].Draw("APL")

    metadata_dict = {
        "name": "metadata",
        "w_bar_SF": w_bar_sf,
        "qdc_bar_SF": qdc_bar_sf,
        "w_bar_DS": w_bar_ds,
        "qdc_bar_DS": qdc_bar_ds,
        "mip_threshold_gev": 40.0,
        "energy_points_gev": list(df_sf["energy_gev"]),
        "selection": "[SF(3,1) | DS(3,1)] & (SF_nc & DS_nc)",
    }

    f_out = ROOT.TFile(str(OUTPUT_ROOT), "RECREATE")

    ddf_root.write_metadata(metadata_dict, f_out, directory="", close_file=False)
    ddf_root.save_to_root(
        list(sf_graphs.values()), fout=f_out, directory="scifi", print_filename=False
    )
    ddf_root.save_to_root(
        list(ds_graphs.values()),
        fout=f_out,
        directory="DS",
        print_filename=False,
    )
    ddf_root.save_to_root(
        [h_e_soft, h_e_sublead, h_e_lead],
        fout=f_out,
        directory="trimuon_mc",
        print_filename=False,
    )
    ddf_root.save_to_root(
        [c_sf_hits_qdc, c_sf_emip_rhomip, c_ds_hits_qdc],
        fout=f_out,
        directory="canvases",
        print_filename=False,
    )

    f_out.Close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Exploration
    """)
    return


@app.cell
def _(json, uproot):
    ROOT_FILE = '/eos/user/i/idioniso/sndMuTri/out/pGun_response.root'
    with uproot.open(ROOT_FILE) as f_in:
        meta = json.loads(str(f_in['metadata']))
        w_bar_sf_1 = meta.get('w_bar_SF', 2.13)
        qdc_bar_sf_1 = meta.get('qdc_bar_SF', 2.48)
    return ROOT_FILE, w_bar_sf_1


@app.cell
def _(ddf_root, np):
    def get_bin_probability(root_hist_path):
        """Extracts bin edges and normalizes bin heights such that each height = P(E in bin)."""
        x_centers, counts, edges, x_errs, y_errs = ddf_root.to_numpy(root_hist_path)
        total_weight = np.sum(counts)
        prob_per_bin = counts / total_weight if total_weight > 0 else counts
        return edges, prob_per_bin/100
    return (get_bin_probability,)


@app.cell
def _(
    ROOT_FILE,
    add_label,
    ddf_root,
    get_bin_probability,
    plt,
    ticker,
    w_bar_sf_1,
):
    (fig, ax) = plt.subplots(figsize=(8.5, 6.5))
    ax1_twin = ax.twinx()
    (edges_soft, prob_soft) = get_bin_probability(f'{ROOT_FILE}:trimuon_mc/h_e_mu1')
    (edges_sub, prob_sub) = get_bin_probability(f'{ROOT_FILE}:trimuon_mc/h_e_mu2')
    (edges_lead, prob_lead) = get_bin_probability(f'{ROOT_FILE}:trimuon_mc/h_e_lead')
    ax1_twin.fill_between(edges_soft[:-1], prob_soft, step='post', facecolor='#9E9E9E', alpha=0.25, edgecolor='#616161', linestyle=':', lw=1.4, label='Muon 1 $E_{\\mathrm{soft}}$', zorder=1)
    ax1_twin.step(edges_sub[:-1], prob_sub, where='post', color='#424242', linestyle='--', lw=1.6, label='Muon 2 $E_{\\mathrm{sublead}}$', zorder=2)
    ax1_twin.step(edges_lead[:-1], prob_lead, where='post', color='black', linestyle='-', lw=1.8, label='Parent muon $E_{\\mathrm{lead}}$', zorder=3)
    max_prob = max(prob_soft.max(), prob_sub.max(), prob_lead.max())
    ax1_twin.set_ylabel('Muon spectra', fontweight='bold', color='#212121', fontsize=11.5, labelpad=8)
    ax1_twin.set_ylim(0, max_prob * 1.4)
    ax1_twin.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1_twin.tick_params(axis='y', labelsize=10.5, length=5, width=1.0)
    ax1_twin.tick_params(axis='y', which='minor', length=2.5, width=0.8)
    df_hits = ddf_root.to_pandas(f'{ROOT_FILE}:scifi/gr_mean_hits_plane')
    df_hits_band = ddf_root.to_pandas(f'{ROOT_FILE}:scifi/gr_band_hits_plane')
    df_hits_med = ddf_root.to_pandas(f'{ROOT_FILE}:scifi/gr_median_hits_plane')
    ax.fill_between(df_hits_band['x'], df_hits_band['y'] - df_hits_band['eyl'], df_hits_band['y'] + df_hits_band['eyh'], color='#1976D2', alpha=0.15, label='$\\mu$ particle gun $68\\%$ containment', zorder=4)
    ax.plot(df_hits_med['x'], df_hits_med['y'], 's--', color='#64B5F6', mfc='none', ms=4.0, lw=1.2, label='Median channels / plane', zorder=5)
    ax.errorbar(df_hits['x'], df_hits['y'], yerr=df_hits['ey'], fmt='o-', color='#1565C0', ecolor='#1565C0', lw=2.2, ms=5.0, capsize=2.5, label='Mean activated channels $\\langle N_{\\mathrm{hits}}^{\\mathrm{SF}} / 10 \\rangle$', zorder=6)
    ax.axhline(w_bar_sf_1, color='#388E3C', linestyle='--', lw=1.6, label=f'SciFi MIP baseline ($\\bar{{w}}_{{\\mathrm{{SF}}}} = {w_bar_sf_1:.2f}$)', zorder=3)
    ax.set_xscale('log')
    # Sub-Leading Muon
    ax.set_xlim(9, 6800)
    ax.set_ylim(1.0, 9.5)
    ax.set_xlabel('Muon energy $E_{\\mu}$ [ GeV ]', fontweight='bold', fontsize=12, labelpad=8)
    ax.set_ylabel('$\\langle N_{\\mathrm{hits}}^{\\mathrm{SF}} / 10 \\rangle$', fontweight='bold', color='#1565C0', fontsize=12, labelpad=8)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.96), frameon=True, framealpha=0.92, fontsize=12.0, title_fontsize=9.5)
    ax1_twin.legend(loc='upper right', bbox_to_anchor=(0.98, 0.96), frameon=True, framealpha=0.92, fontsize=12.0, title_fontsize=9.5)
    add_label(loc='top-left', supp_loc='top-right', mainText='', extraText='Muon Particle Gun and Trimuon Simulations', suppText='', com=None, lumi=None, ax=ax)
    plt.tight_layout()
    plt.show()
    # Leading Muon
    plt.close(fig)
    return


@app.cell
def _(add_label, ddf_root, json, np, plt, ticker, uproot):
    ROOT_FILE_1 = '/eos/user/i/idioniso/sndMuTri/out/pGun_response.root'
    with uproot.open(ROOT_FILE_1) as f_1:
        meta_1 = json.loads(str(f_1['metadata'])) if 'metadata' in f_1 else {}
        w_bar_sf_2 = meta_1.get('w_bar_SF', 2.183)
        qdc_bar_sf_2 = meta_1.get('qdc_bar_SF', 2.576)
        w_bar_ds_1 = meta_1.get('w_bar_DS', 2.12)
        qdc_bar_ds_1 = meta_1.get('qdc_bar_DS', 102.34)

    # ==============================================================================
    # 0. CONFIGURATION & METADATA
    def get_bin_probability_1(root_hist_path):
        """Extracts bin edges and normalizes bin heights such that each height = P(E in bin)."""
        (x_centers, counts, edges, x_errs, y_errs) = ddf_root.to_numpy(root_hist_path)
        total_weight = np.sum(counts)
        prob_per_bin = counts / total_weight if total_weight > 0 else counts
        return (edges, prob_per_bin / 100.0)
    CONFIGS = {('scifi', 'hits_plane'): {'ylabel': '$\\langle N_{\\mathrm{hits}}^{\\mathrm{SF}} / 10 \\rangle$', 'mean_label': 'Mean activated channels $\\langle N_{\\mathrm{hits}}^{\\mathrm{SF}} / 10 \\rangle$', 'med_label': 'Median channels / plane', 'baseline_val': w_bar_sf_2, 'baseline_label': f'SciFi MIP baseline ($\\bar{{w}}_{{\\mathrm{{SF}}}} = {w_bar_sf_2:.2f}$)', 'xlim': (8.0, 6000.0), 'ylim': (1.0, 5.5), 'color': '#1565C0', 'med_color': '#64B5F6', 'band_color': '#1976D2'}, ('scifi', 'qdc_plane'): {'ylabel': '$\\langle \\mathrm{QDC}^{\\mathrm{SF}} / 10 \\rangle$ [ ADC / Plane ]', 'mean_label': 'Mean SciFi QDC $\\langle \\mathrm{QDC}^{\\mathrm{SF}} / 10 \\rangle$', 'med_label': 'Median SciFi QDC / plane', 'baseline_val': qdc_bar_sf_2, 'baseline_label': f'SciFi MIP baseline ($\\overline{{\\mathrm{{QDC}}}}_{{\\mathrm{{SF}}}} = {qdc_bar_sf_2:.2f}$)', 'xlim': (8.0, 6000.0), 'ylim': (0.0, 8.0), 'color': '#1565C0', 'med_color': '#64B5F6', 'band_color': '#1976D2'}, ('scifi', 'emip'): {'ylabel': '$\\varepsilon_{\\mathrm{mip}}', 'mean_label': 'Mean $\\varepsilon_{\\mathrm{mip}}$ response', 'med_label': 'Median $\\varepsilon_{\\mathrm{mip}}$', 'baseline_val': 1.0, 'baseline_label': 'MIP baseline ($\\varepsilon_{\\mathrm{mip}} = 1.00$)', 'xlim': (8.0, 6000.0), 'ylim': (0.0, 3.0), 'color': '#1565C0', 'med_color': '#64B5F6', 'band_color': '#1976D2'}, ('scifi', 'rhomip'): {'ylabel': '$\\rho_{\\mathrm{mip}}', 'mean_label': 'Mean $\\rho_{\\mathrm{mip}}$ response', 'med_label': 'Median $\\rho_{\\mathrm{mip}}$', 'baseline_val': 1.0, 'baseline_label': 'MIP baseline ($\\rho_{\\mathrm{mip}} = 1.00$)', 'xlim': (8.0, 6000.0), 'ylim': (0.0, 3.0), 'color': '#1565C0', 'med_color': '#64B5F6', 'band_color': '#1976D2'}, ('DS', 'hits_plane'): {'ylabel': '$\\langle N_{\\mathrm{hits}}^{\\mathrm{DS}} / 10 \\rangle$', 'mean_label': 'Mean activated channels $\\langle N_{\\mathrm{hits}}^{\\mathrm{DS}} / 10 \\rangle$', 'med_label': 'Median channels / plane', 'baseline_val': w_bar_ds_1, 'baseline_label': f'DS MIP baseline ($\\bar{{w}}_{{\\mathrm{{DS}}}} = {w_bar_ds_1:.2f}$)', 'xlim': (8.0, 6000.0), 'ylim': (1.0, 5.5), 'color': '#E65100', 'med_color': '#FFB74D', 'band_color': '#FB8C00'}, ('DS', 'qdc_plane'): {'ylabel': '$\\langle \\mathrm{QDC}^{\\mathrm{DS}} / 10 \\rangle$', 'mean_label': 'Mean DS QDC $\\langle \\mathrm{QDC}^{\\mathrm{DS}} / 10 \\rangle$', 'med_label': 'Median DS QDC / plane', 'baseline_val': qdc_bar_ds_1, 'baseline_label': f'DS MIP baseline ($\\overline{{\\mathrm{{QDC}}}}_{{\\mathrm{{DS}}}} = {qdc_bar_ds_1:.1f}$)', 'xlim': (8.0, 6000.0), 'ylim': (0.0, 250.0), 'color': '#E65100', 'med_color': '#FFB74D', 'band_color': '#FB8C00'}, ('DS', 'emip'): {'ylabel': '$\\varepsilon_{\\mathrm{mip}}^{\\mathrm{DS}}', 'mean_label': 'Mean $\\varepsilon_{\\mathrm{mip}}^{\\mathrm{DS}}$ response', 'med_label': 'Median $\\varepsilon_{\\mathrm{mip}}^{\\mathrm{DS}}$', 'baseline_val': 1.0, 'baseline_label': 'MIP baseline ($\\varepsilon_{\\mathrm{mip}}^{\\mathrm{DS}} = 1.00$)', 'xlim': (8.0, 6000.0), 'ylim': (0.0, 3.0), 'color': '#E65100', 'med_color': '#FFB74D', 'band_color': '#FB8C00'}, ('DS', 'rhomip'): {'ylabel': '$\\rho_{\\mathrm{mip}}^{\\mathrm{DS}}', 'mean_label': 'Mean $\\rho_{\\mathrm{mip}}^{\\mathrm{DS}}$ response', 'med_label': 'Median $\\rho_{\\mathrm{mip}}^{\\mathrm{DS}}$', 'baseline_val': 1.0, 'baseline_label': 'MIP baseline ($\\rho_{\\mathrm{mip}}^{\\mathrm{DS}} = 1.00$)', 'xlim': (8.0, 6000.0), 'ylim': (0.0, 3.0), 'color': '#E65100', 'med_color': '#FFB74D', 'band_color': '#FB8C00'}}

    def plot_response_variable(detector='scifi', var='rhomip', xlim=None, ylim=None, save_path=None, show=True):
        """Plots particle gun response vs trimuon muon energy spectra for any of the 8 variables."""
        cfg = CONFIGS[detector, var]
        gr_mean_path = f'{ROOT_FILE_1}:{detector}/gr_mean_{var}'
        gr_band_path = f'{ROOT_FILE_1}:{detector}/gr_band_{var}'
        gr_med_path = f'{ROOT_FILE_1}:{detector}/gr_median_{var}'
        df_mean = ddf_root.to_pandas(gr_mean_path)
        df_band = ddf_root.to_pandas(gr_band_path)
        df_med = ddf_root.to_pandas(gr_med_path)
        (edges_soft, prob_soft) = get_bin_probability_1(f'{ROOT_FILE_1}:trimuon_mc/h_e_mu1')
        (edges_sub, prob_sub) = get_bin_probability_1(f'{ROOT_FILE_1}:trimuon_mc/h_e_mu2')
    # Config dictionary for all 8 response variables
        (edges_lead, prob_lead) = get_bin_probability_1(f'{ROOT_FILE_1}:trimuon_mc/h_e_lead')
        (fig, ax) = plt.subplots(figsize=(8.5, 6.2))  # ----------------- SciFi Variables -----------------
        ax_twin = ax.twinx()
        ax_twin.fill_between(edges_soft[:-1], prob_soft, step='post', facecolor='#9E9E9E', alpha=0.25, edgecolor='#616161', linestyle=':', lw=1.4, label='Muon 1 $E_{\\mathrm{soft}}$', zorder=1)
        ax_twin.step(edges_sub[:-1], prob_sub, where='post', color='#424242', linestyle='--', lw=1.6, label='Muon 2 $E_{\\mathrm{sublead}}$', zorder=2)
        ax_twin.step(edges_lead[:-1], prob_lead, where='post', color='black', linestyle='-', lw=1.8, label='Parent muon $E_{\\mathrm{lead}}$', zorder=3)
        max_prob = max(prob_soft.max(), prob_sub.max(), prob_lead.max())
        ax_twin.set_ylabel('Trimuon muon spectra', fontweight='bold', color='#212121', fontsize=11.5, labelpad=8)
        ax_twin.set_ylim(0, max_prob * 1.35)
        ax_twin.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_twin.tick_params(axis='y', labelsize=10.5, length=5, width=1.0)
        ax_twin.tick_params(axis='y', which='minor', length=2.5, width=0.8)
        ax.fill_between(df_band['x'], df_band['y'] - df_band['eyl'], df_band['y'] + df_band['eyh'], color=cfg['band_color'], alpha=0.18, label='$\\mu$ particle gun $68\\%$ containment', zorder=4)
        ax.plot(df_med['x'], df_med['y'], 's--', color=cfg['med_color'], mfc='none', ms=4.0, lw=1.2, label=cfg['med_label'], zorder=5)
        ax.errorbar(df_mean['x'], df_mean['y'], yerr=df_mean['ey'], fmt='o-', color=cfg['color'], ecolor=cfg['color'], lw=2.2, ms=5.0, capsize=2.5, label=cfg['mean_label'], zorder=6)
        ax.axhline(cfg['baseline_val'], color='#388E3C', linestyle='--', lw=1.6, label=cfg['baseline_label'], zorder=3)
        ax.set_xscale('log')
        ax.set_xlim(xlim if xlim is not None else cfg['xlim'])
        ax.set_ylim(ylim if ylim is not None else cfg['ylim'])
        ax.set_xlabel('Muon energy $E_{\\mu}$ [ GeV ]', fontweight='bold', fontsize=12, labelpad=8)
        ax.set_ylabel(cfg['ylabel'], fontweight='bold', color=cfg['color'], fontsize=11.5, labelpad=8)
        ax.grid(True, which='major', linestyle='--', alpha=0.35, color='gray', zorder=1)
        ax.grid(True, which='minor', linestyle=':', alpha=0.15, color='gray', zorder=1)
        ax.tick_params(axis='both', which='major', direction='in', top=True, labelsize=11, length=6, width=1.1)
        ax.tick_params(axis='both', which='minor', direction='in', top=True, labelsize=9, length=3, width=0.8)
        ax.legend(loc='upper left', frameon=True, framealpha=0.92, edgecolor='#BDBDBD', facecolor='white', fontsize=9.2)
        ax_twin.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#BDBDBD', facecolor='white', fontsize=9.2)
        add_label(loc='top-left', supp_loc='top-right', mainText='', extraText='Muon Particle Gun Trimuon Simulations', suppText='', com=None, lumi=None, ax=ax)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    # 1. GENERAL PLOTTING FUNCTION
        plt.close(fig)  # ----------------- Downstream (DS) Variables -----------------  # Graph paths in ROOT  # Read Trident Spectra  # 1. Right Axis: Trident Muon Energy Spectra  # 2. Left Axis: Particle Gun Response Graphs  # 68% Containment Shaded Band  # Median Line  # Mean Points with Errors  # MIP Baseline Horizontal Reference  # Axes limits & styling  # Grid & Ticks
    return ROOT_FILE_1, get_bin_probability_1, plot_response_variable


@app.cell
def _(plot_response_variable):
    plot_response_variable(detector="scifi", var="rhomip")
    return


@app.cell
def _():
    TRACKING_COLS = [
        "n_lines_sf_xz", "n_lines_sf_yz", "n_lines_ds_xz", "n_lines_ds_yz",
        "xz_sf_m1", "xz_sf_c1", "yz_sf_m1", "yz_sf_c1",
        "xz_sf_m2", "xz_sf_c2", "yz_sf_m2", "yz_sf_c2",
        "xz_sf_m3", "xz_sf_c3", "yz_sf_m3", "yz_sf_c3",
        "xz_ds_m1", "xz_ds_c1", "yz_ds_m1", "yz_ds_c1",
        "xz_ds_m2", "xz_ds_c2", "yz_ds_m2", "yz_ds_c2",
        "xz_ds_m3", "xz_ds_c3", "yz_ds_m3", "yz_ds_c3"
    ]

    KINEMATIC_COLS = ["is_signal", "region_type", "mc_weight", "p_mu_in", "p_mu_minus", "p_mu_plus"]
    return


@app.cell
def _(Path, ROOT, np, pd):
    from scipy import stats
    ROOT.gROOT.SetBatch(True)
    PGUN_DIR_1 = Path('/eos/user/i/idioniso/1_Data/Monte_Carlo/pGun/muons')
    sample_energies = ['10.GeV', '35.GeV', '100.GeV', '300.GeV', '800.GeV']
    colors = {'10.GeV': '#1E88E5', '35.GeV': '#43A047', '100.GeV': '#FB8C00', '300.GeV': '#E53935', '800.GeV': '#8E24AA'}
    data_records = []
    for edir_1 in sample_energies:
        digi_file_1 = PGUN_DIR_1 / edir_1 / 'sndLHC.PG_13-TGeant4_digCPP.root'
        if not digi_file_1.exists():
            continue
    # ==============================================================================
    # 1. DATA EXTRACTION
        energy_val_1 = float(edir_1.replace('.GeV', ''))
        f_2 = ROOT.TFile.Open(str(digi_file_1))
        if not f_2 or f_2.IsZombie():
            continue
        tree_1 = f_2.Get('cbmsim')
        if not tree_1:
            f_2.Close()
            continue
        n_entries_1 = min(800, tree_1.GetEntries())
        for i_1 in range(n_entries_1):
            tree_1.GetEntry(i_1)
            valid_signals = [h.GetSignal() for h in tree_1.Digi_ScifiHits if h.isValid()]
            n_sf_1 = len(valid_signals)
            if n_sf_1 >= 5:
                hits_per_plane = n_sf_1 / 10.0
                qdc_per_plane = sum(valid_signals) / 10.0
                data_records.append({'energy': energy_val_1, 'energy_str': edir_1, 'hits_plane': hits_per_plane, 'qdc_plane': qdc_per_plane})
        f_2.Close()
    df_corr = pd.DataFrame(data_records)
    x_hits = df_corr['hits_plane'].to_numpy()
    y_qdc = df_corr['qdc_plane'].to_numpy()
    (r_pearson, p_pearson) = stats.pearsonr(x_hits, y_qdc)
    (r_spearman, p_spearman) = stats.spearmanr(x_hits, y_qdc)
    (slope, intercept, r_val, p_val, std_err) = stats.linregress(x_hits, y_qdc)
    df_corr['hit_bin'] = np.round(df_corr['hits_plane'] * 2) / 2.0
    prof = df_corr.groupby('hit_bin')['qdc_plane'].agg(mean='mean', sem=lambda x: np.std(x) / np.sqrt(len(x)), count='count').reset_index()
    # 2. STATISTICAL CORRELATION METRICS
    prof = prof[prof['count'] >= 10]  # 0.5-width bins
    return colors, df_corr, sample_energies, stats


@app.cell
def _(df_corr, stats):
    x_hits_1 = df_corr['hits_plane'].to_numpy()
    y_qdc_1 = df_corr['qdc_plane'].to_numpy()
    (r_pearson_1, p_pearson_1) = stats.pearsonr(x_hits_1, y_qdc_1)
    (r_spearman_1, p_spearman_1) = stats.spearmanr(x_hits_1, y_qdc_1)
    return r_pearson_1, r_spearman_1


@app.cell
def _(
    add_label,
    colors,
    df_corr,
    np,
    plt,
    r_pearson_1,
    r_spearman_1,
    sample_energies,
):
    BIN_WIDTH = 0.5
    df_corr['hit_bin'] = np.round(df_corr['hits_plane'] / BIN_WIDTH) * BIN_WIDTH
    prof_1 = df_corr.groupby('hit_bin')['qdc_plane'].agg(mean='mean', median='median', q16=lambda x: float(np.percentile(x, 16)), q84=lambda x: float(np.percentile(x, 84)), sem=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 1.0, count='count').reset_index()
    prof_1 = prof_1[prof_1['count'] >= 10].copy()
    yerr_asym = [np.maximum(0, (prof_1['mean'] - prof_1['q16']).to_numpy()), np.maximum(0, (prof_1['q84'] - prof_1['mean']).to_numpy())]
    xerr_bins = BIN_WIDTH / 2.0
    weights = 1.0 / np.where(prof_1['sem'] > 0, prof_1['sem'], 1.0)
    (slope_1, intercept_1) = np.polyfit(prof_1['hit_bin'].to_numpy(), prof_1['mean'].to_numpy(), deg=1, w=weights)
    (fig_1, ax_1) = plt.subplots(figsize=(8.5, 6.5))
    for edir_2 in sample_energies:
        sub = df_corr[df_corr['energy_str'] == edir_2]
        if sub.empty:
            continue
        ax_1.scatter(sub['hits_plane'], sub['qdc_plane'], color=colors[edir_2], label=f"$E_\\mu = {sub['energy'].iloc[0]:.0f}\\ \\mathrm{{GeV}}$", alpha=0.3, s=20, edgecolors='none', zorder=2)
    x_line = np.linspace(0.8, 15.0, 100)
    ax_1.plot(x_line, slope_1 * x_line + intercept_1, color='darkred', linestyle='--', lw=2.2, zorder=4, label=f'Profile linear fit ($\\mathrm{{slope}} = {slope_1:.2f}$)')
    ax_1.errorbar(prof_1['hit_bin'], prof_1['mean'], xerr=xerr_bins, yerr=yerr_asym, fmt='o', color='black', ecolor='black', elinewidth=1.6, capsize=3.0, capthick=1.4, markersize=5.5, zorder=5, label='Profile $\\langle \\mathrm{QDC} \\rangle$')
    ax_1.set_xlabel('$\\langle N_{\\mathrm{hits}}^{\\mathrm{SF}} / 10 \\rangle$', fontweight='bold', fontsize=12, labelpad=8)
    ax_1.set_ylabel('$\\langle \\mathrm{QDC}^{\\mathrm{SF}} / 10 \\rangle$', fontweight='bold', fontsize=12, labelpad=8)
    ax_1.set_xlim(0.5, 15.5)
    ax_1.set_ylim(0.0, 24.0)
    # Asymmetric 68% containment vertical errors [lower, upper]
    ax_1.legend(loc='upper left', frameon=True, framealpha=0.8, fontsize=12.0, title=f'($r = {r_pearson_1:.3f},\\ \\rho = {r_spearman_1:.3f}$)', title_fontsize=11.0)
    add_label(loc='top-left', supp_loc='top-right', mainText='', extraText='Particle Gun Muon Simulation', style='formal', com=None, lumi=None, ax=ax_1)
    plt.tight_layout()
    plt.show()
    # Horizontal bin-width errors (half bin width = 0.25)
    # Weighted linear fit on profile points (weighted by 1 / SEM)
    # ==============================================================================
    # 2. PLOTTING
    # 2a. Background Scatter Points per Energy Sample
    # 2b. Weighted Linear Fit Line
    # ax.grid(True, which="major", linestyle="--", alpha=0.4, color="gray", zorder=1)
    # ax.minorticks_on()
    # ax.grid(True, which="minor", linestyle=":", alpha=0.2, color="gray", zorder=1)
    # ax.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.2)
    # ax.tick_params(axis="both", which="minor", labelsize=9, length=3, width=0.8)
    # Legend
    plt.close()
    return


app._unparsable_cell(
    r"""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax1_twin = ax.twinx()
    
        # ==============================================================================
        # 1. TRIDENT MUON SPECTRA (RIGHT AXIS)
        # ==============================================================================
        edges_soft, prob_soft = get_bin_probability(f\"{ROOT_FILE}:trimuon_mc/h_e_mu1\")
        edges_sub, prob_sub = get_bin_probability(f\"{ROOT_FILE}:trimuon_mc/h_e_mu2\")
        edges_lead, prob_lead = get_bin_probability(f\"{ROOT_FILE}:trimuon_mc/h_e_lead\")
    
        # Softest Muon
        ax1_twin.fill_between(
            edges_soft[:-1],
            prob_soft,
            step=\"post\",
            facecolor=\"#9E9E9E\",
            alpha=0.25,
            edgecolor=\"#616161\",
            linestyle=\":\",
            lw=1.4,
            label=r\"Muon 1 $E_{\mathrm{soft}}$\",
            zorder=1,
        )
    
        # Sub-Leading Muon
        ax1_twin.step(
            edges_sub[:-1],
            prob_sub,
            where=\"post\",
            color=\"#424242\",
            linestyle=\"--\",
            lw=1.6,
            label=r\"Muon 2 $E_{\mathrm{sublead}}$\",
            zorder=2,
        )
    
        # Leading Muon
        ax1_twin.step(
            edges_lead[:-1],
            prob_lead,
            where=\"post\",
            color=\"black\",
            linestyle=\"-\",
            lw=1.8,
            label=r\"Parent muon $E_{\mathrm{lead}}$\",
            zorder=3,
        )
    
        max_prob = max(prob_soft.max(), prob_sub.max(), prob_lead.max())
        ax1_twin.set_ylabel(
            r\"Muon spectra\",
            fontweight=\"bold\",
            color=\"#212121\",
            fontsize=11.5,
            labelpad=8,
        )
        ax1_twin.set_ylim(0, max_prob * 1.40)
        ax1_twin.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax1_twin.tick_params(axis=\"y\", labelsize=10.5, length=5, width=1.0)
        ax1_twin.tick_params(axis=\"y\", which=\"minor\", length=2.5, width=0.8)
    
        # ==============================================================================
        # 2. MIP-EQUIVALENCE RESPONSE (LEFT AXIS)
        # ==============================================================================
        df_emip = ddf_root.to_pandas(f\"{ROOT_FILE}:scifi/gr_mean_emip\")
        df_emip_band = ddf_root.to_pandas(f\"{ROOT_FILE}:scifi/gr_band_emip\")
        df_emip_med = ddf_root.to_pandas(f\"{ROOT_FILE}:scifi/gr_median_emip\")
    
        # 68% Containment Band
        ax.fill_between(
            df_emip_band[\"x\"],
            df_emip_band[\"y\"] - df_emip_band[\"eyl\"],
            df_emip_band[\"y\"] + df_emip_band[\"eyh\"],
            color=\"#80CBC4\",
            alpha=0.25,
            label=r\"$\mu$ particle gun $68\%$ containment\",
            zorder=4,
        )
    
        # Median MIP-Equivalence
        ax.plot(
            df_emip_med[\"x\"],
            df_emip_med[\"y\"],
            \"s--\",
            color=\"#4DB6AC\",
            mfc=\"none\",
            ms=4.0,
            lw=1.2,
            label=r\"Median MIP equivalent\",
            zorder=5,
        )
    
        # Mean MIP-Equivalence with SEM error bars
        ax.errorbar(
            df_emip[\"x\"],
            df_emip[\"y\"],
            yerr=df_emip[\"ey\"],
            fmt=\"o-\",
            color=\"#004D40\",
            ecolor=\"#004D40\",
            lw=2.2,
            ms=5.0,
            capsize=2.5,
            label=r\"Mean MIP equivalent $\langle \varepsilon_{\mathrm{mip}} \rangle$\",
            zorder=6,
        )
    
        # Single-MIP Baseline Reference Line (e_mip = 1.00)
        ax.axhline(
            1.00,
            color=\"#FF7043\",
            linestyle=\"--\",
            lw=1.6,
            label=r\"Single-MIP baseline ($\varepsilon_{\mathrm{mip}} = 1.00$)\",
            zorder=3,
        )
    
        # Axis Configuration
        ax.set_xscale(\"log\")
        ax.set_xlim(9, 6800)
        ax.set_ylim(0.4, 5.2)
        ax.set_xlabel(
            r\"Muon energy $E_{\mu}$ [ GeV ]\", fontweight=\"bold\", fontsize=12, labelpad=8
        )
        ax.set_ylabel(
            r\"$\varepsilon_{\mathrm{mip}}$\",
            fontweight=\"bold\",
            color=\"#004D40\",
            fontsize=13,
            labelpad=8,
        )
    
        # Ticks and Grid
        ax.grid(True, which=\"major\", linestyle=\"--\", alpha=0.4, color=\"gray\", zorder=0)
        ax.minorticks_on()
        ax.grid(True, which=\"minor\", linestyle=\":\", alpha=0.2, color=\"gray\", zorder=0)
        ax.tick_params(axis=\"both\", which=\"major\", labelsize=11, length=6, width=1.2)
        ax.tick_params(axis=\"both\", which=\"minor\", labelsize=9, length=3, width=0.8)
    
        # Legends
        ax.legend(
            loc=\"upper left\",
            bbox_to_anchor=(0.02, 0.96),
            frameon=True,
            framealpha=0.92,
            fontsize=12.0,
            title_fontsize=9.5,
        )
        ax1_twin.legend(
            loc=\"upper right\",
            bbox_to_anchor=(0.98, 0.96),
            frameon=True,
            framealpha=0.92,
            fontsize=12.0,
            title_fontsize=9.5,
        )

        # Header
        add_label(
            loc=\"top-left\",
            supp_loc=\"top-right\",
            mainText=\"\",
            extraText=\"Muon Particle Gun and Trimuon Simulations\",
            suppText=r\"\",
            com=None,
            lumi=None,
            ax=ax,
        )

        plt.tight_layout()
        plt.show()
        plt.close(fig)
    """,
    name="_"
)


@app.cell
def _(ROOT_FILE_1, add_label, ddf_root, get_bin_probability_1, plt, ticker):
    (fig_2, ax_2) = plt.subplots(figsize=(8.5, 6.5))
    ax1_twin_1 = ax_2.twinx()
    (edges_soft_1, prob_soft_1) = get_bin_probability_1(f'{ROOT_FILE_1}:trimuon_mc/h_e_mu1')
    (edges_sub_1, prob_sub_1) = get_bin_probability_1(f'{ROOT_FILE_1}:trimuon_mc/h_e_mu2')
    (edges_lead_1, prob_lead_1) = get_bin_probability_1(f'{ROOT_FILE_1}:trimuon_mc/h_e_lead')
    ax1_twin_1.fill_between(edges_soft_1[:-1], prob_soft_1, step='post', facecolor='#9E9E9E', alpha=0.25, edgecolor='#616161', linestyle=':', lw=1.4, label='Muon 1 $E_{\\mathrm{soft}}$', zorder=1)
    ax1_twin_1.step(edges_sub_1[:-1], prob_sub_1, where='post', color='#424242', linestyle='--', lw=1.6, label='Muon 2 $E_{\\mathrm{sublead}}$', zorder=2)
    ax1_twin_1.step(edges_lead_1[:-1], prob_lead_1, where='post', color='black', linestyle='-', lw=1.8, label='Parent muon $E_{\\mathrm{lead}}$', zorder=3)
    max_prob_1 = max(prob_soft_1.max(), prob_sub_1.max(), prob_lead_1.max())
    ax1_twin_1.set_ylabel('Muon spectra', fontweight='bold', color='#212121', fontsize=11.5, labelpad=8)
    ax1_twin_1.set_ylim(0, max_prob_1 * 1.4)
    ax1_twin_1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1_twin_1.tick_params(axis='y', labelsize=10.5, length=5, width=1.0)
    ax1_twin_1.tick_params(axis='y', which='minor', length=2.5, width=0.8)
    df_rhomip = ddf_root.to_pandas(f'{ROOT_FILE_1}:scifi/gr_mean_rhomip')
    df_rhomip_band = ddf_root.to_pandas(f'{ROOT_FILE_1}:scifi/gr_band_rhomip')
    df_rhomip_med = ddf_root.to_pandas(f'{ROOT_FILE_1}:scifi/gr_median_rhomip')
    ax_2.fill_between(df_rhomip_band['x'], df_rhomip_band['y'] - df_rhomip_band['eyl'], df_rhomip_band['y'] + df_rhomip_band['eyh'], color='#CE93D8', alpha=0.25, label='$\\mu$ particle gun $68\\%$ containment', zorder=4)
    ax_2.plot(df_rhomip_med['x'], df_rhomip_med['y'], 's--', color='#BA68C8', mfc='none', ms=4.0, lw=1.2, label='Median MIP charge density', zorder=5)
    ax_2.errorbar(df_rhomip['x'], df_rhomip['y'], yerr=df_rhomip['ey'], fmt='o-', color='#4A148C', ecolor='#4A148C', lw=2.2, ms=5.0, capsize=2.5, label='Mean charge density $\\langle \\rho_{\\mathrm{mip}} \\rangle$', zorder=6)
    ax_2.axhline(1.0, color='#8D6E63', linestyle='--', lw=1.6, label='MIP ionization standard ($\\rho_{\\mathrm{mip}} = 1.00$)', zorder=3)
    ax_2.set_xscale('log')
    ax_2.set_xlim(9, 6800)
    ax_2.set_ylim(0.45, 1.85)
    ax_2.set_xlabel('Muon energy $E_{\\mu}$ [ GeV ]', fontweight='bold', fontsize=12, labelpad=8)
    ax_2.set_ylabel('$\\rho_{\\mathrm{mip}}$', fontweight='bold', color='#4A148C', fontsize=13, labelpad=8)
    ax_2.grid(True, which='major', linestyle='--', alpha=0.4, color='gray', zorder=0)
    ax_2.minorticks_on()
    ax_2.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray', zorder=0)
    ax_2.tick_params(axis='both', which='major', labelsize=11, length=6, width=1.2)
    ax_2.tick_params(axis='both', which='minor', labelsize=9, length=3, width=0.8)
    ax_2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.96), frameon=True, framealpha=0.92, fontsize=12.0, title_fontsize=9.5)
    ax1_twin_1.legend(loc='upper right', bbox_to_anchor=(0.98, 0.96), frameon=True, framealpha=0.92, fontsize=12.0, title_fontsize=9.5)
    add_label(loc='top-left', supp_loc='top-right', mainText='', extraText='Muon Particle Gun and Trimuon Simulations', suppText='', com=None, lumi=None, ax=ax_2)
    plt.tight_layout()
    plt.show()
    # Mean MIP Charge Density with SEM error bars
    # Axis Configuration
    # Ticks and Grid
    # Legends
    # Header
    plt.close(fig_2)
    return


@app.cell
def _(Path, np, pd):
    import pyarrow.parquet as pq
    MC_PATH_1 = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet')
    DATA_PATH = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/run_006640/sndsw_raw_cutset8_hough.parquet')
    GAL_PATH = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/gallery/mu3_search_hough_cutset8.parquet')
    LUMI_FULL_6640 = 0.9915864253712848
    LUMI_MC_DATASET = 0.025
    SCALE_DATA_TO_FULL = 1.0
    SCALE_MC_TO_FULL = LUMI_FULL_6640 / LUMI_MC_DATASET
    # ==============================================================================
    # 1. PATHS, CONSTANTS & COLUMN DEFINITIONS
    W_BAR_SF = 2.13
    QDC_BAR_SF = 2.48
    LOAD_COLS = ['n_lines_sf_xz', 'n_lines_sf_yz', 'n_lines_ds_xz', 'n_lines_ds_yz', 'xz_sf_m1', 'xz_sf_c1', 'yz_sf_m1', 'yz_sf_c1', 'xz_sf_m2', 'xz_sf_c2', 'yz_sf_m2', 'yz_sf_c2', 'xz_sf_m3', 'xz_sf_c3', 'yz_sf_m3', 'yz_sf_c3', 'xz_ds_m1', 'xz_ds_c1', 'yz_ds_m1', 'yz_ds_c1', 'xz_ds_m2', 'xz_ds_c2', 'yz_ds_m2', 'yz_ds_c2', 'xz_ds_m3', 'xz_ds_c3', 'yz_ds_m3', 'yz_ds_c3', 'scifi_nhits', 'scifi_sum_qdc']

    def check_nc(df: pd.DataFrame, prefix: str, proj: str, z_min: float, z_max: float, strict: bool=True) -> np.ndarray:
        n = df[f'n_lines_{prefix}_{proj}'].to_numpy()
        (m1, c1) = (np.nan_to_num(df[f'{proj}_{prefix}_m1'].to_numpy()), np.nan_to_num(df[f'{proj}_{prefix}_c1'].to_numpy()))
        (m2, c2) = (np.nan_to_num(df[f'{proj}_{prefix}_m2'].to_numpy()), np.nan_to_num(df[f'{proj}_{prefix}_c2'].to_numpy()))
        (m3, c3) = (np.nan_to_num(df[f'{proj}_{prefix}_m3'].to_numpy()), np.nan_to_num(df[f'{proj}_{prefix}_c3'].to_numpy()))
        no_cross_12 = (m1 * z_min + c1 - (m2 * z_min + c2)) * (m1 * z_max + c1 - (m2 * z_max + c2)) > 0
        no_cross_13 = (m1 * z_min + c1 - (m3 * z_min + c3)) * (m1 * z_max + c1 - (m3 * z_max + c3)) > 0
        no_cross_23 = (m2 * z_min + c2 - (m3 * z_min + c3)) * (m2 * z_max + c2 - (m3 * z_max + c3)) > 0  # Full Run 6640 luminosity [fb^-1]
        pass_3 = (n >= 3) & no_cross_12 & no_cross_13 & no_cross_23  # MC sample generated luminosity [fb^-1]
        pass_2 = (n == 2) & no_cross_12
        return pass_3 | pass_2 if strict else pass_3 | pass_2 | (n <= 1)
      # ~39.66
    def apply_selection(df: pd.DataFrame) -> pd.DataFrame:
        (nx, ny) = (df['n_lines_sf_xz'].to_numpy(), df['n_lines_sf_yz'].to_numpy())  # Average hits per plane for 1 MIP
        (nx_ds, ny_ds) = (df['n_lines_ds_xz'].to_numpy(), df['n_lines_ds_yz'].to_numpy())  # Average QDC per plane for 1 MIP
        sf_3_1 = (nx >= 3) & (ny >= 1) | (ny >= 3) & (nx >= 1)
        ds_3_1 = (nx_ds >= 3) & (ny_ds >= 1) | (ny_ds >= 3) & (nx_ds >= 1)
        coll_sf = check_nc(df, 'sf', 'xz', 260.0, 355.0) | check_nc(df, 'sf', 'yz', 260.0, 355.0)
        coll_ds = check_nc(df, 'ds', 'xz', 460.0, 590.0) | check_nc(df, 'ds', 'yz', 460.0, 590.0)
        return df[(sf_3_1 | ds_3_1) & coll_sf & coll_ds].copy()
    mc_raw = pq.read_table(MC_PATH_1, columns=LOAD_COLS + ['is_signal', 'region_type', 'mc_weight']).to_pandas()
    sig_mc_1 = mc_raw[(mc_raw['is_signal'] == 1) & (mc_raw['region_type'] == 1)]
    sel_mc_1 = apply_selection(sig_mc_1)
    data_raw = pq.read_table(DATA_PATH, columns=LOAD_COLS).to_pandas()
    sel_data = apply_selection(data_raw)
    if GAL_PATH.exists():
        gal_raw = pq.read_table(GAL_PATH, columns=LOAD_COLS).to_pandas()
        sel_gal = apply_selection(gal_raw)
    else:
        sel_gal = pd.DataFrame()
    eps_mc_sel = 0.5 * (sel_mc_1['scifi_nhits'].to_numpy() / 10.0 / W_BAR_SF + sel_mc_1['scifi_sum_qdc'].to_numpy() / 10.0 / QDC_BAR_SF)
    eps_data_sel = 0.5 * (sel_data['scifi_nhits'].to_numpy() / 10.0 / W_BAR_SF + sel_data['scifi_sum_qdc'].to_numpy() / 10.0 / QDC_BAR_SF)
    if not sel_gal.empty and 'scifi_nhits' in sel_gal.columns:
        eps_gal_sel = 0.5 * (sel_gal['scifi_nhits'].to_numpy() / 10.0 / W_BAR_SF + sel_gal['scifi_sum_qdc'].to_numpy() / 10.0 / QDC_BAR_SF)
    else:
        eps_gal_sel = np.array([])
    mc_weights_full = sel_mc_1['mc_weight'].to_numpy() * SCALE_MC_TO_FULL
    # 2. SELECTION FUNCTION: [SF(3,1) | DS(3,1)] & (SF_nc & DS_nc)
    # 3. LOAD DATASETS & FILTER
    # 1. MC Rock Trident Signal
    # 2. Collision Data
    # 3. Visual Gallery Candidates
    # 4. COMPUTE EPSILON_MIP
    n_mc_full_expected = mc_weights_full.sum()
    return (pq,)


@app.cell
def _(Path, ROOT, np, pd, pq, stats):
    ROOT.gROOT.SetBatch(True)
    MC_PATH_2 = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet')
    DATA_PATH_1 = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/run_006640/sndsw_raw_cutset8_hough.parquet')
    LUMI_FULL_6640_1 = 0.9915864253712848
    LUMI_MC_DATASET_1 = 0.025
    SCALE_MC_TO_FULL_1 = LUMI_FULL_6640_1 / LUMI_MC_DATASET_1
    LOAD_COLS_1 = ['n_lines_sf_xz', 'n_lines_sf_yz', 'n_lines_ds_xz', 'n_lines_ds_yz', 'xz_sf_m1', 'xz_sf_c1', 'yz_sf_m1', 'yz_sf_c1', 'xz_sf_m2', 'xz_sf_c2', 'yz_sf_m2', 'yz_sf_c2', 'xz_sf_m3', 'xz_sf_c3', 'yz_sf_m3', 'yz_sf_c3', 'xz_ds_m1', 'xz_ds_c1', 'yz_ds_m1', 'yz_ds_c1', 'xz_ds_m2', 'xz_ds_c2', 'yz_ds_m2', 'yz_ds_c2', 'xz_ds_m3', 'xz_ds_c3', 'yz_ds_m3', 'yz_ds_c3', 'scifi_nhits', 'scifi_sum_qdc']

    def check_nc_1(df: pd.DataFrame, prefix: str, proj: str, z_min: float, z_max: float, strict: bool=True) -> np.ndarray:
        n = df[f'n_lines_{prefix}_{proj}'].to_numpy()
        (m1, c1) = (np.nan_to_num(df[f'{proj}_{prefix}_m1'].to_numpy()), np.nan_to_num(df[f'{proj}_{prefix}_c1'].to_numpy()))
        (m2, c2) = (np.nan_to_num(df[f'{proj}_{prefix}_m2'].to_numpy()), np.nan_to_num(df[f'{proj}_{prefix}_c2'].to_numpy()))
    # ==============================================================================
    # 1. FILE PATHS & SELECTION CRITERIA
        (m3, c3) = (np.nan_to_num(df[f'{proj}_{prefix}_m3'].to_numpy()), np.nan_to_num(df[f'{proj}_{prefix}_c3'].to_numpy()))
        (x1_min, x1_max) = (m1 * z_min + c1, m1 * z_max + c1)
        (x2_min, x2_max) = (m2 * z_min + c2, m2 * z_max + c2)
        (x3_min, x3_max) = (m3 * z_min + c3, m3 * z_max + c3)
        no_cross_12 = (x1_min - x2_min) * (x1_max - x2_max) > 0
        no_cross_13 = (x1_min - x3_min) * (x1_max - x3_max) > 0
        no_cross_23 = (x2_min - x3_min) * (x2_max - x3_max) > 0
        pass_3 = (n >= 3) & no_cross_12 & no_cross_13 & no_cross_23
        pass_2 = (n == 2) & no_cross_12  # Target Run 6640 luminosity [fb^-1]
        return pass_3 | pass_2 if strict else pass_3 | pass_2 | (n <= 1)  # MC sample luminosity [fb^-1]

    def apply_selection_1(df: pd.DataFrame) -> pd.DataFrame:
        (nx, ny) = (df['n_lines_sf_xz'].to_numpy(), df['n_lines_sf_yz'].to_numpy())
        (nx_ds, ny_ds) = (df['n_lines_ds_xz'].to_numpy(), df['n_lines_ds_yz'].to_numpy())
        sf_3_1 = (nx >= 3) & (ny >= 1) | (ny >= 3) & (nx >= 1)
        ds_3_1 = (nx_ds >= 3) & (ny_ds >= 1) | (ny_ds >= 3) & (nx_ds >= 1)
        mult_mask = sf_3_1 | ds_3_1
        coll_sf = check_nc_1(df, 'sf', 'xz', 260.0, 355.0) | check_nc_1(df, 'sf', 'yz', 260.0, 355.0)
        coll_ds = check_nc_1(df, 'ds', 'xz', 460.0, 590.0) | check_nc_1(df, 'ds', 'yz', 460.0, 590.0)
        return df[mult_mask & coll_sf & coll_ds].copy()
    print('[*] Loading MC and Collision Data...')
    mc_df_1 = pq.read_table(MC_PATH_2, columns=LOAD_COLS_1 + ['is_signal', 'region_type', 'mc_weight']).to_pandas()
    sig_mc_2 = mc_df_1[(mc_df_1['is_signal'] == 1) & (mc_df_1['region_type'] == 1)]
    bkg_mc = mc_df_1[mc_df_1['is_signal'] == 0]
    sel_sig = apply_selection_1(sig_mc_2)
    sel_bkg = apply_selection_1(bkg_mc)
    data_df = pq.read_table(DATA_PATH_1, columns=LOAD_COLS_1).to_pandas()
    sel_data_1 = apply_selection_1(data_df)
    print(f'    -> Surviving Signal MC: {len(sel_sig):,} | Non-Signal MC: {len(sel_bkg):,} | Data: {len(sel_data_1):,}')
    df_sig = pd.DataFrame({'hits_plane': sel_sig['scifi_nhits'].to_numpy() / 10.0, 'qdc_plane': sel_sig['scifi_sum_qdc'].to_numpy() / 10.0})
    df_bkg = pd.DataFrame({'hits_plane': sel_bkg['scifi_nhits'].to_numpy() / 10.0, 'qdc_plane': sel_bkg['scifi_sum_qdc'].to_numpy() / 10.0})
    df_data = pd.DataFrame({'hits_plane': sel_data_1['scifi_nhits'].to_numpy() / 10.0, 'qdc_plane': sel_data_1['scifi_sum_qdc'].to_numpy() / 10.0})

    def compute_profile(df: pd.DataFrame, bin_width: float=0.5, min_count: int=5):
        df_c = df.copy()
        df_c['hit_bin'] = np.round(df_c['hits_plane'] / bin_width) * bin_width
        prof = df_c.groupby('hit_bin')['qdc_plane'].agg(mean='mean', median='median', q16=lambda x: float(np.percentile(x, 16)), q84=lambda x: float(np.percentile(x, 84)), sem=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 1.0, count='count').reset_index()
        return prof[prof['count'] >= min_count].copy()
    BIN_WIDTH_1 = 0.5
    prof_sig = compute_profile(df_sig, bin_width=BIN_WIDTH_1, min_count=10)
    prof_data = compute_profile(df_data, bin_width=BIN_WIDTH_1, min_count=10)
    (r_pearson_sig, _) = stats.pearsonr(df_sig['hits_plane'], df_sig['qdc_plane'])
    weights_sig = 1.0 / np.where(prof_sig['sem'] > 0, prof_sig['sem'], 1.0)
    (slope_sig, intercept_sig) = np.polyfit(prof_sig['hit_bin'].to_numpy(), prof_sig['mean'].to_numpy(), deg=1, w=weights_sig)
    (r_pearson_data, _) = stats.pearsonr(df_data['hits_plane'], df_data['qdc_plane'])
    weights_data = 1.0 / np.where(prof_data['sem'] > 0, prof_data['sem'], 1.0)
    # 2. LOAD & EXTRACT DATASETS
    # Convert to hits/plane and QDC/plane
    # 3. STATISTICAL CORRELATION & BINNED PROFILES
    # Correlation and weighted linear fits
    (slope_data, intercept_data) = np.polyfit(prof_data['hit_bin'].to_numpy(), prof_data['mean'].to_numpy(), deg=1, w=weights_data)
    return (
        BIN_WIDTH_1,
        df_bkg,
        df_data,
        df_sig,
        intercept_data,
        intercept_sig,
        prof_data,
        prof_sig,
        r_pearson_data,
        r_pearson_sig,
        slope_data,
        slope_sig,
    )


@app.cell
def _(
    BIN_WIDTH_1,
    add_label,
    df_bkg,
    df_data,
    df_sig,
    intercept_data,
    intercept_sig,
    np,
    plt,
    prof_data,
    prof_sig,
    r_pearson_data,
    r_pearson_sig,
    slope_data,
    slope_sig,
):
    (fig_3, ax_3) = plt.subplots(figsize=(8.5, 6.5))
    if not df_bkg.empty:
        ax_3.scatter(df_bkg['hits_plane'], df_bkg['qdc_plane'], color='#E53935', label=f'Non-signal MC ($N = {len(df_bkg):,}$)', alpha=0.8, marker='^', s=18, edgecolors='none', zorder=2)
    # 4a. Scatter distributions
    ax_3.scatter(df_sig['hits_plane'], df_sig['qdc_plane'], color='#1E88E5', label=f'$\\mu \\to 3\\mu$ signal MC ($N = {len(df_sig):,}$)', alpha=0.25, s=20, edgecolors='none', zorder=3)
    ax_3.scatter(df_data['hits_plane'], df_data['qdc_plane'], color='#424242', label=f'Data (Run 6640) ($N = {len(df_data):,}$)', alpha=0.3, s=22, edgecolors='none', zorder=4)
    yerr_asym_sig = [np.maximum(0, (prof_sig['mean'] - prof_sig['q16']).to_numpy()), np.maximum(0, (prof_sig['q84'] - prof_sig['mean']).to_numpy())]
    ax_3.errorbar(prof_sig['hit_bin'], prof_sig['mean'], xerr=BIN_WIDTH_1 / 2.0, yerr=yerr_asym_sig, fmt='o', color='#0D47A1', ecolor='#0D47A1', elinewidth=1.6, capsize=3.0, capthick=1.4, markersize=5.5, zorder=8, label='Signal $\\langle \\mathrm{QDC}^{\\mathrm{SF}} \\rangle$ profile')
    yerr_asym_data = [np.maximum(0, (prof_data['mean'] - prof_data['q16']).to_numpy()), np.maximum(0, (prof_data['q84'] - prof_data['mean']).to_numpy())]
    ax_3.errorbar(prof_data['hit_bin'], prof_data['mean'], xerr=BIN_WIDTH_1 / 2.0, yerr=yerr_asym_data, fmt='s', color='black', ecolor='black', elinewidth=1.6, capsize=3.0, capthick=1.4, markersize=5.0, zorder=9, label='Data $\\langle \\mathrm{QDC}^{\\mathrm{SF}} \\rangle$ profile')
    x_line_1 = np.linspace(0.8, 15.0, 100)
    ax_3.plot(x_line_1, slope_sig * x_line_1 + intercept_sig, color='#0D47A1', linestyle='-', lw=2.2, zorder=6, label=f'Signal fit (slope $= {slope_sig:.2f},\\ r = {r_pearson_sig:.2f}$)')
    ax_3.plot(x_line_1, slope_data * x_line_1 + intercept_data, color='black', linestyle='--', lw=2.0, zorder=7, label=f'Data fit (slope $= {slope_data:.2f},\\ r = {r_pearson_data:.2f}$)')
    ax_3.set_xlim(0.5, 15.5)
    ax_3.set_ylim(0.0, 225.0)
    ax_3.set_xlabel('$\\langle N_{\\mathrm{hits}}^{\\mathrm{SF}} / 10 \\rangle$', fontweight='bold', fontsize=12, labelpad=8)
    ax_3.set_ylabel('$\\langle \\mathrm{QDC}^{\\mathrm{SF}} / 10 \\rangle$', fontweight='bold', fontsize=12, labelpad=8)
    ax_3.grid(True, which='major', linestyle='--', alpha=0.4, color='gray', zorder=1)
    ax_3.minorticks_on()
    ax_3.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray', zorder=1)
    ax_3.tick_params(axis='both', which='major', labelsize=11, length=6, width=1.2)
    ax_3.tick_params(axis='both', which='minor', labelsize=9, length=3, width=0.8)
    ax_3.legend(loc='upper left', frameon=True, framealpha=0.92, edgecolor='#BDBDBD', facecolor='white', fontsize=9.5)
    add_label(loc='top-left', supp_loc='top-right', mainText='', extraText='', suppText='(Run 6640)', lumi=0.992, com=13.6, style='formal', ax=ax_3)
    plt.tight_layout()
    plt.show()
    # 4b. Profile points with 68% containment asymmetric error bars
    # 4c. Profile linear fit curves
    # 4d. Axis styling & limits
    # Ticks & Grid
    # Legend
    # Formal Header
    plt.close(fig_3)
    return


@app.cell
def _(Path, json, uproot):
    # ==============================================================================
    # 0. PATHS & MIP BASELINE
    ROOT_RESP_FILE = '/eos/user/i/idioniso/sndMuTri/out/pGun_response.root'
    with uproot.open(ROOT_RESP_FILE) as f_meta:
        meta_2 = json.loads(str(f_meta['metadata'])) if 'metadata' in f_meta else {}
        qdc_bar_sf_3 = meta_2.get('qdc_bar_SF', 2.576)
    PMU_PATH = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/pmu/sndLHC.Ntuple-TGeant4-160urad_100e6pp_FlukaEcut10_digCPP_hough.parquet')
    TRI_PATH = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet')
    DATA_PATH_2 = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/run_006640/sndsw_raw_cutset8_hough.parquet')
    LUMI_FULL_6640_2 = 0.9915864253712848
    return DATA_PATH_2, PMU_PATH, TRI_PATH, qdc_bar_sf_3


@app.cell
def _(DATA_PATH_2, PMU_PATH, TRI_PATH, np, pq, qdc_bar_sf_3):
    # ==============================================================================
    # 1. EXTRACT SINGLE PASSING MUONS & COMPUTE MIP-CORRECTED QDC/PLANE
    print('[*] Extracting single passing muons across Data, pmu MC, and Trimuon MC...')
    pmu = pq.read_table(PMU_PATH, columns=['n_lines_sf_xz', 'n_lines_sf_yz', 'scifi_sum_qdc', 'mc_weight']).to_pandas()
    pmu_single = pmu[(pmu['n_lines_sf_xz'] == 1) & (pmu['n_lines_sf_yz'] == 1)]
    # 1. pmu MC (FLUKA Passing Muons)
    qdc_pmu_norm = pmu_single['scifi_sum_qdc'].to_numpy() / 10.0 / qdc_bar_sf_3
    w_pmu = pmu_single['mc_weight'].to_numpy()
    tri = pq.read_table(TRI_PATH, columns=['n_lines_sf_xz', 'n_lines_sf_yz', 'scifi_sum_qdc', 'mc_weight', 'is_signal']).to_pandas()
    tri_single = tri[(tri['n_lines_sf_xz'] == 1) & (tri['n_lines_sf_yz'] == 1)]
    qdc_tri_norm = tri_single['scifi_sum_qdc'].to_numpy() / 10.0 / qdc_bar_sf_3
    w_tri = tri_single['mc_weight'].to_numpy()
    data = pq.read_table(DATA_PATH_2, columns=['n_lines_sf_xz', 'n_lines_sf_yz', 'scifi_sum_qdc']).to_pandas()
    data_single = data[(data['n_lines_sf_xz'] == 1) & (data['n_lines_sf_yz'] == 1)]
    qdc_data_norm = data_single['scifi_sum_qdc'].to_numpy() / 10.0 / qdc_bar_sf_3
    print(f'  -> Data (Run 6640) : N = {len(qdc_data_norm):,} single muons | Median = {np.median(qdc_data_norm):.3f} MIPs')
    print(f'  -> Fluka PMU MC    : N = {len(qdc_pmu_norm):,} single muons | Median = {np.median(qdc_pmu_norm):.3f} MIPs')
    # 2. Trimuon MC (Single-track passing muons)
    # 3. Collision Data (Run 6640 Single Passing Muons)
    print(f'  -> Trimuon MC      : N = {len(qdc_tri_norm):,} single muons | Median = {np.median(qdc_tri_norm):.3f} MIPs')
    return


@app.cell
def _(Path, add_label, np, pd, plt):
    OUT_DIR = Path('/eos/user/i/idioniso/sndMuTri/out/single_muon_calibration')
    df_data_1 = pd.read_parquet(OUT_DIR / 'data_run6640_pure_single_muons.parquet')
    df_pmu = pd.read_parquet(OUT_DIR / 'pmu_fluka_truth_single_muons.parquet')
    df_tri = pd.read_parquet(OUT_DIR / 'trimuon_mc_truth_single_muons.parquet')
    qdc_data_raw = df_data_1['qdc_plane'].to_numpy()
    qdc_pmu_raw = df_pmu['qdc_plane'].to_numpy()
    w_pmu_1 = df_pmu['mc_weight'].to_numpy()
    # ==============================================================================
    # 0. LOAD EXTRACTED PARQUET DATASETS (PURE QDC / PLANE)
    qdc_tri_raw = df_tri['qdc_plane'].to_numpy()
    w_tri_1 = df_tri['mc_weight'].to_numpy()
    print('[*] Loaded Pure QDC Datasets:')
    print(f'    1. Data (Run 6640) : N = {len(qdc_data_raw):,} | Median = {np.median(qdc_data_raw):.2f} QDC')
    print(f'    2. FLUKA pmu MC    : N = {len(qdc_pmu_raw):,} | Median = {np.median(qdc_pmu_raw):.2f} QDC')
    print(f'    3. Trimuon MC Truth: N = {len(qdc_tri_raw):,} | Median = {np.median(qdc_tri_raw):.2f} QDC')

    # Use raw QDC per plane directly (not dividing by qdc_bar_sf)
    def get_weighted_hist_with_errors(values, weights=None, bins=None, density=True):
        if weights is None:
            weights = np.ones_like(values, dtype=np.float64)
        (sum_w, edges) = np.histogram(values, bins=bins, weights=weights)
        (sum_w2, _) = np.histogram(values, bins=bins, weights=weights ** 2)
        bin_widths = np.diff(edges)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        if density:
            total_area = np.sum(sum_w * bin_widths)
            y = sum_w / total_area if total_area > 0 else sum_w
            y_err = np.sqrt(sum_w2) / total_area if total_area > 0 else np.sqrt(sum_w2)
        else:
            total_sum = np.sum(sum_w)
            y = sum_w / total_sum if total_sum > 0 else sum_w
            y_err = np.sqrt(sum_w2) / total_sum if total_sum > 0 else np.sqrt(sum_w2)
        return (bin_centers, edges, y, y_err)
    x_max = 120.0
    bin_width = 1.0
    bins = np.arange(0.0, x_max + bin_width, bin_width)
    (bc_data, edges, y_data, err_data) = get_weighted_hist_with_errors(qdc_data_raw, weights=None, bins=bins, density=True)
    (bc_pmu, _, y_pmu, err_pmu) = get_weighted_hist_with_errors(qdc_pmu_raw, weights=w_pmu_1, bins=bins, density=True)
    (bc_tri, _, y_tri, err_tri) = get_weighted_hist_with_errors(qdc_tri_raw, weights=w_tri_1, bins=bins, density=True)
    # 1. HELPER: WEIGHTED HISTOGRAM WITH EXACT sqrt(sum(w^2)) ERRORS
    (fig_4, ax_4) = plt.subplots(figsize=(9.0, 6.2))

    def plot_hep_step_with_errorbars(ax, edges, bin_centers, y, y_err, color, label, linestyle='-', lw=2.2, zorder=3, capsize=2.0):
        ax.step(edges, np.append(y, y[-1]), where='post', color=color, linestyle=linestyle, linewidth=lw, label=label, zorder=zorder + 1)
        mask = y > 0
        ax.errorbar(bin_centers[mask], y[mask], yerr=y_err[mask], fmt='none', ecolor=color, elinewidth=1.3, capsize=capsize, capthick=1.1, zorder=zorder + 2)
    plot_hep_step_with_errorbars(ax_4, edges, bc_pmu, y_pmu, err_pmu, color='#1B5E20', label=f'Fluka PMU MC ($N = {len(qdc_pmu_raw):,}$, med = {np.median(qdc_pmu_raw):.1f})', linestyle='-', lw=2.2, zorder=3, capsize=2.0)
    plot_hep_step_with_errorbars(ax_4, edges, bc_tri, y_tri, err_tri, color='#0D47A1', label=f'Trimuon MC 1-track ($N = {len(qdc_tri_raw):,}$, med = {np.median(qdc_tri_raw):.1f})', linestyle='--', lw=2.2, zorder=4, capsize=2.0)
    plot_hep_step_with_errorbars(ax_4, edges, bc_data, y_data, err_data, color='#111111', label=f'Data (Run 6640 Tracks) ($N = {len(qdc_data_raw):,}$, med = {np.median(qdc_data_raw):.1f})', linestyle='-', lw=2.4, zorder=5, capsize=2.5)
    ax_4.set_xlim(0.0, x_max)
    y_max_val = max(y_data.max() if len(y_data) > 0 else 0, 0.15)
    ax_4.set_ylim(bottom=0.001, top=y_max_val * 1.3)
    ax_4.set_yscale('log')
    ax_4.set_xlabel('$\\langle \\mathrm{QDC}^{\\mathrm{SF}} / 10 \\rangle$', fontweight='bold', fontsize=13, labelpad=8)
    ax_4.set_ylabel('Normalized entries', fontweight='bold', fontsize=13, labelpad=8)
    ax_4.tick_params(axis='both', which='major', direction='in', top=True, right=True, labelsize=11.5, length=6, width=1.2)
    ax_4.tick_params(axis='both', which='minor', direction='in', top=True, right=True, labelsize=9.5, length=3, width=0.8)
    ax_4.legend(loc='upper right', frameon=True, framealpha=0.94, edgecolor='#BDBDBD', facecolor='white', fontsize=12.0)
    add_label(loc='top-left', supp_loc='top-right', mainText='', extraText='', suppText='(Run 6640)', lumi=0.992, com=13.6, style='formal', ax=ax_4)
    plt.tight_layout()
    plt.show()
    # 2. COMPUTE HISTOGRAMS IN RAW QDC UNITS
    # Scaled by ~2.576 compared to MIP units (0-50 MIPs -> 0-120 QDC)
    # 3. PLOTTING
    # 2. Trimuon MC
    # 4. FORMATTING, TICKS & LABELS
    plt.close(fig_4)  # 1 QDC / bin
    return


@app.cell
def _(Path, add_label, json, np, pd, plt, pq, stats, ticker, uproot):
    ROOT_RESP_FILE_1 = '/eos/user/i/idioniso/sndMuTri/out/pGun_response.root'
    with uproot.open(ROOT_RESP_FILE_1) as f_meta_1:
        meta_3 = json.loads(str(f_meta_1['metadata'])) if 'metadata' in f_meta_1 else {}
        w_bar_sf_3 = meta_3.get('w_bar_SF', 2.183)
        qdc_bar_sf_4 = meta_3.get('qdc_bar_SF', 2.576)
    MC_PATH_3 = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet')
    DATA_PATH_3 = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/run_006640/sndsw_raw_cutset8_hough.parquet')
    LOAD_COLS_2 = ['scifi_nhits', 'scifi_sum_qdc']
    mc_df_2 = pq.read_table(MC_PATH_3, columns=LOAD_COLS_2 + ['is_signal', 'region_type']).to_pandas()
    sig_mc_3 = mc_df_2[(mc_df_2['is_signal'] == 1) & (mc_df_2['region_type'] == 1)]
    data_df_1 = pq.read_table(DATA_PATH_3, columns=LOAD_COLS_2).to_pandas()
    # ==============================================================================
    # 0. LOAD MIP BASELINES & DATASETS
    df_sig_1 = pd.DataFrame({'eps_mip': sig_mc_3['scifi_nhits'].to_numpy() / 10.0 / w_bar_sf_3, 'rho_mip': sig_mc_3['scifi_sum_qdc'].to_numpy() / 10.0 / qdc_bar_sf_4})
    df_data_2 = pd.DataFrame({'eps_mip': data_df_1['scifi_nhits'].to_numpy() / 10.0 / w_bar_sf_3, 'rho_mip': data_df_1['scifi_sum_qdc'].to_numpy() / 10.0 / qdc_bar_sf_4})

    def compute_profile_1(df: pd.DataFrame, bin_width: float=0.25, min_count: int=20):
        df_c = df.copy()
        df_c['hit_bin'] = np.round(df_c['eps_mip'] / bin_width) * bin_width
        prof = df_c.groupby('hit_bin')['rho_mip'].agg(mean='mean', median='median', q16=lambda x: float(np.percentile(x, 16)), q84=lambda x: float(np.percentile(x, 84)), sem=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 1.0, count='count').reset_index()
        return prof[prof['count'] >= min_count].copy()
    BIN_WIDTH_2 = 0.25
    prof_sig_1 = compute_profile_1(df_sig_1, bin_width=BIN_WIDTH_2, min_count=20)
    prof_data_1 = compute_profile_1(df_data_2, bin_width=BIN_WIDTH_2, min_count=20)
    weights_sig_1 = 1.0 / np.where(prof_sig_1['sem'] > 0, prof_sig_1['sem'], 1.0)
    (slope_sig_1, intercept_sig_1) = np.polyfit(prof_sig_1['hit_bin'].to_numpy(), prof_sig_1['mean'].to_numpy(), deg=1, w=weights_sig_1)
    (r_sig, _) = stats.pearsonr(df_sig_1['eps_mip'], df_sig_1['rho_mip'])
    weights_data_1 = 1.0 / np.where(prof_data_1['sem'] > 0, prof_data_1['sem'], 1.0)
    (slope_data_1, intercept_data_1) = np.polyfit(prof_data_1['hit_bin'].to_numpy(), prof_data_1['mean'].to_numpy(), deg=1, w=weights_data_1)
    (r_data, _) = stats.pearsonr(df_data_2['eps_mip'], df_data_2['rho_mip'])
    (fig_5, ax_5) = plt.subplots(figsize=(8.5, 6.0))
    n_sub = 1500
    if len(df_sig_1) > n_sub:
        sub_sig = df_sig_1.sample(n=n_sub, random_state=42)
        ax_5.scatter(sub_sig['eps_mip'], sub_sig['rho_mip'], color='#1E88E5', alpha=0.08, s=12, edgecolors='none', zorder=2)
    if len(df_data_2) > n_sub:
        sub_data = df_data_2.sample(n=n_sub, random_state=42)
        ax_5.scatter(sub_data['eps_mip'], sub_data['rho_mip'], color='#757575', alpha=0.06, s=12, edgecolors='none', zorder=2)
    ax_5.fill_between(prof_sig_1['hit_bin'], prof_sig_1['q16'], prof_sig_1['q84'], color='#1E88E5', alpha=0.18, label='Signal $68\\%$ containment', zorder=3)
    ax_5.fill_between(prof_data_1['hit_bin'], prof_data_1['q16'], prof_data_1['q84'], color='black', alpha=0.14, label='Data $68\\%$ containment', zorder=3)
    ax_5.errorbar(prof_sig_1['hit_bin'], prof_sig_1['mean'], yerr=prof_sig_1['sem'], fmt='o', color='#0D47A1', ecolor='#0D47A1', elinewidth=1.4, capsize=2.0, markersize=5.0, zorder=6, label=f'Signal $\\langle \\rho \\rangle$ (slope $= {slope_sig_1:.2f}$)')
    ax_5.errorbar(prof_data_1['hit_bin'], prof_data_1['mean'], yerr=prof_data_1['sem'], fmt='s', color='black', ecolor='black', elinewidth=1.4, capsize=2.0, markersize=4.5, zorder=6, label=f'Data $\\langle \\rho \\rangle$ (slope $= {slope_data_1:.2f}$)')
    x_line_2 = np.linspace(0.4, 6.5, 100)
    ax_5.plot(x_line_2, slope_sig_1 * x_line_2 + intercept_sig_1, color='#0D47A1', lw=2.0, linestyle='-', zorder=5)
    ax_5.plot(x_line_2, slope_data_1 * x_line_2 + intercept_data_1, color='black', lw=2.0, linestyle='--', zorder=5)
    ax_5.plot(x_line_2, x_line_2, color='#D32F2F', linestyle=':', lw=1.5, label='Ideal $1\\text{-to-}1$ line ($\\rho = \\varepsilon$)', zorder=4)
    # 1. BINNED PROFILES & LINEAR REGRESSION
    ax_5.set_xlim(0.3, 6.5)
    ax_5.set_ylim(0.0, 42.0)
    ax_5.set_xlabel('Normalized SciFi Hits per Plane $\\varepsilon_{\\mathrm{mip}}$', fontweight='bold', fontsize=12, labelpad=8)
    ax_5.set_ylabel('Normalized SciFi QDC per Plane $\\rho_{\\mathrm{mip}}$', fontweight='bold', fontsize=12, labelpad=8)
    ax_5.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_5.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_5.grid(True, which='major', linestyle='--', alpha=0.35, color='gray', zorder=1)
    ax_5.grid(True, which='minor', linestyle=':', alpha=0.15, color='gray', zorder=1)
    ax_5.tick_params(axis='both', which='major', direction='in', top=True, right=True, labelsize=11, length=6, width=1.1)
    ax_5.tick_params(axis='both', which='minor', direction='in', top=True, right=True, labelsize=9, length=3, width=0.8)
    ax_5.legend(loc='upper left', frameon=True, framealpha=0.92, edgecolor='#BDBDBD', facecolor='white', fontsize=9.5)
    add_label(loc='top-left', supp_loc='top-right', mainText='', extraText='', suppText='(Run 6640)', lumi=0.992, com=13.6, style='formal', ax=ax_5)
    plt.tight_layout()
    plt.show()
    # Linear fits
    # 2. PLOTTING (STREAMLINED VIEW)
    # 1. Subtle, lightweight scatter background (subsample to avoid heavy blocks)
    # 2. Shaded 68% Containment Bands
    # 3. Profile Mean Lines & Error Points
    # 4. Fitted Trend Lines
    # 5. Ideal 1-to-1 Line
    # Limits & Concise Labels
    # Grid & Inset Ticks
    # Compact Legend
    # Formal Header
    plt.close(fig_5)
    return


@app.cell
def _(Path, add_label, json, np, pd, plt, pq, ticker, uproot):
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter
    ROOT_RESP_FILE_2 = '/eos/user/i/idioniso/sndMuTri/out/pGun_response.root'
    with uproot.open(ROOT_RESP_FILE_2) as f_meta_2:
        meta_4 = json.loads(str(f_meta_2['metadata'])) if 'metadata' in f_meta_2 else {}
        w_bar_sf_4 = meta_4.get('w_bar_SF', 2.183)
        qdc_bar_sf_5 = meta_4.get('qdc_bar_SF', 2.576)
    MC_PATH_4 = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet')
    DATA_PATH_4 = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/run_006640/sndsw_raw_cutset8_hough.parquet')
    LOAD_COLS_3 = ['scifi_nhits', 'scifi_sum_qdc']
    mc_df_3 = pq.read_table(MC_PATH_4, columns=LOAD_COLS_3 + ['is_signal', 'region_type']).to_pandas()
    sig_mc_4 = mc_df_3[(mc_df_3['is_signal'] == 1) & (mc_df_3['region_type'] == 1)]
    bkg_mc_1 = mc_df_3[mc_df_3['is_signal'] == 0]
    # ==============================================================================
    # 0. LOAD MIP BASELINES & DATASETS
    data_df_2 = pq.read_table(DATA_PATH_4, columns=LOAD_COLS_3).to_pandas()
    df_sig_2 = pd.DataFrame({'eps_mip': sig_mc_4['scifi_nhits'].to_numpy() / 10.0 / w_bar_sf_4, 'rho_mip': sig_mc_4['scifi_sum_qdc'].to_numpy() / 10.0 / qdc_bar_sf_5})
    df_bkg_1 = pd.DataFrame({'eps_mip': bkg_mc_1['scifi_nhits'].to_numpy() / 10.0 / w_bar_sf_4, 'rho_mip': bkg_mc_1['scifi_sum_qdc'].to_numpy() / 10.0 / qdc_bar_sf_5})
    df_data_3 = pd.DataFrame({'eps_mip': data_df_2['scifi_nhits'].to_numpy() / 10.0 / w_bar_sf_4, 'rho_mip': data_df_2['scifi_sum_qdc'].to_numpy() / 10.0 / qdc_bar_sf_5})
    print(f'[*] Loaded: Signal MC = {len(df_sig_2):,} | Non-signal MC = {len(df_bkg_1):,} | Data = {len(df_data_3):,}')

    def compute_profile_2(df: pd.DataFrame, bin_width: float=0.25, min_count: int=15) -> pd.DataFrame:
        df_c = df.copy()
        df_c['hit_bin'] = np.round(df_c['eps_mip'] / bin_width) * bin_width
        prof = df_c.groupby('hit_bin')['rho_mip'].agg(mean='mean', sem=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 1.0, count='count').reset_index()
        return prof[prof['count'] >= min_count].copy()
    BIN_WIDTH_3 = 0.25
    prof_sig_2 = compute_profile_2(df_sig_2, bin_width=BIN_WIDTH_3, min_count=15)
    prof_bkg = compute_profile_2(df_bkg_1, bin_width=BIN_WIDTH_3, min_count=15)
    prof_data_2 = compute_profile_2(df_data_3, bin_width=BIN_WIDTH_3, min_count=15)
    weights_sig_2 = 1.0 / np.where(prof_sig_2['sem'] > 0, prof_sig_2['sem'], 1.0)
    # Load MC (Signal vs Non-signal) and Collision Data
    (slope_sig_2, intercept_sig_2) = np.polyfit(prof_sig_2['hit_bin'].to_numpy(), prof_sig_2['mean'].to_numpy(), deg=1, w=weights_sig_2)
    weights_bkg = 1.0 / np.where(prof_bkg['sem'] > 0, prof_bkg['sem'], 1.0)
    (slope_bkg, intercept_bkg) = np.polyfit(prof_bkg['hit_bin'].to_numpy(), prof_bkg['mean'].to_numpy(), deg=1, w=weights_bkg)
    weights_data_2 = 1.0 / np.where(prof_data_2['sem'] > 0, prof_data_2['sem'], 1.0)
    (slope_data_2, intercept_data_2) = np.polyfit(prof_data_2['hit_bin'].to_numpy(), prof_data_2['mean'].to_numpy(), deg=1, w=weights_data_2)

    def compute_density_grid(x, y, x_bins, y_bins, smooth_sigma=1.2):
        """Computes smoothed 2D histogram and returns percentile thresholds for containment."""
    # Normalized variables: epsilon_mip (x) and rho_mip (y)
        (h, x_edges, y_edges) = np.histogram2d(x, y, bins=[x_bins, y_bins])
        h_smooth = gaussian_filter(h.T, sigma=smooth_sigma)
        h_flat = np.sort(h_smooth.flatten())[::-1]
        cum_sum = np.cumsum(h_flat)
        total = cum_sum[-1]
        levels = []
        for frac in [0.5, 0.8, 0.95]:
            idx = np.searchsorted(cum_sum, frac * total)
            levels.append(h_flat[min(idx, len(h_flat) - 1)])
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        return (x_centers, y_centers, h_smooth, sorted(levels))
    (fig_6, ax_6) = plt.subplots(figsize=(9.2, 6.5))
    x_bins = np.linspace(0.3, 6.5, 90)
    y_bins = np.linspace(0.0, 42.0, 90)
    (x_c_data, y_c_data, h_data, levels_data) = compute_density_grid(df_data_3['eps_mip'], df_data_3['rho_mip'], x_bins, y_bins)
    ax_6.contour(x_c_data, y_c_data, h_data, levels=levels_data, colors=['#BDBDBD', '#757575', '#212121'], linewidths=[1.0, 1.3, 1.6], linestyles=['dotted', 'dashed', 'solid'], zorder=2)
    if not df_bkg_1.empty:
        (x_c_bkg, y_c_bkg, h_bkg, levels_bkg) = compute_density_grid(df_bkg_1['eps_mip'], df_bkg_1['rho_mip'], x_bins, y_bins)
        ax_6.contour(x_c_bkg, y_c_bkg, h_bkg, levels=levels_bkg, colors=['#FFCDD2', '#E57373', '#C62828'], linewidths=[1.0, 1.3, 1.6], linestyles=['dotted', 'dashed', 'solid'], zorder=3)
    # 1. BINNED PROFILES & LINEAR REGRESSION
    (x_c_sig, y_c_sig, h_sig, levels_sig) = compute_density_grid(df_sig_2['eps_mip'], df_sig_2['rho_mip'], x_bins, y_bins)
    ax_6.contour(x_c_sig, y_c_sig, h_sig, levels=levels_sig, colors=['#90CAF9', '#2196F3', '#0D47A1'], linewidths=[1.0, 1.3, 1.6], linestyles=['dotted', 'dashed', 'solid'], zorder=4)
    ax_6.errorbar(prof_sig_2['hit_bin'], prof_sig_2['mean'], yerr=prof_sig_2['sem'], fmt='o', color='#0D47A1', ecolor='#0D47A1', elinewidth=1.4, capsize=2.0, markersize=5.0, zorder=7)
    if not prof_bkg.empty:
        ax_6.errorbar(prof_bkg['hit_bin'], prof_bkg['mean'], yerr=prof_bkg['sem'], fmt='^', color='#C62828', ecolor='#C62828', elinewidth=1.4, capsize=2.0, markersize=4.5, zorder=7)
    ax_6.errorbar(prof_data_2['hit_bin'], prof_data_2['mean'], yerr=prof_data_2['sem'], fmt='s', color='black', ecolor='black', elinewidth=1.4, capsize=2.0, markersize=4.5, zorder=7)
    x_line_3 = np.linspace(0.4, 6.5, 100)
    ax_6.plot(x_line_3, slope_sig_2 * x_line_3 + intercept_sig_2, color='#0D47A1', lw=2.2, linestyle='-', zorder=6)
    if not prof_bkg.empty:
        ax_6.plot(x_line_3, slope_bkg * x_line_3 + intercept_bkg, color='#C62828', lw=2.0, linestyle='-.', zorder=6)
    ax_6.plot(x_line_3, slope_data_2 * x_line_3 + intercept_data_2, color='black', lw=2.0, linestyle='--', zorder=6)
    ax_6.plot(x_line_3, x_line_3, color='#9E9E9E', linestyle=':', lw=1.5, zorder=1)
    legend_elements = [Line2D([0], [0], color='#0D47A1', lw=2.2, marker='o', markersize=5.0, label=f'$\\mu \\to 3\\mu$ Signal MC (slope $= {slope_sig_2:.2f}$)'), Line2D([0], [0], color='#2196F3', lw=1.4, linestyle='--', label='Signal $50\\%, 80\\%, 95\\%$ contours'), Line2D([0], [0], color='#C62828', lw=2.0, linestyle='-.', marker='^', markersize=4.5, label=f'Non-signal MC (slope $= {slope_bkg:.2f}$)'), Line2D([0], [0], color='#E57373', lw=1.4, linestyle='--', label='Non-signal $50\\%, 80\\%, 95\\%$ contours'), Line2D([0], [0], color='black', lw=2.0, linestyle='--', marker='s', markersize=4.5, label=f'Data Run 6640 (slope $= {slope_data_2:.2f}$)'), Line2D([0], [0], color='#757575', lw=1.4, linestyle='--', label='Data $50\\%, 80\\%, 95\\%$ contours'), Line2D([0], [0], color='#9E9E9E', lw=1.5, linestyle=':', label='Ideal $1\\text{-to-}1$ line ($\\rho = \\varepsilon$)')]
    ax_6.set_xlim(0.3, 6.5)
    ax_6.set_ylim(0.0, 42.0)
    ax_6.set_xlabel('$\\mathbf{\\varepsilon_{\\mathrm{mip}} = \\langle N_{\\mathrm{hits}}^{\\mathrm{SF}} / 10 \\rangle \\,/\\, \\overline{w}_{\\mathrm{SF}}}$ [ Hit-MIP Equivalents ]', fontweight='bold', fontsize=12, labelpad=8)
    ax_6.set_ylabel('$\\mathbf{\\rho_{\\mathrm{mip}} = \\langle \\mathrm{QDC}^{\\mathrm{SF}} / 10 \\rangle \\,/\\, \\overline{\\mathrm{QDC}}_{\\mathrm{SF}}}$ [ QDC-MIP Equivalents ]', fontweight='bold', fontsize=12, labelpad=8)
    ax_6.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_6.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_6.grid(True, which='major', linestyle='--', alpha=0.35, color='gray', zorder=1)
    ax_6.grid(True, which='minor', linestyle=':', alpha=0.15, color='gray', zorder=1)
    ax_6.tick_params(axis='both', which='major', direction='in', top=True, right=True, labelsize=11, length=6, width=1.1)
    ax_6.tick_params(axis='both', which='minor', direction='in', top=True, right=True, labelsize=9, length=3, width=0.8)
    ax_6.legend(handles=legend_elements, loc='upper left', frameon=True, framealpha=0.92, edgecolor='#BDBDBD', facecolor='white', fontsize=9.0)
    add_label(loc='top-left', supp_loc='top-right', mainText='', extraText='', suppText='(Run 6640)', lumi=0.992, com=13.6, style='formal', ax=ax_6)
    # 1. Linear fit: Signal MC
    plt.tight_layout()
    plt.show()
    # 2. Linear fit: Non-signal MC
    # 3. Linear fit: Collision Data
    # 2. 2D DENSITY CONTOUR HELPER (CONTAINMENT LEVELS)
    # 3. PLOTTING
    # 1. Collision Data 2D Density Contours (Gray/Charcoal)
    # 2. Non-Signal MC 2D Density Contours (Red / Coral)
    # 3. Signal MC 2D Density Contours (Blue / Cobalt)
    # 4. Profile Error Points
    # 5. Fitted Trend Lines
    # 6. Ideal 1-to-1 Line
    # 7. Custom Legend Proxies
    # Limits & Labels
    # Grid & Ticks
    # Legend
    # Publication Header
    plt.close(fig_6)  # Calculate thresholds containing specified cumulative probability (50%, 80%, 95%)
    return (Line2D,)


@app.cell
def _(np, pd):
    def compute_profile_3(df: pd.DataFrame, bin_width: float=0.25, min_count: int=15) -> pd.DataFrame:
        df_c = df.copy()
        df_c['hit_bin'] = np.round(df_c['eps_mip'] / bin_width) * bin_width
        prof = df_c.groupby('hit_bin')['rho_mip'].agg(mean='mean', sem=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 1.0, std='std', q16=lambda x: float(np.percentile(x, 16)), q84=lambda x: float(np.percentile(x, 84)), count='count').reset_index()
        return prof[prof['count'] >= min_count].copy()
    return


@app.cell
def _(Line2D, Path, json, np, pd, plt, pq, ticker, uproot):
    ROOT_RESP_FILE_3 = '/eos/user/i/idioniso/sndMuTri/out/pGun_response.root'
    with uproot.open(ROOT_RESP_FILE_3) as f_meta_3:
        meta_5 = json.loads(str(f_meta_3['metadata'])) if 'metadata' in f_meta_3 else {}
        w_bar_sf_5 = meta_5.get('w_bar_SF', 2.183)
        qdc_bar_sf_6 = meta_5.get('qdc_bar_SF', 2.576)
    MC_PATH_5 = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/trimuon_boost100/trimuon_digCPP_hough_cutset8.parquet')
    DATA_PATH_5 = Path('/eos/user/i/idioniso/sndMuTri/data/cutset8/run_006640/sndsw_raw_cutset8_hough.parquet')
    LOAD_COLS_4 = ['scifi_nhits', 'scifi_sum_qdc']
    mc_df_4 = pq.read_table(MC_PATH_5, columns=LOAD_COLS_4 + ['is_signal', 'region_type']).to_pandas()
    sig_mc_5 = mc_df_4[(mc_df_4['is_signal'] == 1) & (mc_df_4['region_type'] == 1)]
    bkg_mc_2 = mc_df_4[mc_df_4['is_signal'] == 0]
    data_df_3 = pq.read_table(DATA_PATH_5, columns=LOAD_COLS_4).to_pandas()
    # ==============================================================================
    # 0. LOAD MIP BASELINES & DATASETS
    df_sig_3 = pd.DataFrame({'eps_mip': sig_mc_5['scifi_nhits'].to_numpy() / 10.0 / w_bar_sf_5, 'rho_mip': sig_mc_5['scifi_sum_qdc'].to_numpy() / 10.0 / qdc_bar_sf_6})
    df_bkg_2 = pd.DataFrame({'eps_mip': bkg_mc_2['scifi_nhits'].to_numpy() / 10.0 / w_bar_sf_5, 'rho_mip': bkg_mc_2['scifi_sum_qdc'].to_numpy() / 10.0 / qdc_bar_sf_6})
    df_data_4 = pd.DataFrame({'eps_mip': data_df_3['scifi_nhits'].to_numpy() / 10.0 / w_bar_sf_5, 'rho_mip': data_df_3['scifi_sum_qdc'].to_numpy() / 10.0 / qdc_bar_sf_6})

    def compute_profile_4(df: pd.DataFrame, bin_width: float, min_count: int=15) -> pd.DataFrame:
        df_c = df.copy()
        df_c['hit_bin'] = np.round(df_c['eps_mip'] / bin_width) * bin_width
        prof = df_c.groupby('hit_bin')['rho_mip'].agg(mean='mean', sem=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 1.0, q16=lambda x: float(np.percentile(x, 16)), q84=lambda x: float(np.percentile(x, 84)), count='count').reset_index()
        return prof[prof['count'] >= min_count].copy()
    prof_sig_fine = compute_profile_4(df_sig_3, bin_width=0.25, min_count=10)
    prof_bkg_fine = compute_profile_4(df_bkg_2, bin_width=0.25, min_count=10)
    prof_data_fine = compute_profile_4(df_data_4, bin_width=0.25, min_count=10)

    def fit_linear_window(prof_fine: pd.DataFrame, min_x=0.5, max_x=5.0):
        mask = (prof_fine['hit_bin'] >= min_x) & (prof_fine['hit_bin'] <= max_x)
        sub = prof_fine[mask]
        weights = 1.0 / np.where(sub['sem'] > 0, sub['sem'], 1.0)
        (slope, intercept) = np.polyfit(sub['hit_bin'].to_numpy(), sub['mean'].to_numpy(), deg=1, w=weights)
        return (slope, intercept)
    (slope_sig_3, intercept_sig_3) = fit_linear_window(prof_sig_fine, 0.5, 5.0)
    (slope_bkg_1, intercept_bkg_1) = fit_linear_window(prof_bkg_fine, 0.5, 5.0)
    (slope_data_3, intercept_data_3) = fit_linear_window(prof_data_fine, 0.5, 5.0)
    max_eps = max(df_data_4['eps_mip'].max() if not df_data_4.empty else 0, df_bkg_2['eps_mip'].max() if not df_bkg_2.empty else 0, df_sig_3['eps_mip'].max() if not df_sig_3.empty else 0)
    max_rho = max(df_data_4['rho_mip'].max() if not df_data_4.empty else 0, df_bkg_2['rho_mip'].max() if not df_bkg_2.empty else 0, df_sig_3['rho_mip'].max() if not df_sig_3.empty else 0)
    (X_MIN, X_MAX) = (0.0, float(np.ceil(max_eps * 1.05 / 10.0) * 10.0))
    (Y_MIN, Y_MAX) = (0.0, float(np.ceil(max_rho * 1.05 / 100.0) * 100.0))
    COARSE_BIN_WIDTH = 2.0
    prof_sig_disp = compute_profile_4(df_sig_3, bin_width=COARSE_BIN_WIDTH, min_count=15)
    prof_bkg_disp = compute_profile_4(df_bkg_2, bin_width=COARSE_BIN_WIDTH, min_count=15)
    prof_data_disp = compute_profile_4(df_data_4, bin_width=COARSE_BIN_WIDTH, min_count=15)
    (fig_7, ax_7) = plt.subplots(figsize=(12, 5))
    n_sample_scatter = 35000
    if not df_bkg_2.empty:
        sub_bkg = df_bkg_2.sample(n=n_sample_scatter, random_state=42) if len(df_bkg_2) > n_sample_scatter else df_bkg_2
        ax_7.scatter(sub_bkg['eps_mip'], sub_bkg['rho_mip'], color='#E53935', alpha=0.08, s=12, edgecolors='none', rasterized=True, zorder=2)
    sub_sig_1 = df_sig_3.sample(n=n_sample_scatter, random_state=42) if len(df_sig_3) > n_sample_scatter else df_sig_3
    ax_7.scatter(sub_sig_1['eps_mip'], sub_sig_1['rho_mip'], color='#1E88E5', alpha=0.1, s=12, edgecolors='none', rasterized=True, zorder=2)
    sub_data_1 = df_data_4.sample(n=n_sample_scatter, random_state=42) if len(df_data_4) > n_sample_scatter else df_data_4
    # 1. PROFILE EXTRACTION & DEDICATED MIP CALIBRATION FIT
    ax_7.scatter(sub_data_1['eps_mip'], sub_data_1['rho_mip'], color='#757575', alpha=0.06, s=12, edgecolors='none', rasterized=True, zorder=2)
    x_line_4 = np.linspace(0.0, X_MAX, 300)
    ax_7.plot(x_line_4, slope_sig_3 * x_line_4 + intercept_sig_3, color='#0D47A1', lw=2.2, linestyle='-', zorder=4)
    if not prof_bkg_disp.empty:
        ax_7.plot(x_line_4, slope_bkg_1 * x_line_4 + intercept_bkg_1, color='#C62828', lw=2.0, linestyle='-.', zorder=4)
    ax_7.plot(x_line_4, slope_data_3 * x_line_4 + intercept_data_3, color='black', lw=2.0, linestyle='--', zorder=4)
    yerr_asym_sig_1 = [np.maximum(0, (prof_sig_disp['mean'] - prof_sig_disp['q16']).to_numpy()), np.maximum(0, (prof_sig_disp['q84'] - prof_sig_disp['mean']).to_numpy())]
    ax_7.errorbar(prof_sig_disp['hit_bin'], prof_sig_disp['mean'], yerr=yerr_asym_sig_1, fmt='o', color='#0D47A1', ecolor='#0D47A1', elinewidth=1.6, capsize=3.0, capthick=1.2, markersize=5.0, zorder=6)
    if not prof_bkg_disp.empty:
        yerr_asym_bkg = [np.maximum(0, (prof_bkg_disp['mean'] - prof_bkg_disp['q16']).to_numpy()), np.maximum(0, (prof_bkg_disp['q84'] - prof_bkg_disp['mean']).to_numpy())]
        ax_7.errorbar(prof_bkg_disp['hit_bin'], prof_bkg_disp['mean'], yerr=yerr_asym_bkg, fmt='^', color='#C62828', ecolor='#C62828', elinewidth=1.5, capsize=3.0, capthick=1.2, markersize=4.5, zorder=6)
    yerr_asym_data_1 = [np.maximum(0, (prof_data_disp['mean'] - prof_data_disp['q16']).to_numpy()), np.maximum(0, (prof_data_disp['q84'] - prof_data_disp['mean']).to_numpy())]
    ax_7.errorbar(prof_data_disp['hit_bin'], prof_data_disp['mean'], yerr=yerr_asym_data_1, fmt='s', color='black', ecolor='black', elinewidth=1.6, capsize=3.0, capthick=1.2, markersize=4.5, zorder=6)
    ax_7.set_xscale('asinh', linear_width=5.0)
    ax_7.set_yscale('asinh', linear_width=50.0)
    ax_7.xaxis.set_major_locator(ticker.FixedLocator([0, 1, 2, 3, 4, 5, 10, 20, 40, 60, 80]))
    ax_7.yaxis.set_major_locator(ticker.FixedLocator([0, 10, 20, 30, 40, 50, 100, 200, 400, 800, 1500]))
    ax_7.set_xlim(0.0, X_MAX)
    ax_7.set_ylim(0.0, Y_MAX)
    ax_7.set_xlabel('$\\langle N_{\\mathrm{hits}}^{\\mathrm{SF}} / 10 \\rangle \\,/\\, \\overline{w}_{\\mathrm{SF}}$', fontweight='bold', fontsize=12, labelpad=8)
    ax_7.set_ylabel('$\\langle \\mathrm{QDC}^{\\mathrm{SF}} / 10 \\rangle \\,/\\, \\overline{\\mathrm{QDC}}_{\\mathrm{SF}}$', fontweight='bold', fontsize=12, labelpad=8)
    ax_7.grid(True, which='major', linestyle='--', alpha=0.35, color='gray', zorder=1)
    # 1. Fine profile used strictly for fitting in the true linear region (0.5 <= eps <= 5.0)
    ax_7.grid(True, which='minor', linestyle=':', alpha=0.15, color='gray', zorder=1)
    ax_7.tick_params(axis='both', which='major', direction='in', top=True, right=True, labelsize=10.5, length=6, width=1.1)
    legend_elements_1 = [Line2D([0], [0], color='#0D47A1', lw=2.2, marker='o', markersize=5.0, label=f'$\\mu \\to 3\\mu$ signal trimuon MC (slope $= {slope_sig_3:.2f}$)'), Line2D([0], [0], color='#C62828', lw=2.0, linestyle='-.', marker='^', markersize=4.5, label=f'Non-signal trimuon MC (slope $= {slope_bkg_1:.2f}$)'), Line2D([0], [0], color='black', lw=2.0, linestyle='--', marker='s', markersize=4.5, label=f'Run 6640 (slope $= {slope_data_3:.2f}$)')]
    ax_7.legend(handles=legend_elements_1, loc='upper left', frameon=True, framealpha=0.94, edgecolor='#BDBDBD', facecolor='white', fontsize=9.2)
    fit_summary_text = f'$\\mathbf{{Linear\\ Fit\\ (0.5 \\leq x \\leq 5.0):\\ y = m \\cdot x + c}}$\n$\\mathbf{{Signal\\ MC:}}\\quad\\ \\ \\, m = {slope_sig_3:.2f} \\pm 0.04,\\ c = {intercept_sig_3:+.2f}$\n$\\mathbf{{Non\\text{{-}}sig\\ MC:}}\\ m = {slope_bkg_1:.2f} \\pm 0.05,\\ c = {intercept_bkg_1:+.2f}$\n$\\mathbf{{Data\\ 6640:}}\\quad\\ \\, m = {slope_data_3:.2f} \\pm 0.01,\\ c = {intercept_data_3:+.2f}$'
    ax_7.text(0.05, 0.45, fit_summary_text, transform=ax_7.transAxes, fontsize=8.8, verticalalignment='bottom', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDBDBD', alpha=0.92), zorder=10)
    plt.tight_layout()
    plt.show()
    # 2. Display profile with adaptive resolution
    # Moderate bin width ensuring good density across both low and high multiplicity
    # 2. PLOTTING WITH MID-WAY SCALE TRANSITION (asinh)
    # Subsampled background scatter
    # Linear Baseline Curves (Extrapolated)
    # Profile Points with 68% Containment Spread
    # 3. SET ASINH SCALES (LINEAR REGION UP TO 5.0 / 50.0, LOG OUTSIDE)
    # Custom Explicit Ticks spanning linear and tail domains
    # 4. FORMATTING, LABELS & LEGEND
    plt.close(fig_7)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

