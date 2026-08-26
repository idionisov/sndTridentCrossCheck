"""
================================================================================
SND@LHC Preselection Histograms & Profiles Configuration
================================================================================
Centralized configuration defining all 1D, 2D, and Profile observables for:
1. SciFi Target System (10 planes, 5 stations, density weights, ratios, asymmetries)
2. MuFilter System (Veto 2 planes, US 5 stations, DS 7 planes, penetration metrics)
3. Cross-System Observables (DS/SciFi ratios, US/SciFi ratios, total activity)
4. 2D Correlation Histograms & Longitudinal TProfiles
================================================================================
"""

# ==============================================================================
# 1. 1D HISTOGRAM CONFIGURATIONS
# Format: (name, title, nbins, xmin, xmax, cut_dir, category)
# cut_dir: "<=" (upper threshold cut) or ">=" (lower threshold cut)
# ==============================================================================
HIST_CONFIGS_1D = [
    # --------------------------------------------------------------------------
    # A. SciFi Global & Integrated Activity
    # --------------------------------------------------------------------------
    ("scifi_nhits", "SciFi Total Hits;SciFi Hits;Normalized Entries", 500, 0, 1000, "<=", "scifi_global"),
    ("scifi_sum_qdc", "SciFi Integrated QDC;Integrated QDC [a.u.];Normalized Entries", 750, 0, 15000, "<=", "scifi_global"),
    ("scifi_max_qdc", "SciFi Max Hit QDC;Max Hit QDC [a.u.];Normalized Entries", 500, 0, 500, "<=", "scifi_global"),
    ("scifi_mean_qdc", "SciFi Mean Hit QDC;Mean Hit QDC [a.u.];Normalized Entries", 400, 0, 100, "<=", "scifi_global"),
    ("scifi_planes_hit", "SciFi Active Tracking Planes;Planes Fired (out of 10);Normalized Entries", 11, 0, 11, ">=", "scifi_global"),
    ("scifi_stations_hit", "SciFi Active Stations;Stations Fired (out of 5);Normalized Entries", 6, 0, 6, ">=", "scifi_global"),
    ("scifi_max_nhits_plane", "Max Hits in Single SciFi Plane;Max Plane Hits;Normalized Entries", 150, 0, 150, "<=", "scifi_global"),
    ("scifi_max_qdc_plane", "Max QDC in Single SciFi Plane;Max Plane QDC [a.u.];Normalized Entries", 600, 0, 3000, "<=", "scifi_global"),

    # --------------------------------------------------------------------------
    # B. SciFi Hit & QDC Density Weights (Radius = 40 Channels)
    # --------------------------------------------------------------------------
    ("scifi_max_hit_density", "SciFi Max Hit Weight Density;Max Hit Weight Density;Normalized Entries", 400, 0, 200, "<=", "scifi_density"),
    ("scifi_sum_hit_density", "SciFi Sum Hit Weight Density;Sum Hit Weight Density;Normalized Entries", 600, 0, 30000, "<=", "scifi_density"),
    ("scifi_max_qdc_density", "SciFi Max QDC Weight Density;Max QDC Weight Density;Normalized Entries", 500, 0, 2000, "<=", "scifi_density"),
    ("scifi_sum_qdc_density", "SciFi Sum QDC Weight Density;Sum QDC Weight Density;Normalized Entries", 600, 0, 150000, "<=", "scifi_density"),

    # --------------------------------------------------------------------------
    # C. SciFi Station Profiles (Stations 1 to 5)
    # --------------------------------------------------------------------------
    ("scifi_nhits_st1", "SciFi Station 1 Hits;Hits;Normalized Entries", 250, 0, 250, "<=", "scifi_stations"),
    ("scifi_nhits_st2", "SciFi Station 2 Hits;Hits;Normalized Entries", 250, 0, 250, "<=", "scifi_stations"),
    ("scifi_nhits_st3", "SciFi Station 3 Hits;Hits;Normalized Entries", 250, 0, 250, "<=", "scifi_stations"),
    ("scifi_nhits_st4", "SciFi Station 4 Hits;Hits;Normalized Entries", 250, 0, 250, "<=", "scifi_stations"),
    ("scifi_nhits_st5", "SciFi Station 5 Hits;Hits;Normalized Entries", 250, 0, 250, "<=", "scifi_stations"),

    ("scifi_qdc_st1", "SciFi Station 1 QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, "<=", "scifi_stations"),
    ("scifi_qdc_st2", "SciFi Station 2 QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, "<=", "scifi_stations"),
    ("scifi_qdc_st3", "SciFi Station 3 QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, "<=", "scifi_stations"),
    ("scifi_qdc_st4", "SciFi Station 4 QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, "<=", "scifi_stations"),
    ("scifi_qdc_st5", "SciFi Station 5 QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, "<=", "scifi_stations"),

    # --------------------------------------------------------------------------
    # D. SciFi Individual 10 Tracking Planes (Horizontal YZ and Vertical XZ)
    # --------------------------------------------------------------------------
    ("scifi_nhits_st1_h", "SciFi Station 1 Horizontal (YZ) Hits;Hits;Normalized Entries", 150, 0, 150, "<=", "scifi_planes"),
    ("scifi_nhits_st1_v", "SciFi Station 1 Vertical (XZ) Hits;Hits;Normalized Entries", 150, 0, 150, "<=", "scifi_planes"),
    ("scifi_nhits_st2_h", "SciFi Station 2 Horizontal (YZ) Hits;Hits;Normalized Entries", 150, 0, 150, "<=", "scifi_planes"),
    ("scifi_nhits_st2_v", "SciFi Station 2 Vertical (XZ) Hits;Hits;Normalized Entries", 150, 0, 150, "<=", "scifi_planes"),
    ("scifi_nhits_st3_h", "SciFi Station 3 Horizontal (YZ) Hits;Hits;Normalized Entries", 150, 0, 150, "<=", "scifi_planes"),
    ("scifi_nhits_st3_v", "SciFi Station 3 Vertical (XZ) Hits;Hits;Normalized Entries", 150, 0, 150, "<=", "scifi_planes"),
    ("scifi_nhits_st4_h", "SciFi Station 4 Horizontal (YZ) Hits;Hits;Normalized Entries", 150, 0, 150, "<=", "scifi_planes"),
    ("scifi_nhits_st4_v", "SciFi Station 4 Vertical (XZ) Hits;Hits;Normalized Entries", 150, 0, 150, "<=", "scifi_planes"),
    ("scifi_nhits_st5_h", "SciFi Station 5 Horizontal (YZ) Hits;Hits;Normalized Entries", 150, 0, 150, "<=", "scifi_planes"),
    ("scifi_nhits_st5_v", "SciFi Station 5 Vertical (XZ) Hits;Hits;Normalized Entries", 150, 0, 150, "<=", "scifi_planes"),

    ("scifi_qdc_st1_h", "SciFi Station 1 Horizontal (YZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 3000, "<=", "scifi_planes"),
    ("scifi_qdc_st1_v", "SciFi Station 1 Vertical (XZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 3000, "<=", "scifi_planes"),
    ("scifi_qdc_st2_h", "SciFi Station 2 Horizontal (YZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 3000, "<=", "scifi_planes"),
    ("scifi_qdc_st2_v", "SciFi Station 2 Vertical (XZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 3000, "<=", "scifi_planes"),
    ("scifi_qdc_st3_h", "SciFi Station 3 Horizontal (YZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 3000, "<=", "scifi_planes"),
    ("scifi_qdc_st3_v", "SciFi Station 3 Vertical (XZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 3000, "<=", "scifi_planes"),
    ("scifi_qdc_st4_h", "SciFi Station 4 Horizontal (YZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 3000, "<=", "scifi_planes"),
    ("scifi_qdc_st4_v", "SciFi Station 4 Vertical (XZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 3000, "<=", "scifi_planes"),
    ("scifi_qdc_st5_h", "SciFi Station 5 Horizontal (YZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 3000, "<=", "scifi_planes"),
    ("scifi_qdc_st5_v", "SciFi Station 5 Vertical (XZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 3000, "<=", "scifi_planes"),

    # --------------------------------------------------------------------------
    # E. SciFi Longitudinal Ratios & Projection Asymmetries
    # --------------------------------------------------------------------------
    ("scifi_qdc_ratio_down_up", "SciFi Downstream/Upstream QDC Ratio;QDC(St4+St5) / QDC(St1+St2);Normalized Entries", 400, 0, 20, ">=", "scifi_ratios"),
    ("scifi_nhits_ratio_down_up", "SciFi Downstream/Upstream Hits Ratio;Hits(St4+St5) / Hits(St1+St2);Normalized Entries", 400, 0, 20, ">=", "scifi_ratios"),
    ("scifi_qdc_ratio_st5_st1", "SciFi Station 5 / Station 1 QDC Ratio;QDC(St5) / QDC(St1);Normalized Entries", 400, 0, 20, ">=", "scifi_ratios"),
    ("scifi_qdc_asym_xy", "SciFi Horizontal vs Vertical QDC Asymmetry;(QDC_XZ - QDC_YZ) / (QDC_XZ + QDC_YZ);Normalized Entries", 200, -1.0, 1.0, "<=", "scifi_ratios"),
    ("scifi_nhits_asym_xy", "SciFi Horizontal vs Vertical Hits Asymmetry;(N_XZ - N_YZ) / (N_XZ + N_YZ);Normalized Entries", 200, -1.0, 1.0, "<=", "scifi_ratios"),

    # --------------------------------------------------------------------------
    # F. MuFilter Veto System (System 1)
    # --------------------------------------------------------------------------
    ("veto_nhits", "Veto System Total Hits;Veto Hits;Normalized Entries", 50, 0, 50, "<=", "mufilter_veto"),
    ("veto_sum_qdc", "Veto System Total QDC;Veto QDC [a.u.];Normalized Entries", 400, 0, 2000, "<=", "mufilter_veto"),
    ("veto_planes_hit", "Veto Planes Fired;Planes (0-2);Normalized Entries", 3, 0, 3, "<=", "mufilter_veto"),
    ("veto_max_qdc", "Veto Max Hit QDC;Max Hit QDC [a.u.];Normalized Entries", 200, 0, 1000, "<=", "mufilter_veto"),

    # --------------------------------------------------------------------------
    # G. Upstream MuFilter (US, System 2 - 5 Stations)
    # --------------------------------------------------------------------------
    ("us_nhits", "Upstream MuFilter Hits;US Hits;Normalized Entries", 100, 0, 100, "<=", "mufilter_us"),
    ("us_sum_qdc", "Upstream MuFilter Total QDC;US Total QDC [a.u.];Normalized Entries", 400, 0, 8000, "<=", "mufilter_us"),
    ("us_planes_hit", "Upstream MuFilter Planes Fired;US Planes Fired (0-5);Normalized Entries", 6, 0, 6, "<=", "mufilter_us"),
    ("us_max_qdc", "Upstream MuFilter Max Hit QDC;Max Hit QDC [a.u.];Normalized Entries", 400, 0, 1500, "<=", "mufilter_us"),
    ("us_nhits_st1", "US Station 1 Hits;Hits;Normalized Entries", 50, 0, 50, "<=", "mufilter_us"),
    ("us_nhits_st2", "US Station 2 Hits;Hits;Normalized Entries", 50, 0, 50, "<=", "mufilter_us"),
    ("us_nhits_st3", "US Station 3 Hits;Hits;Normalized Entries", 50, 0, 50, "<=", "mufilter_us"),
    ("us_nhits_st4", "US Station 4 Hits;Hits;Normalized Entries", 50, 0, 50, "<=", "mufilter_us"),
    ("us_nhits_st5", "US Station 5 Hits;Hits;Normalized Entries", 50, 0, 50, "<=", "mufilter_us"),
    ("us_qdc_st1", "US Station 1 QDC;QDC [a.u.];Normalized Entries", 300, 0, 3000, "<=", "mufilter_us"),
    ("us_qdc_st2", "US Station 2 QDC;QDC [a.u.];Normalized Entries", 300, 0, 3000, "<=", "mufilter_us"),
    ("us_qdc_st3", "US Station 3 QDC;QDC [a.u.];Normalized Entries", 300, 0, 3000, "<=", "mufilter_us"),
    ("us_qdc_st4", "US Station 4 QDC;QDC [a.u.];Normalized Entries", 300, 0, 3000, "<=", "mufilter_us"),
    ("us_qdc_st5", "US Station 5 QDC;QDC [a.u.];Normalized Entries", 300, 0, 3000, "<=", "mufilter_us"),

    # --------------------------------------------------------------------------
    # H. Downstream MuFilter (DS, System 3 - 4 Stations, 7 Planes)
    # --------------------------------------------------------------------------
    ("ds_nhits", "Downstream MuFilter Hits;DS Hits;Normalized Entries", 300, 0, 300, ">=", "mufilter_ds"),
    ("ds_sum_qdc", "Downstream MuFilter Total QDC;DS Total QDC [a.u.];Normalized Entries", 600, 0, 15000, ">=", "mufilter_ds"),
    ("ds_max_qdc", "Downstream MuFilter Max Hit QDC;Max Hit QDC [a.u.];Normalized Entries", 400, 0, 2000, ">=", "mufilter_ds"),
    ("ds_planes_hit", "Downstream MuFilter Planes Fired;DS Planes Fired (0-7);Normalized Entries", 8, 0, 8, ">=", "mufilter_ds"),
    ("ds_stations_hit", "Downstream MuFilter Stations Fired;DS Stations Fired (0-4);Normalized Entries", 5, 0, 5, ">=", "mufilter_ds"),
    ("ds_deepest_station", "Deepest DS Station Penetrated;DS Station (1-4);Normalized Entries", 5, 0, 5, ">=", "mufilter_ds"),
    ("ds_deepest_plane", "Deepest DS Plane Reached;DS Plane Index (1-7);Normalized Entries", 8, 0, 8, ">=", "mufilter_ds"),
    ("ds_max_nhits_plane", "Max Hits in Single DS Plane;Max DS Plane Hits;Normalized Entries", 100, 0, 100, ">=", "mufilter_ds"),
    ("ds_max_qdc_plane", "Max QDC in Single DS Plane;Max DS Plane QDC [a.u.];Normalized Entries", 500, 0, 5000, ">=", "mufilter_ds"),

    # --------------------------------------------------------------------------
    # I. Downstream Individual 7 Tracking Planes
    # --------------------------------------------------------------------------
    ("ds_nhits_st1_h", "DS Station 1 Horizontal (YZ) Hits;Hits;Normalized Entries", 100, 0, 100, ">=", "mufilter_ds_planes"),
    ("ds_nhits_st1_v", "DS Station 1 Vertical (XZ) Hits;Hits;Normalized Entries", 100, 0, 100, ">=", "mufilter_ds_planes"),
    ("ds_nhits_st2_h", "DS Station 2 Horizontal (YZ) Hits;Hits;Normalized Entries", 100, 0, 100, ">=", "mufilter_ds_planes"),
    ("ds_nhits_st2_v", "DS Station 2 Vertical (XZ) Hits;Hits;Normalized Entries", 100, 0, 100, ">=", "mufilter_ds_planes"),
    ("ds_nhits_st3_h", "DS Station 3 Horizontal (YZ) Hits;Hits;Normalized Entries", 100, 0, 100, ">=", "mufilter_ds_planes"),
    ("ds_nhits_st3_v", "DS Station 3 Vertical (XZ) Hits;Hits;Normalized Entries", 100, 0, 100, ">=", "mufilter_ds_planes"),
    ("ds_nhits_st4_v", "DS Station 4 Vertical (XZ) Hits;Hits;Normalized Entries", 100, 0, 100, ">=", "mufilter_ds_planes"),

    ("ds_qdc_st1_h", "DS Station 1 Horizontal (YZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, ">=", "mufilter_ds_planes"),
    ("ds_qdc_st1_v", "DS Station 1 Vertical (XZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, ">=", "mufilter_ds_planes"),
    ("ds_qdc_st2_h", "DS Station 2 Horizontal (YZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, ">=", "mufilter_ds_planes"),
    ("ds_qdc_st2_v", "DS Station 2 Vertical (XZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, ">=", "mufilter_ds_planes"),
    ("ds_qdc_st3_h", "DS Station 3 Horizontal (YZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, ">=", "mufilter_ds_planes"),
    ("ds_qdc_st3_v", "DS Station 3 Vertical (XZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, ">=", "mufilter_ds_planes"),
    ("ds_qdc_st4_v", "DS Station 4 Vertical (XZ) QDC;QDC [a.u.];Normalized Entries", 500, 0, 5000, ">=", "mufilter_ds_planes"),

    # --------------------------------------------------------------------------
    # J. Downstream Longitudinal Ratios & Shower Penetration
    # --------------------------------------------------------------------------
    ("ds_qdc_ratio_back_front", "DS Back/Front QDC Ratio;QDC(DS3+DS4) / QDC(DS1+DS2);Normalized Entries", 400, 0, 20, ">=", "mufilter_ds_ratios"),
    ("ds_nhits_ratio_back_front", "DS Back/Front Hits Ratio;Hits(DS3+DS4) / Hits(DS1+DS2);Normalized Entries", 400, 0, 20, ">=", "mufilter_ds_ratios"),
    ("ds_qdc_ratio_ds4_ds1", "DS Station 4 / Station 1 QDC Ratio;QDC(DS4) / QDC(DS1);Normalized Entries", 400, 0, 20, ">=", "mufilter_ds_ratios"),

    # --------------------------------------------------------------------------
    # K. Cross-System Global Metrics & Relative Proportions
    # --------------------------------------------------------------------------
    ("total_nhits", "Detector Total Hits;Total Detector Hits;Normalized Entries", 600, 0, 1200, "<=", "cross_system"),
    ("total_sum_qdc", "Detector Total Integrated QDC;Total QDC [a.u.];Normalized Entries", 600, 0, 30000, "<=", "cross_system"),
    ("ratio_ds_to_scifi_qdc", "Downstream to SciFi QDC Ratio;QDC(DS) / QDC(SciFi);Normalized Entries", 400, 0, 20, ">=", "cross_system"),
    ("ratio_ds_to_scifi_nhits", "Downstream to SciFi Hits Ratio;Hits(DS) / Hits(SciFi);Normalized Entries", 250, 0, 5, ">=", "cross_system"),
    ("ratio_us_to_scifi_qdc", "Upstream MuFilter to SciFi QDC Ratio;QDC(US) / QDC(SciFi);Normalized Entries", 400, 0, 20, "<=", "cross_system"),
    ("ratio_mufi_to_scifi_qdc", "Total MuFilter to SciFi QDC Ratio;QDC(US+DS) / QDC(SciFi);Normalized Entries", 400, 0, 20, ">=", "cross_system"),
]

# ==============================================================================
# 2. 2D CORRELATION HISTOGRAM CONFIGURATIONS
# Format: (name, title, nbins_x, xmin, xmax, nbins_y, ymin, ymax, x_var, y_var, category)
# ==============================================================================
HIST_CONFIGS_2D = [
    ("h2_scifi_qdc_vs_nhits", "SciFi Total QDC vs Hits;SciFi Hits;SciFi Total QDC [a.u.]", 100, 0, 500, 100, 0, 10000, "scifi_nhits", "scifi_sum_qdc", "scifi_correlations"),
    ("h2_ds_qdc_vs_nhits", "DS Total QDC vs Hits;DS Hits;DS Total QDC [a.u.]", 80, 0, 160, 80, 0, 10000, "ds_nhits", "ds_sum_qdc", "ds_correlations"),
    ("h2_ds_qdc_vs_scifi_qdc", "DS Total QDC vs SciFi Total QDC;SciFi Total QDC [a.u.];DS Total QDC [a.u.]", 100, 0, 10000, 100, 0, 10000, "scifi_sum_qdc", "ds_sum_qdc", "cross_correlations"),
    ("h2_ds_nhits_vs_scifi_nhits", "DS Hits vs SciFi Hits;SciFi Hits;DS Hits", 100, 0, 400, 80, 0, 160, "scifi_nhits", "ds_nhits", "cross_correlations"),
    ("h2_ds_ratio_vs_scifi_ratio", "DS Back/Front Ratio vs SciFi Down/Up Ratio;SciFi QDC Ratio (St4+5)/(St1+2);DS QDC Ratio (DS3+4)/(DS1+2)", 100, 0, 10, 100, 0, 10, "scifi_qdc_ratio_down_up", "ds_qdc_ratio_back_front", "ratio_correlations"),
    ("h2_max_qdc_density_vs_sum_qdc", "Max QDC Density vs Total SciFi QDC;SciFi Total QDC [a.u.];Max QDC Density [a.u.]", 100, 0, 10000, 100, 0, 2000, "scifi_sum_qdc", "scifi_max_qdc_density", "density_correlations"),
    ("h2_ds_deepest_st_vs_scifi_nhits", "Deepest DS Station vs SciFi Hits;SciFi Hits;Deepest DS Station (1-4)", 100, 0, 400, 5, 0.5, 5.5, "scifi_nhits", "ds_deepest_station", "penetration_correlations"),
]

# ==============================================================================
# 3. TPROFILE CONFIGURATIONS (Longitudinal Detector Profiles)
# Format: (name, title, nbins_x, xmin, xmax, category)
# ==============================================================================
PROFILE_CONFIGS = [
    ("prof_scifi_qdc_vs_station", "SciFi Longitudinal <QDC> Profile;Station (1 to 5);<QDC> [a.u.]", 5, 0.5, 5.5, "scifi_profiles"),
    ("prof_scifi_nhits_vs_station", "SciFi Longitudinal <Hits> Profile;Station (1 to 5);<Hits>", 5, 0.5, 5.5, "scifi_profiles"),
    ("prof_ds_qdc_vs_station", "DS Longitudinal <QDC> Profile;DS Station (1 to 4);<QDC> [a.u.]", 4, 0.5, 4.5, "ds_profiles"),
    ("prof_ds_nhits_vs_station", "DS Longitudinal <Hits> Profile;DS Station (1 to 4);<Hits>", 4, 0.5, 4.5, "ds_profiles"),
    ("prof_detector_longitudinal_qdc", "Complete Detector Longitudinal Profile;Plane Index (1=Veto, 3=SF, 13=US, 18=DS);<QDC> [a.u.]", 24, 0.5, 24.5, "global_profiles"),
]
