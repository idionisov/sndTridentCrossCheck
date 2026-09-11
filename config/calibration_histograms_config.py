"""
================================================================================
SND@LHC SciFi Muon Calibration Histograms & Profiles Configuration
================================================================================
Centralized configuration defining 1D, 2D, and Profile observables for:
1. Single-MIP hit charge & plane-summed QDC distributions (Landau fits)
2. Position-dependent fiber attenuation profiles (QDC vs distance along fiber)
3. Single muon track kinematics, collimation, fiducial volume, and quality metrics
4. Downstream (DS) MuFilter penetration and veto isolation observables
================================================================================
"""

# ==============================================================================
# 1. 1D HISTOGRAM CONFIGURATIONS
# Format: (name, title, nbins, xmin, xmax, category)
# ==============================================================================
CALIB_CONFIGS_1D = [
    # --------------------------------------------------------------------------
    # A. Core QDC Calibration Observables (Single MIP Standard Candle)
    # --------------------------------------------------------------------------
    ("hit_qdc", "SciFi Single Hit QDC (Clean Muon Track);Hit QDC [a.u.];Hits", 150, 0, 30, "calibration"),
    ("plane_qdc", "SciFi Plane Summed QDC (Clean Muon Track);Plane QDC [a.u.];Planes", 160, 0, 40, "calibration"),
    ("hit_distance", "Track Impact Distance to SiPM;Distance along Fiber [cm];Hits", 80, 0, 40, "attenuation"),
    ("plane_nhits", "SciFi Hits per Plane;Hits per Plane;Planes", 15, 0, 15, "tracking"),

    # --------------------------------------------------------------------------
    # B. Track Reconstruction & Fit Quality Metrics
    # --------------------------------------------------------------------------
    ("track_chi2_ndf", "Track Fit #chi^{2}/ndf;#chi^{2}/ndf;Tracks", 150, 0, 15, "quality"),
    ("track_slope_xz", "Track Slope dx/dz;dx/dz;Tracks", 100, -0.06, 0.06, "kinematics"),
    ("track_slope_yz", "Track Slope dy/dz;dy/dz;Tracks", 100, -0.06, 0.06, "kinematics"),
    ("track_angle_xz", "Track Angle #theta_{xz};#theta_{xz} [rad];Tracks", 100, -0.06, 0.06, "kinematics"),
    ("track_angle_yz", "Track Angle #theta_{yz};#theta_{yz} [rad];Tracks", 100, -0.06, 0.06, "kinematics"),

    # --------------------------------------------------------------------------
    # C. Spatial Containment & Fiducial Geometry
    # --------------------------------------------------------------------------
    ("track_start_x", "Track Extrapolation X at Station 1;X [cm];Tracks", 100, -50, -5, "geometry"),
    ("track_start_y", "Track Extrapolation Y at Station 1;Y [cm];Tracks", 100, 10, 60, "geometry"),
    ("track_stop_x", "Track Extrapolation X at Station 5;X [cm];Tracks", 100, -50, -5, "geometry"),
    ("track_stop_y", "Track Extrapolation Y at Station 5;Y [cm];Tracks", 100, 10, 60, "geometry"),

    # --------------------------------------------------------------------------
    # D. SciFi Activity & Station Profiles
    # --------------------------------------------------------------------------
    ("scifi_nhits", "Total SciFi Hits (Clean Event);SciFi Hits;Tracks", 60, 0, 60, "activity"),
    ("scifi_sum_qdc", "Total SciFi Integrated QDC;Total QDC [a.u.];Tracks", 150, 0, 300, "activity"),
    ("scifi_mean_qdc", "SciFi Mean Hit QDC;Mean Hit QDC [a.u.];Tracks", 100, 0, 20, "activity"),
    ("scifi_stations_hit", "SciFi Stations Fired;Stations Fired (out of 5);Tracks", 6, 0.5, 6.5, "activity"),
    ("scifi_planes_hit", "SciFi Planes Fired;Planes Fired (out of 10);Tracks", 11, 0.5, 11.5, "activity"),

    # --------------------------------------------------------------------------
    # E. MuFilter Downstream Penetration & Isolation
    # --------------------------------------------------------------------------
    ("ds_nhits", "Downstream MuFilter Hits;DS Hits;Tracks", 20, 0, 20, "penetration"),
    ("ds_stations_hit", "Downstream Stations Fired;DS Stations Fired (out of 4);Tracks", 5, 0.5, 5.5, "penetration"),
    ("ds_sum_qdc", "Downstream Integrated QDC;DS Total QDC [a.u.];Tracks", 100, 0, 200, "penetration"),
    ("veto_nhits", "Veto Station Hits;Veto Hits;Tracks", 10, 0, 10, "isolation"),
    ("us_nhits", "Upstream MuFilter Hits;US Hits;Tracks", 20, 0, 20, "isolation"),
]

# ==============================================================================
# 2. 2D CORRELATION HISTOGRAM CONFIGURATIONS
# Format: (name, title, nbins_x, xmin, xmax, nbins_y, ymin, ymax, x_var, y_var, category)
# ==============================================================================
CALIB_CONFIGS_2D = [
    ("h2_hit_qdc_vs_distance", "Hit QDC vs Distance along Fiber;Distance to SiPM [cm];Hit QDC [a.u.]", 40, 0, 40, 100, 0, 30, "hit_distance", "hit_qdc", "attenuation"),
    ("h2_plane_qdc_vs_station", "Plane QDC vs Station;Station Index (1 to 5);Plane QDC [a.u.]", 5, 0.5, 5.5, 100, 0, 40, "plane_station", "plane_qdc", "longitudinal"),
    ("h2_slope_yz_vs_xz", "Track Angular Collimation;dx/dz;dy/dz", 80, -0.06, 0.06, 80, -0.06, 0.06, "track_slope_xz", "track_slope_yz", "kinematics"),
    ("h2_track_xy_station1", "Track Impact Position at Station 1;X [cm];Y [cm]", 80, -50, -5, 80, 10, 60, "track_start_x", "track_start_y", "geometry"),
]

# ==============================================================================
# 3. TPROFILE CONFIGURATIONS (Attenuation & Longitudinal Profiles)
# Format: (name, title, nbins_x, xmin, xmax, x_var, y_var, category)
# ==============================================================================
CALIB_PROFILE_CONFIGS = [
    ("prof_qdc_vs_distance", "Mean Hit QDC vs Distance along Fiber;Distance to SiPM [cm];<Hit QDC> [a.u.]", 40, 0, 40, "hit_distance", "hit_qdc", "attenuation"),
    ("prof_plane_qdc_vs_station", "Longitudinal Plane <QDC> Profile;Station (1 to 5);<Plane QDC> [a.u.]", 5, 0.5, 5.5, "plane_station", "plane_qdc", "longitudinal"),
    ("prof_plane_nhits_vs_station", "Longitudinal Plane <Hits> Profile;Station (1 to 5);<Hits per Plane>", 5, 0.5, 5.5, "plane_station", "plane_nhits", "longitudinal"),
]
