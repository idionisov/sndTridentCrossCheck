#pragma once

#include <vector>
#include <string>
#include <cmath>
#include "TClonesArray.h"
#include "TVector3.h"
#include "sndRecoTrack.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"
#include "TSeqCollection.h"
#include "TObjArray.h"

namespace snd::trident {

struct MuonCalibrationConfig {
    double chi2_max{10.0};          // Kalman fit chi2/ndf threshold in sndsw
    double max_slope{0.05};         // ~3 degrees near-normal incidence
    double fiducial_margin{1.5};    // cm margin from detector borders
    int scifi_hits_min{10};         // Typical single MIP across 10 planes
    int scifi_hits_max{35};
    int ds_hits_min{2};             // Muon filter penetration (DS)
    double ds_match_slope_max{0.04};// Max slope difference between SF and DS tracks
};

struct MuonCalibrationMetrics {
    // 1. Classification & Quality
    bool is_clean{false};
    bool pass_cut_single_scifi_track{false};
    bool pass_cut_chi2{false};
    bool pass_cut_slope{false};
    bool pass_cut_ds_match{false};
    bool pass_cut_fiducial{false};
    bool pass_cut_scifi_hits{false};
    bool pass_cut_ds_hits{false};
    int n_tracks{0};
    int n_scifi_tracks{0};
    int n_ds_tracks{0};
    float track_chi2_ndf{-1.0f};
    float track_chi2{0.0f};
    int track_ndf{0};
    bool track_flag{false};
    int track_type{0};

    // 2. Track Kinematics & Geometry (from SciFi Track)
    float track_slope_xz{0.0f};
    float track_slope_yz{0.0f};
    float track_angle_xz{0.0f};
    float track_angle_yz{0.0f};
    float track_start_x{0.0f};
    float track_start_y{0.0f};
    float track_start_z{0.0f};
    float track_stop_x{0.0f};
    float track_stop_y{0.0f};
    float track_stop_z{0.0f};

    // DS Track matching kinematics
    float ds_track_slope_xz{0.0f};
    float ds_track_slope_yz{0.0f};
    float diff_slope_xz{0.0f};
    float diff_slope_yz{0.0f};

    // 3. SciFi System Activity
    int scifi_nhits{0};
    int scifi_stations_hit{0};
    int scifi_planes_hit{0};
    double scifi_sum_qdc{0.0};
    double scifi_mean_qdc{0.0};

    // 4. SciFi Station Hit & QDC Counts (Stations 1 to 5)
    int scifi_nhits_st1{0};
    int scifi_nhits_st2{0};
    int scifi_nhits_st3{0};
    int scifi_nhits_st4{0};
    int scifi_nhits_st5{0};

    double scifi_qdc_st1{0.0};
    double scifi_qdc_st2{0.0};
    double scifi_qdc_st3{0.0};
    double scifi_qdc_st4{0.0};
    double scifi_qdc_st5{0.0};

    // 5. SciFi Individual 10 Tracking Planes (Horizontal YZ and Vertical XZ)
    int scifi_nhits_st1_h{0};
    int scifi_nhits_st1_v{0};
    int scifi_nhits_st2_h{0};
    int scifi_nhits_st2_v{0};
    int scifi_nhits_st3_h{0};
    int scifi_nhits_st3_v{0};
    int scifi_nhits_st4_h{0};
    int scifi_nhits_st4_v{0};
    int scifi_nhits_st5_h{0};
    int scifi_nhits_st5_v{0};

    double scifi_qdc_st1_h{0.0};
    double scifi_qdc_st1_v{0.0};
    double scifi_qdc_st2_h{0.0};
    double scifi_qdc_st2_v{0.0};
    double scifi_qdc_st3_h{0.0};
    double scifi_qdc_st3_v{0.0};
    double scifi_qdc_st4_h{0.0};
    double scifi_qdc_st4_v{0.0};
    double scifi_qdc_st5_h{0.0};
    double scifi_qdc_st5_v{0.0};

    // 6. MuFilter Activity
    int mufi_nhits{0};
    int ds_nhits{0};
    int ds_stations_hit{0};
    double ds_sum_qdc{0.0};
    int veto_nhits{0};
    int us_nhits{0};

    // 7. Vector Observables for RDataFrame Histo1D / Histo2D / Profile1D
    std::vector<double> hit_qdc;
    std::vector<double> hit_distance;
    std::vector<double> hit_station;
    std::vector<double> hit_orientation; // 0 = Horizontal, 1 = Vertical
    std::vector<double> plane_qdc;
    std::vector<double> plane_nhits;
    std::vector<double> plane_station;

    // Helper accessors for Profile1D / RDataFrame
    const std::vector<double>& get_hit_qdc() const { return hit_qdc; }
    const std::vector<double>& get_hit_distance() const { return hit_distance; }
    const std::vector<double>& get_hit_station() const { return hit_station; }
    const std::vector<double>& get_hit_orientation() const { return hit_orientation; }
    const std::vector<double>& get_plane_qdc() const { return plane_qdc; }
    const std::vector<double>& get_plane_nhits() const { return plane_nhits; }
    const std::vector<double>& get_plane_station() const { return plane_station; }
};

class MuonCalibrationProcessor {
public:
    explicit MuonCalibrationProcessor(const MuonCalibrationConfig& config = MuonCalibrationConfig())
        : fConfig(config) {}

    MuonCalibrationMetrics process(
        const TClonesArray* tracks,
        const TClonesArray* scifiHits,
        const TClonesArray* mufiHits
    ) const;

    // Callable interface for ROOT RDataFrame:
    MuonCalibrationMetrics operator()(
        const TClonesArray& tracks,
        const TClonesArray& scifiHits,
        const TClonesArray& mufiHits
    ) const {
        return process(&tracks, &scifiHits, &mufiHits);
    }

    void setConfig(const MuonCalibrationConfig& cfg) { fConfig = cfg; }
    const MuonCalibrationConfig& getConfig() const { return fConfig; }

    bool isFiducial(double x, double y) const;
    double calculateFiberDistance(sndRecoTrack* track, sndScifiHit* hit) const;

private:
    MuonCalibrationConfig fConfig;
};

} // namespace snd::trident
