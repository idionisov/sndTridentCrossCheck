#pragma once

#include <vector>
#include <string>
#include <set>
#include <cmath>
#include "TClonesArray.h"
#include "TVector3.h"
#include "ShipMCTrack.h"
#include "ScifiPoint.h"
#include "MuFilterPoint.h"

namespace snd::trident {

enum class MuonTruthCategory : int {
    kNoDetectorHit = 0,         // No activity in SciFi or MuFilter
    kCleanPassingMuon = 1,      // Exactly 1 through-going muon traversing SciFi into Downstream MuFilter (Signal)
    kCatastrophicEMShower = 2,  // Muon undergoing hard catastrophic EM cascade (E_sec > 2.0 GeV or N_pts > 150)
    kHadronicInteraction = 3,   // Muon undergoing nuclear / hadronic interaction in detector
    kHadronicShower = 3,        // Backward-compatible alias
    kMultiMuon = 4,             // Multiple muons traversing detector simultaneously
    kHaloShowerNoMuon = 5,      // Secondary shower products (e-, gamma, n) in detector without primary muon
    kStoppingScatteredMuon = 6  // Muon entering SciFi but stopping, decaying, or scattering out before DS
};

struct PassingMuonTruthConfig {
    double shower_energy_threshold{2.0};    // 2.0 GeV secondary kinetic energy in tracker defining a catastrophic cascade
    double hadronic_energy_threshold{0.5};  // 0.5 GeV secondary energy threshold defining a hadronic interaction
    int max_scifi_points_clean{150};        // Max SciFi points for single MIP (nominal is ~30-70)
    double fiducial_margin{1.5};            // Margin in cm from SciFi borders
    double tracker_z_min{260.0};            // Target/SciFi tracker entrance z in cm
    double tracker_z_max{360.0};            // SciFi tracker exit z in cm
};

struct PassingMuonTruthInfo {
    MuonTruthCategory category{MuonTruthCategory::kNoDetectorHit};
    int category_id{0};
    std::string category_name{"NoDetectorHit"};
    bool is_signal{false};
    bool is_in_acceptance{false};
    bool is_fiducial_truth{false};
    double mc_weight{1.0};

    // Muon truth parameters (if muon exists)
    int primary_muon_track_id{-1};
    int primary_muon_pdg{0};
    double primary_muon_p{0.0};
    double primary_muon_pt{0.0};
    double primary_muon_pz{0.0};
    double primary_muon_eta{0.0};
    double primary_muon_start_x{0.0};
    double primary_muon_start_y{0.0};
    double primary_muon_start_z{0.0};
    double primary_muon_slope_xz{0.0};
    double primary_muon_slope_yz{0.0};

    // Multiplicities and points
    int n_muons_in_scifi{0};
    int n_muons_in_ds{0};
    int n_scifi_points{0};
    int n_mufi_points{0};
    int n_muon_scifi_points{0};
    int n_non_muon_scifi_points{0};
    double non_muon_point_fraction{0.0};
    int n_secondaries_in_detector{0};
    double max_secondary_energy{0.0};
    std::string dominant_process{""};
};

class PassingMuonTruthProcessor {
public:
    explicit PassingMuonTruthProcessor(const PassingMuonTruthConfig& config = PassingMuonTruthConfig())
        : fConfig(config) {}

    PassingMuonTruthInfo process(
        const TClonesArray* mcTracks,
        const TClonesArray* scifiPoints,
        const TClonesArray* mufiPoints
    ) const;

    // Callable operator for ROOT RDataFrame
    PassingMuonTruthInfo operator()(
        const TClonesArray& mcTracks,
        const TClonesArray& scifiPoints,
        const TClonesArray& mufiPoints
    ) const {
        return process(&mcTracks, &scifiPoints, &mufiPoints);
    }

    void setConfig(const PassingMuonTruthConfig& cfg) { fConfig = cfg; }
    const PassingMuonTruthConfig& getConfig() const { return fConfig; }

private:
    PassingMuonTruthConfig fConfig;
    bool isFiducial(double x, double y) const;
};

} // namespace snd::trident
