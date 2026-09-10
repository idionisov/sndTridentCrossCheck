#include "PassingMuonTruthProcessor.h"
#include <algorithm>
#include <iostream>

namespace snd::trident {

bool PassingMuonTruthProcessor::isFiducial(double x, double y) const {
    const double x_min = -47.5 + fConfig.fiducial_margin;
    const double x_max =  -8.5 - fConfig.fiducial_margin;
    const double y_min =  15.5 + fConfig.fiducial_margin;
    const double y_max =  54.5 - fConfig.fiducial_margin;

    return (x >= x_min && x <= x_max && y >= y_min && y <= y_max);
}

PassingMuonTruthInfo PassingMuonTruthProcessor::process(
    const TClonesArray* mcTracks,
    const TClonesArray* scifiPoints,
    const TClonesArray* mufiPoints
) const
{
    PassingMuonTruthInfo info;

    const int n_tracks = mcTracks ? mcTracks->GetEntries() : 0;
    const int n_sf_pts = scifiPoints ? scifiPoints->GetEntries() : 0;
    const int n_mf_pts = mufiPoints ? mufiPoints->GetEntries() : 0;

    info.n_scifi_points = n_sf_pts;
    info.n_mufi_points  = n_mf_pts;

    if (mcTracks && n_tracks > 0) {
        auto* trk0 = static_cast<ShipMCTrack*>(mcTracks->At(0));
        if (trk0) {
            info.mc_weight = trk0->GetWeight();
        }
    }

    if (n_sf_pts == 0 && n_mf_pts == 0) {
        info.category = MuonTruthCategory::kNoDetectorHit;
        info.category_id = 0;
        info.category_name = "NoDetectorHit";
        info.is_signal = false;
        info.is_in_acceptance = false;
        return info;
    }

    info.is_in_acceptance = true;

    // 1. Identify track IDs and particle species producing ScifiPoints
    std::set<int> muon_trkids_sf;
    std::set<int> em_trkids_sf;
    std::set<int> hadron_trkids_sf;
    int n_mu_pts = 0;
    int n_non_mu_pts = 0;

    for (int i = 0; i < n_sf_pts; ++i) {
        auto* pt = static_cast<ScifiPoint*>(scifiPoints->At(i));
        if (!pt) continue;

        int tid = pt->GetTrackID();
        if (tid >= 0 && tid < n_tracks) {
            auto* trk = static_cast<ShipMCTrack*>(mcTracks->At(tid));
            if (trk) {
                int pdg = std::abs(trk->GetPdgCode());
                if (pdg == 13) {
                    muon_trkids_sf.insert(tid);
                    n_mu_pts++;
                } else if (pdg == 11 || pdg == 22) {
                    em_trkids_sf.insert(tid);
                    n_non_mu_pts++;
                } else {
                    hadron_trkids_sf.insert(tid);
                    n_non_mu_pts++;
                }
            } else {
                n_non_mu_pts++;
            }
        } else {
            // Unindexed sub-threshold steps (Geant4 trackID < 0): associated with primary ionization
            n_mu_pts++;
        }
    }

    info.n_muons_in_scifi       = static_cast<int>(muon_trkids_sf.size());
    info.n_muon_scifi_points    = n_mu_pts;
    info.n_non_muon_scifi_points= n_non_mu_pts;
    info.non_muon_point_fraction= (n_sf_pts > 0) ? static_cast<double>(n_non_mu_pts) / n_sf_pts : 0.0;

    // 2. Identify muons producing points in Downstream MuFilter (DS: system 3 or z > 430 cm)
    std::set<int> muon_trkids_ds;
    for (int i = 0; i < n_mf_pts; ++i) {
        auto* pt = static_cast<MuFilterPoint*>(mufiPoints->At(i));
        if (!pt) continue;

        int detID = pt->GetDetectorID();
        bool is_ds = (detID / 10000 == 3) || (pt->GetZ() > 430.0);
        if (!is_ds) continue;

        int tid = pt->GetTrackID();
        if (tid >= 0 && tid < n_tracks) {
            auto* trk = static_cast<ShipMCTrack*>(mcTracks->At(tid));
            if (trk && std::abs(trk->GetPdgCode()) == 13) {
                muon_trkids_ds.insert(tid);
            }
        }
    }

    info.n_muons_in_ds = static_cast<int>(muon_trkids_ds.size());

    // 3. Category Classification Logic
    if (info.n_muons_in_scifi == 0) {
        // No primary muon entered SciFi tracker: activity is purely from halo shower
        info.category = MuonTruthCategory::kHaloShowerNoMuon;
        info.category_id = static_cast<int>(MuonTruthCategory::kHaloShowerNoMuon);
        info.category_name = "Halo_Shower_NoMuon";
        info.is_signal = false;
        info.dominant_process = "UpstreamShowerHalo";
        return info;
    }

    if (info.n_muons_in_scifi > 1) {
        // Multi-muon bundle in tracker
        info.category = MuonTruthCategory::kMultiMuon;
        info.category_id = static_cast<int>(MuonTruthCategory::kMultiMuon);
        info.category_name = "MultiMuon";
        info.is_signal = false;
        info.dominant_process = "MuonBundle";
        return info;
    }

    // Exactly 1 muon in SciFi
    int mu_id = *muon_trkids_sf.begin();
    info.primary_muon_track_id = mu_id;
    auto* mu_trk = static_cast<ShipMCTrack*>(mcTracks->At(mu_id));

    if (mu_trk) {
        info.primary_muon_pdg = mu_trk->GetPdgCode();
        info.primary_muon_p   = mu_trk->GetP();
        info.primary_muon_pt  = mu_trk->GetPt();
        info.primary_muon_pz  = mu_trk->GetPz();
        info.primary_muon_eta = TVector3(mu_trk->GetPx(), mu_trk->GetPy(), mu_trk->GetPz()).Eta();
        info.primary_muon_start_x = mu_trk->GetStartX();
        info.primary_muon_start_y = mu_trk->GetStartY();
        info.primary_muon_start_z = mu_trk->GetStartZ();

        if (std::abs(mu_trk->GetPz()) > 1e-6) {
            info.primary_muon_slope_xz = mu_trk->GetPx() / mu_trk->GetPz();
            info.primary_muon_slope_yz = mu_trk->GetPy() / mu_trk->GetPz();
        }

        // Extrapolate to Station 1 (z=300 cm) and Station 5 (z=340 cm) to check truth fiducial
        double x_st1 = mu_trk->GetStartX() + info.primary_muon_slope_xz * (300.0 - mu_trk->GetStartZ());
        double y_st1 = mu_trk->GetStartY() + info.primary_muon_slope_yz * (300.0 - mu_trk->GetStartZ());
        double x_st5 = mu_trk->GetStartX() + info.primary_muon_slope_xz * (340.0 - mu_trk->GetStartZ());
        double y_st5 = mu_trk->GetStartY() + info.primary_muon_slope_yz * (340.0 - mu_trk->GetStartZ());
        info.is_fiducial_truth = isFiducial(x_st1, y_st1) && isFiducial(x_st5, y_st5);
    }

    bool reaches_ds = (muon_trkids_ds.count(mu_id) > 0);

    // Check secondaries produced by this muon specifically in the SciFi / target tracker volume (260 < z < 360 cm)
    bool has_hadronic = (hadron_trkids_sf.size() > 0);
    bool has_hard_em  = false;
    double max_sec_e  = 0.0;
    int n_sec_det     = 0;
    std::string dom_proc = "Ionization";

    for (int i = 0; i < n_tracks; ++i) {
        auto* trk = static_cast<ShipMCTrack*>(mcTracks->At(i));
        if (!trk) continue;

        if (trk->GetMotherId() == mu_id && trk->GetStartZ() >= 260.0 && trk->GetStartZ() <= 360.0) {
            n_sec_det++;
            double p_sec = trk->GetP();
            if (p_sec > max_sec_e) {
                max_sec_e = p_sec;
                dom_proc  = trk->GetProcName().Data();
            }

            std::string proc_str = trk->GetProcName().Data();
            std::string proc_lower = proc_str;
            std::transform(proc_lower.begin(), proc_lower.end(), proc_lower.begin(), ::tolower);

            if (proc_lower.find("nuclear") != std::string::npos || 
                proc_lower.find("hadron") != std::string::npos ||
                proc_lower.find("inelastic") != std::string::npos) {
                has_hadronic = true;
            } else if (p_sec > fConfig.shower_energy_threshold) {
                has_hard_em = true;
            }
        }
    }

    // Check if secondary multiplication produced a large cascade in tracker
    if (n_sf_pts > fConfig.max_scifi_points_clean) {
        has_hard_em = true;
    }

    info.n_secondaries_in_detector = n_sec_det;
    info.max_secondary_energy      = max_sec_e;
    info.dominant_process          = dom_proc;

    // Decision tree
    if (has_hadronic) {
        info.category = MuonTruthCategory::kHadronicShower;
        info.category_id = static_cast<int>(MuonTruthCategory::kHadronicShower);
        info.category_name = "Hadronic_Shower";
        info.is_signal = false;
    } else if (!reaches_ds) {
        info.category = MuonTruthCategory::kStoppingScatteredMuon;
        info.category_id = static_cast<int>(MuonTruthCategory::kStoppingScatteredMuon);
        info.category_name = "Stopping_Scattered_Muon";
        info.is_signal = false;
    } else if (has_hard_em) {
        info.category = MuonTruthCategory::kCatastrophicEMShower;
        info.category_id = static_cast<int>(MuonTruthCategory::kCatastrophicEMShower);
        info.category_name = "Catastrophic_EM_Shower";
        info.is_signal = false;
    } else {
        info.category = MuonTruthCategory::kCleanPassingMuon;
        info.category_id = static_cast<int>(MuonTruthCategory::kCleanPassingMuon);
        info.category_name = "Clean_Passing_Muon";
        info.is_signal = true;
    }

    return info;
}

} // namespace snd::trident
