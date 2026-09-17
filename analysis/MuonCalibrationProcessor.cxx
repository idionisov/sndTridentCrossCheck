#include "MuonCalibrationProcessor.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace snd::trident {

bool MuonCalibrationProcessor::isFiducial(double x, double y) const {
    const double x_min = -47.5 + fConfig.fiducial_margin;
    const double x_max =  -8.5 - fConfig.fiducial_margin;
    const double y_min =  15.5 + fConfig.fiducial_margin;
    const double y_max =  54.5 - fConfig.fiducial_margin;

    return (x >= x_min && x <= x_max && y >= y_min && y <= y_max);
}

double MuonCalibrationProcessor::calculateFiberDistance(sndRecoTrack* track, sndScifiHit* hit) const {
    if (!track || !hit) return 0.0;

    TVector3 start = track->getStart();
    TVector3 stop  = track->getStop();
    double dz = stop.Z() - start.Z();
    if (std::abs(dz) < 1e-6) dz = 1e-6;

    double slopeX = (stop.X() - start.X()) / dz;
    double slopeY = (stop.Y() - start.Y()) / dz;

    int st = hit->GetStation();
    if (st < 1) st = 1;
    if (st > 5) st = 5;

    // Nominal station Z locations (~300 cm + (st-1)*8.5 cm)
    double z_st = 300.0 + (st - 1) * 8.5;
    double x_impact = start.X() + slopeX * (z_st - start.Z());
    double y_impact = start.Y() + slopeY * (z_st - start.Z());

    if (hit->isVertical()) {
        // Fibers run along Y: readout is at the top (y ~ 54.5 cm)
        return std::abs(54.5 - y_impact);
    } else {
        // Fibers run along X: readout is at the side (x ~ -47.5 cm)
        return std::abs(x_impact - (-47.5));
    }
}

TVector3 MuonCalibrationProcessor::extrapolateToZ(sndRecoTrack* track, double z) const {
    if (!track) return TVector3(0.0, 0.0, z);
    TVector3 start = track->getStart();
    double slopeX = track->getSlopeXZ();
    double slopeY = track->getSlopeYZ();
    return TVector3(
        start.X() + slopeX * (z - start.Z()),
        start.Y() + slopeY * (z - start.Z()),
        z
    );
}

MuonCalibrationMetrics MuonCalibrationProcessor::process(
    const TClonesArray* tracks,
    const TClonesArray* scifiHits,
    const TClonesArray* mufiHits
) const
{
    MuonCalibrationMetrics m;

    // -------------------------------------------------------------------------
    // 1. Process Track Reconstruction (Hough Tracks: SciFi type 11, DS type 13)
    // -------------------------------------------------------------------------
    sndRecoTrack* scifiTrack = nullptr;
    sndRecoTrack* dsTrack    = nullptr;

    int n_scifi_tracks = 0;
    int n_ds_tracks    = 0;

    if (tracks) {
        m.n_tracks = tracks->GetEntries();
        for (int i = 0; i < m.n_tracks; ++i) {
            auto* trk = dynamic_cast<sndRecoTrack*>(tracks->At(i));
            if (!trk) continue;

            int trk_type = trk->getTrackType();
            if (trk_type == fConfig.scifi_track_type) {
                n_scifi_tracks++;
                if (!scifiTrack || (!scifiTrack->getTrackFlag() && trk->getTrackFlag())) {
                    scifiTrack = trk;
                }
            } else if (trk_type == fConfig.ds_track_type) {
                n_ds_tracks++;
                if (!dsTrack || (!dsTrack->getTrackFlag() && trk->getTrackFlag())) {
                    dsTrack = trk;
                }
            }
        }
    }

    m.n_scifi_tracks = n_scifi_tracks;
    m.n_ds_tracks    = n_ds_tracks;

    // Requirement 1: Exactly 1 SciFi track (Hough transform type 11)
    m.pass_cut_single_scifi_track = (m.n_scifi_tracks == 1 && scifiTrack != nullptr);

    // Requirement 2: At least 1 DS track (Hough transform type 13)
    m.pass_cut_ds_track = (m.n_ds_tracks >= 1 && dsTrack != nullptr);

    if (scifiTrack) {
        m.track_flag     = scifiTrack->getTrackFlag();
        m.track_chi2     = scifiTrack->getChi2();
        m.track_ndf      = scifiTrack->getNdf();
        m.track_chi2_ndf = scifiTrack->getChi2Ndf();
        m.track_type     = scifiTrack->getTrackType();
        m.track_slope_xz = scifiTrack->getSlopeXZ();
        m.track_slope_yz = scifiTrack->getSlopeYZ();
        m.track_angle_xz = scifiTrack->getAngleXZ();
        m.track_angle_yz = scifiTrack->getAngleYZ();

        TVector3 start = scifiTrack->getStart();
        TVector3 stop  = scifiTrack->getStop();
        m.track_start_x  = start.X();
        m.track_start_y  = start.Y();
        m.track_start_z  = start.Z();
        m.track_stop_x   = stop.X();
        m.track_stop_y   = stop.Y();
        m.track_stop_z   = stop.Z();

        m.pass_cut_chi2_scifi  = (m.track_flag && m.track_chi2_ndf > 0.0f && m.track_chi2_ndf <= fConfig.chi2_max_scifi);
        m.pass_cut_slope_scifi = (std::abs(m.track_slope_xz) < fConfig.max_slope && 
                                  std::abs(m.track_slope_yz) < fConfig.max_slope);
        m.pass_cut_fiducial    = isFiducial(start.X(), start.Y()) && isFiducial(stop.X(), stop.Y());
    }

    if (dsTrack) {
        m.ds_track_flag     = dsTrack->getTrackFlag();
        m.ds_track_chi2     = dsTrack->getChi2();
        m.ds_track_ndf      = dsTrack->getNdf();
        m.ds_track_chi2_ndf = dsTrack->getChi2Ndf();
        m.ds_track_type     = dsTrack->getTrackType();
        m.ds_track_slope_xz = dsTrack->getSlopeXZ();
        m.ds_track_slope_yz = dsTrack->getSlopeYZ();
        m.ds_track_angle_xz = dsTrack->getAngleXZ();
        m.ds_track_angle_yz = dsTrack->getAngleYZ();

        m.pass_cut_chi2_ds  = (m.ds_track_flag && m.ds_track_chi2_ndf > 0.0f && m.ds_track_chi2_ndf <= fConfig.chi2_max_ds);
        m.pass_cut_slope_ds = (std::abs(m.ds_track_slope_xz) < fConfig.max_slope && 
                               std::abs(m.ds_track_slope_yz) < fConfig.max_slope);
    }

    // Separate chi2 and slope requirements:
    m.pass_cut_chi2  = m.pass_cut_chi2_scifi && m.pass_cut_chi2_ds;
    m.pass_cut_slope = m.pass_cut_slope_scifi && m.pass_cut_slope_ds;

    // Extrapolation to fiducial plane at z = z_match (430.0 cm) and track matching
    if (scifiTrack && dsTrack) {
        TVector3 pt_sf = extrapolateToZ(scifiTrack, fConfig.z_match);
        TVector3 pt_ds = extrapolateToZ(dsTrack, fConfig.z_match);

        m.scifi_x_430 = pt_sf.X();
        m.scifi_y_430 = pt_sf.Y();
        m.ds_x_430    = pt_ds.X();
        m.ds_y_430    = pt_ds.Y();

        m.diff_x_430   = std::abs(m.scifi_x_430 - m.ds_x_430);
        m.diff_y_430   = std::abs(m.scifi_y_430 - m.ds_y_430);
        m.diff_pos_430 = std::sqrt(m.diff_x_430 * m.diff_x_430 + m.diff_y_430 * m.diff_y_430);

        m.diff_slope_xz = std::abs(m.track_slope_xz - m.ds_track_slope_xz);
        m.diff_slope_yz = std::abs(m.track_slope_yz - m.ds_track_slope_yz);
        m.diff_angle_xz = std::abs(m.track_angle_xz - m.ds_track_angle_xz);
        m.diff_angle_yz = std::abs(m.track_angle_yz - m.ds_track_angle_yz);
        m.diff_angle_3d = std::sqrt(m.diff_angle_xz * m.diff_angle_xz + m.diff_angle_yz * m.diff_angle_yz);

        // 1. Cross fiducial plane at z=430 cm
        m.pass_cut_fiducial_430 = isFiducial(m.scifi_x_430, m.scifi_y_430) && isFiducial(m.ds_x_430, m.ds_y_430);

        // 2. Position match within tolerance (<= 3.0 cm)
        m.pass_cut_pos_match_430 = (m.diff_pos_430 <= fConfig.pos_match_max);

        // 3. Angular match within tolerance (<= 0.015 rad)
        m.pass_cut_angle_match = (m.diff_angle_xz <= fConfig.angle_match_max && m.diff_angle_yz <= fConfig.angle_match_max);

        // Combined DS match requirement
        m.pass_cut_ds_match = m.pass_cut_pos_match_430 && m.pass_cut_angle_match;
    }

    // -------------------------------------------------------------------------
    // 2. Process SciFi Hits
    // -------------------------------------------------------------------------
    if (scifiHits) {
        int n_sf = scifiHits->GetEntries();
        int n_sf_valid = 0;

        int nhits_st[5] = {0};
        double qdc_st[5] = {0.0};
        int nhits_plane_h[5] = {0};
        int nhits_plane_v[5] = {0};
        double qdc_plane_h[5] = {0.0};
        double qdc_plane_v[5] = {0.0};

        std::set<int> stations_hit;
        std::set<int> planes_hit;

        m.hit_qdc.reserve(n_sf);
        m.hit_distance.reserve(n_sf);
        m.hit_station.reserve(n_sf);
        m.hit_orientation.reserve(n_sf);

        for (int i = 0; i < n_sf; ++i) {
            auto* hit = static_cast<sndScifiHit*>(scifiHits->At(i));
            if (!hit) continue;

            if (fConfig.scifi_qdc_min > 0.0) {
                if (hit->GetSignal(0) < fConfig.scifi_qdc_min) continue;
            } else {
                if (!hit->isValid()) continue;
            }

            double qdc = hit->GetSignal(0);
            if (qdc <= 0.0) continue;

            int st = hit->GetStation();
            if (st < 1 || st > 5) continue;
            int st_idx = st - 1;
            bool vert = hit->isVertical();
            int plane_idx = st_idx * 2 + (vert ? 1 : 0);

            stations_hit.insert(st);
            planes_hit.insert(plane_idx);

            m.scifi_sum_qdc += qdc;
            nhits_st[st_idx]++;
            qdc_st[st_idx] += qdc;
            n_sf_valid++;

            if (vert) {
                nhits_plane_v[st_idx]++;
                qdc_plane_v[st_idx] += qdc;
            } else {
                nhits_plane_h[st_idx]++;
                qdc_plane_h[st_idx] += qdc;
            }

            // Hit observables for calibration
            m.hit_qdc.push_back(qdc);
            m.hit_station.push_back(static_cast<double>(st));
            m.hit_orientation.push_back(vert ? 1.0 : 0.0);

            if (scifiTrack) {
                double dist = calculateFiberDistance(scifiTrack, hit);
                m.hit_distance.push_back(dist);
            } else {
                m.hit_distance.push_back(-1.0);
            }
        }

        m.scifi_nhits = n_sf_valid;
        m.scifi_stations_hit = static_cast<int>(stations_hit.size());
        m.scifi_planes_hit   = static_cast<int>(planes_hit.size());
        if (m.scifi_nhits > 0) {
            m.scifi_mean_qdc = m.scifi_sum_qdc / m.scifi_nhits;
        }

        m.scifi_nhits_st1 = nhits_st[0];
        m.scifi_nhits_st2 = nhits_st[1];
        m.scifi_nhits_st3 = nhits_st[2];
        m.scifi_nhits_st4 = nhits_st[3];
        m.scifi_nhits_st5 = nhits_st[4];

        m.scifi_qdc_st1 = qdc_st[0];
        m.scifi_qdc_st2 = qdc_st[1];
        m.scifi_qdc_st3 = qdc_st[2];
        m.scifi_qdc_st4 = qdc_st[3];
        m.scifi_qdc_st5 = qdc_st[4];

        m.scifi_nhits_st1_h = nhits_plane_h[0];
        m.scifi_nhits_st1_v = nhits_plane_v[0];
        m.scifi_nhits_st2_h = nhits_plane_h[1];
        m.scifi_nhits_st2_v = nhits_plane_v[1];
        m.scifi_nhits_st3_h = nhits_plane_h[2];
        m.scifi_nhits_st3_v = nhits_plane_v[2];
        m.scifi_nhits_st4_h = nhits_plane_h[3];
        m.scifi_nhits_st4_v = nhits_plane_v[3];
        m.scifi_nhits_st5_h = nhits_plane_h[4];
        m.scifi_nhits_st5_v = nhits_plane_v[4];

        m.scifi_qdc_st1_h = qdc_plane_h[0];
        m.scifi_qdc_st1_v = qdc_plane_v[0];
        m.scifi_qdc_st2_h = qdc_plane_h[1];
        m.scifi_qdc_st2_v = qdc_plane_v[1];
        m.scifi_qdc_st3_h = qdc_plane_h[2];
        m.scifi_qdc_st3_v = qdc_plane_v[2];
        m.scifi_qdc_st4_h = qdc_plane_h[3];
        m.scifi_qdc_st4_v = qdc_plane_v[3];
        m.scifi_qdc_st5_h = qdc_plane_h[4];
        m.scifi_qdc_st5_v = qdc_plane_v[4];

        // Fill active plane vector observables for RDataFrame
        for (int st_i = 0; st_i < 5; ++st_i) {
            if (nhits_plane_h[st_i] > 0) {
                m.plane_qdc.push_back(qdc_plane_h[st_i]);
                m.plane_nhits.push_back(static_cast<double>(nhits_plane_h[st_i]));
                m.plane_station.push_back(static_cast<double>(st_i + 1));
            }
            if (nhits_plane_v[st_i] > 0) {
                m.plane_qdc.push_back(qdc_plane_v[st_i]);
                m.plane_nhits.push_back(static_cast<double>(nhits_plane_v[st_i]));
                m.plane_station.push_back(static_cast<double>(st_i + 1));
            }
        }
    }

    // -------------------------------------------------------------------------
    // 3. Process MuFilter Hits
    // -------------------------------------------------------------------------
    if (mufiHits) {
        int n_mf = mufiHits->GetEntries();
        int n_mf_valid = 0;
        std::set<int> ds_stations;

        for (int i = 0; i < n_mf; ++i) {
            auto* hit = static_cast<MuFilterHit*>(mufiHits->At(i));
            if (!hit || !hit->isValid()) continue;

            int sys = hit->GetSystem(); // 1=Veto, 2=US, 3=DS
            int st  = hit->GetPlane();  // station/plane index

            n_mf_valid++;
            std::map<Int_t, Float_t> sigs = hit->GetAllSignals(true, true, false);
            double hit_qdc = 0.0;
            for (const auto& kv : sigs) {
                hit_qdc += kv.second;
            }

            if (sys == 1) {
                m.veto_nhits++;
                m.veto_sum_qdc += hit_qdc;
            } else if (sys == 2) {
                m.us_nhits++;
                m.us_sum_qdc += hit_qdc;
            } else if (sys == 3) {
                m.ds_nhits++;
                m.ds_sum_qdc += hit_qdc;
                ds_stations.insert(st);
            }
        }
        m.mufi_nhits = n_mf_valid;
        m.ds_stations_hit = static_cast<int>(ds_stations.size());
    }

    // -------------------------------------------------------------------------
    // 4. Final Clean Muon Selection Decision
    // -------------------------------------------------------------------------
    // Hit count cuts are computed for offline monitoring but NOT applied in is_clean
    m.pass_cut_scifi_hits = (m.scifi_nhits >= fConfig.scifi_hits_min && 
                             m.scifi_nhits <= fConfig.scifi_hits_max);
    m.pass_cut_ds_hits    = (m.ds_nhits >= fConfig.ds_hits_min);

    // Clean single muon selection based purely on SciFi and DS track kinematics & geometry:
    // (Strictly without number of hits or QDC cuts)
    m.is_clean = m.pass_cut_single_scifi_track && 
                 m.pass_cut_ds_track && 
                 m.pass_cut_chi2 && 
                 m.pass_cut_slope && 
                 m.pass_cut_fiducial_430 && 
                 m.pass_cut_ds_match;

    return m;
}

} // namespace snd::trident
