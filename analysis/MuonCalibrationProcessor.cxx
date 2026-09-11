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

MuonCalibrationMetrics MuonCalibrationProcessor::process(
    const TClonesArray* tracks,
    const TClonesArray* scifiHits,
    const TClonesArray* mufiHits
) const
{
    MuonCalibrationMetrics m;

    // -------------------------------------------------------------------------
    // 1. Process Track Reconstruction (Distinguishing SciFi vs DS tracks)
    // -------------------------------------------------------------------------
    sndRecoTrack* scifiKalmanTrack = nullptr;
    sndRecoTrack* scifiHoughTrack  = nullptr;
    sndRecoTrack* dsKalmanTrack    = nullptr;
    sndRecoTrack* dsHoughTrack     = nullptr;

    int n_scifi_kalman = 0;
    int n_scifi_hough  = 0;
    int n_ds_kalman    = 0;
    int n_ds_hough     = 0;

    if (tracks) {
        m.n_tracks = tracks->GetEntries();
        for (int i = 0; i < m.n_tracks; ++i) {
            auto* trk = dynamic_cast<sndRecoTrack*>(tracks->At(i));
            if (!trk) continue;

            int trk_type = trk->getTrackType();
            if (trk_type == 1) {
                n_scifi_kalman++;
                if (!scifiKalmanTrack && trk->getTrackFlag()) scifiKalmanTrack = trk;
            } else if (trk_type == 11) {
                n_scifi_hough++;
                if (!scifiHoughTrack && trk->getTrackFlag()) scifiHoughTrack = trk;
            } else if (trk_type == 3) {
                n_ds_kalman++;
                if (!dsKalmanTrack && trk->getTrackFlag()) dsKalmanTrack = trk;
            } else if (trk_type == 13) {
                n_ds_hough++;
                if (!dsHoughTrack && trk->getTrackFlag()) dsHoughTrack = trk;
            }
        }
    }

    // Single physical SciFi track: prefer Kalman track count, fallback to Hough count
    m.n_scifi_tracks = (n_scifi_kalman > 0) ? n_scifi_kalman : n_scifi_hough;
    m.n_ds_tracks    = (n_ds_kalman > 0) ? n_ds_kalman : n_ds_hough;

    sndRecoTrack* scifiTrack = scifiKalmanTrack ? scifiKalmanTrack : scifiHoughTrack;
    sndRecoTrack* dsTrack    = dsKalmanTrack ? dsKalmanTrack : dsHoughTrack;

    m.pass_cut_single_scifi_track = (m.n_scifi_tracks == 1);
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

        m.pass_cut_chi2 = (m.track_flag && m.track_chi2_ndf > 0.0f && m.track_chi2_ndf <= fConfig.chi2_max);
        m.pass_cut_slope = (std::abs(m.track_slope_xz) <= fConfig.max_slope && 
                            std::abs(m.track_slope_yz) <= fConfig.max_slope);
        m.pass_cut_fiducial = isFiducial(start.X(), start.Y()) && isFiducial(stop.X(), stop.Y());

        bool ds_match = true;
        if (dsTrack) {
            m.ds_track_slope_xz = dsTrack->getSlopeXZ();
            m.ds_track_slope_yz = dsTrack->getSlopeYZ();
            m.diff_slope_xz = std::abs(m.track_slope_xz - m.ds_track_slope_xz);
            m.diff_slope_yz = std::abs(m.track_slope_yz - m.ds_track_slope_yz);
            if (m.diff_slope_xz > fConfig.ds_match_slope_max || m.diff_slope_yz > fConfig.ds_match_slope_max) {
                ds_match = false;
            }
        }
        m.pass_cut_ds_match = ds_match;
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
            if (sys == 1) {
                m.veto_nhits++;
            } else if (sys == 2) {
                m.us_nhits++;
            } else if (sys == 3) {
                m.ds_nhits++;
                ds_stations.insert(st);
                for (int ch = 0; ch < 16; ++ch) {
                    double sig = hit->GetSignal(ch);
                    if (sig > 0.0) m.ds_sum_qdc += sig;
                }
            }
        }
        m.mufi_nhits = n_mf_valid;
        m.ds_stations_hit = static_cast<int>(ds_stations.size());
    }

    // -------------------------------------------------------------------------
    // 4. Final Clean Muon Selection Decision
    // -------------------------------------------------------------------------
    m.pass_cut_scifi_hits = (m.scifi_nhits >= fConfig.scifi_hits_min && 
                             m.scifi_nhits <= fConfig.scifi_hits_max);
    m.pass_cut_ds_hits    = (m.ds_nhits >= fConfig.ds_hits_min);

    m.is_clean = m.pass_cut_single_scifi_track && 
                 m.pass_cut_chi2 && 
                 m.pass_cut_slope && 
                 m.pass_cut_ds_match && 
                 m.pass_cut_fiducial && 
                 m.pass_cut_scifi_hits;

    return m;
}

} // namespace snd::trident
