#include "DigiValidationProcessor.h"
#include <algorithm>
#include <iostream>

namespace snd::trident {

DigiValidationProcessor::DigiValidationProcessor(Scifi* scifi) {
    if (scifi) {
        initCache(scifi);
    }
}

void DigiValidationProcessor::initCache(Scifi* scifi) {
    if (!scifi) return;
    fChannelMap.clear();
    fChannelMap.reserve(16000);

    TVector3 A, B;
    for (int st = 1; st <= 5; ++st) {
        for (int p = 0; p <= 1; ++p) {
            int plane_offset = p * 100000;
            for (int m = 0; m < 3; ++m) {
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 128; ++c) {
                        int det_id = st * 1000000 + plane_offset + m * 10000 + r * 1000 + c;
                        try {
                            scifi->GetSiPMPosition(det_id, A, B);
                            ChannelGeometry cg;
                            cg.xA = static_cast<float>(A.X());
                            cg.yA = static_cast<float>(A.Y());
                            cg.zA = static_cast<float>(A.Z());
                            cg.xB = static_cast<float>(B.X());
                            cg.yB = static_cast<float>(B.Y());
                            cg.zB = static_cast<float>(B.Z());
                            cg.z_mid = 0.5f * (cg.zA + cg.zB);
                            cg.is_vertical = (p == 1);

                            // In SND@LHC SciFi:
                            // Vertical fibers run along Y: readout SiPM is at B (top, y ~ 54 cm)
                            // Horizontal fibers run along X: readout SiPM is at A (side, x ~ -46 cm)
                            if (cg.is_vertical) {
                                cg.sipm_x = cg.xB;
                                cg.sipm_y = cg.yB;
                                cg.sipm_z = cg.zB;
                            } else {
                                cg.sipm_x = cg.xA;
                                cg.sipm_y = cg.yA;
                                cg.sipm_z = cg.zA;
                            }
                            fChannelMap[det_id] = cg;
                        } catch (...) {
                            // Skip non-existent channels
                        }
                    }
                }
            }
        }
    }
    fHasGeometryCache = !fChannelMap.empty();
}

DigiValidationSummary DigiValidationProcessor::process(
    const TClonesArray* scifiHits,
    const TClonesArray* clusters,
    const TClonesArray* mufiHits,
    const TClonesArray* tracks,
    double event_weight
) const {
    DigiValidationSummary s;
    s.event_weight = event_weight;

    // 1. SciFi Hits
    std::unordered_map<int, double> hit_qdc_map;
    if (scifiHits) {
        int n_sf = scifiHits->GetEntries();
        hit_qdc_map.reserve(n_sf);
        for (int i = 0; i < n_sf; ++i) {
            auto* h = static_cast<sndScifiHit*>(scifiHits->At(i));
            if (!h || !h->isValid()) continue;

            double qdc = h->GetSignal(0);
            s.n_scifi_hits += 1.0;
            s.scifi_sum_qdc += qdc;

            int st = h->GetStation();
            if (st == 1)      s.n_scifi_st1 += 1.0;
            else if (st == 2) s.n_scifi_st2 += 1.0;
            else if (st == 3) s.n_scifi_st3 += 1.0;
            else if (st == 4) s.n_scifi_st4 += 1.0;
            else if (st == 5) s.n_scifi_st5 += 1.0;

            if (st >= 1 && st <= 5) {
                s.station_hits[st - 1] += 1.0;
                s.station_qdc[st - 1] += qdc;
            }

            hit_qdc_map[h->GetDetectorID()] = qdc;
        }
        if (s.n_scifi_hits > 0.0) {
            s.scifi_mean_qdc = s.scifi_sum_qdc / s.n_scifi_hits;
        }
    }

    // 2. Clusters
    if (clusters) {
        int n_cl = clusters->GetEntries();
        s.n_scifi_clusters = static_cast<double>(n_cl);
        s.cluster_size.reserve(n_cl);
        s.cluster_qdc.reserve(n_cl);

        for (int i = 0; i < n_cl; ++i) {
            auto* cl = static_cast<sndCluster*>(clusters->At(i));
            if (!cl) continue;
            int n_ch = cl->GetN();
            s.cluster_size.push_back(static_cast<double>(n_ch));

            double qdc_sum = 0.0;
            int first = cl->GetFirst();
            for (int ch = first; ch < first + n_ch; ++ch) {
                auto it = hit_qdc_map.find(ch);
                if (it != hit_qdc_map.end()) {
                    qdc_sum += it->second;
                }
            }
            s.cluster_qdc.push_back(qdc_sum);
            s.scifi_cluster_sum_qdc += qdc_sum;
        }
    }

    // 3. MuFilter Hits
    if (mufiHits) {
        int n_mf = mufiHits->GetEntries();
        for (int i = 0; i < n_mf; ++i) {
            auto* h = static_cast<MuFilterHit*>(mufiHits->At(i));
            if (!h || !h->isValid()) continue;
            s.n_mufi_hits += 1.0;

            int sys = h->GetSystem(); // 1=Veto, 2=US, 3=DS
            if (sys == 1)      s.n_mufi_veto_hits += 1.0;
            else if (sys == 2) s.n_mufi_us_hits   += 1.0;
            else if (sys == 3) s.n_mufi_ds_hits   += 1.0;

            std::map<Int_t, Float_t> sigs = h->GetAllSignals(true, true, false);
            for (const auto& kv : sigs) {
                s.mufi_sum_qdc += kv.second;
            }
        }
    }

    // 4. Tracks & Clean Single-Track Observables
    if (tracks) {
        int n_tr = tracks->GetEntries();
        s.n_tracks = static_cast<double>(n_tr);

        sndRecoTrack* best_track = nullptr;
        float best_chi2 = 1e9f;

        for (int i = 0; i < n_tr; ++i) {
            auto* trk = static_cast<sndRecoTrack*>(tracks->At(i));
            if (!trk || !trk->getTrackFlag()) continue;
            float c2 = trk->getChi2Ndf();
            if (c2 >= 0.0f) {
                s.track_chi2.push_back(static_cast<double>(c2));
                s.track_slope_xz.push_back(static_cast<double>(trk->getSlopeXZ()));
                s.track_slope_yz.push_back(static_cast<double>(trk->getSlopeYZ()));

                if (c2 < best_chi2) {
                    best_chi2 = c2;
                    best_track = trk;
                }
            }
        }

        // Clean track selection: best track with chi2/ndf < 2.0
        if (best_track && best_chi2 < 2.0f && scifiHits) {
            s.has_clean_track = true;
            s.track_chi2_ndf_clean = static_cast<double>(best_chi2);
            s.track_slope_xz_clean = best_track->getSlopeXZ();
            s.track_slope_yz_clean = best_track->getSlopeYZ();

            TVector3 start = best_track->getStart();
            TVector3 mom   = best_track->getTrackMom();
            double pz = (std::abs(mom.Z()) > 1e-10) ? mom.Z() : 1e-10;

            // Extrapolate to z = 300 cm (front of SciFi tracker)
            double t300 = (300.0 - start.Z()) / pz;
            s.track_x_300 = start.X() + t300 * mom.X();
            s.track_y_300 = start.Y() + t300 * mom.Y();

            int n_sf = scifiHits->GetEntries();
            s.hit_qdc.reserve(n_sf);
            s.hit_distance.reserve(n_sf);
            s.hit_station.reserve(n_sf);
            s.hit_is_vertical.reserve(n_sf);

            for (int i = 0; i < n_sf; ++i) {
                auto* h = static_cast<sndScifiHit*>(scifiHits->At(i));
                if (!h || !h->isValid()) continue;

                double qdc = h->GetSignal(0);
                int det_id = h->GetDetectorID();
                int st = h->GetStation();
                bool is_vert = h->isVertical();

                double dist = 0.0;
                auto it = fChannelMap.find(det_id);
                if (it != fChannelMap.end()) {
                    const auto& cg = it->second;
                    double t = (cg.z_mid - start.Z()) / pz;
                    double x_pos = start.X() + t * mom.X();
                    double y_pos = start.Y() + t * mom.Y();

                    if (cg.is_vertical) {
                        dist = std::abs(cg.sipm_y - y_pos);
                    } else {
                        dist = std::abs(cg.sipm_x - x_pos);
                    }
                } else {
                    // Fallback using nominal geometry
                    if (st < 1) st = 1;
                    if (st > 5) st = 5;
                    double z_st = 300.0 + (st - 1) * 8.5;
                    double t = (z_st - start.Z()) / pz;
                    double x_pos = start.X() + t * mom.X();
                    double y_pos = start.Y() + t * mom.Y();

                    dist = is_vert ? std::abs(54.5 - y_pos) : std::abs(-46.0 - x_pos);
                }

                s.hit_qdc.push_back(qdc);
                s.hit_distance.push_back(dist);
                s.hit_station.push_back(static_cast<double>(st));
                s.hit_is_vertical.push_back(is_vert ? 1.0 : 0.0);

                if (!is_vert) {
                    s.hit_qdc_horiz.push_back(qdc);
                    s.hit_dist_horiz.push_back(dist);
                } else {
                    s.hit_qdc_vert.push_back(qdc);
                    s.hit_dist_vert.push_back(dist);
                }
            }
        }
    }

    return s;
}

} // namespace snd::trident
