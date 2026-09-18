#include "DigiValidationProcessor.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include "TROOT.h"
#include "TError.h"

namespace snd::trident {

DigiValidationProcessor::DigiValidationProcessor(
    Scifi* scifi,
    MuFilter* mufi,
    const DigiValidationConfig& config
) : fConfig(config), fScifi(scifi), fMuFilter(mufi) {
    if (scifi || mufi) {
        initCache(scifi, mufi);
    }
}

void DigiValidationProcessor::initCache(Scifi* scifi, MuFilter* mufi) {
    if (scifi) {
        fScifi = scifi;
    } else if (!fScifi && gROOT && gROOT->GetListOfGlobals()) {
        fScifi = dynamic_cast<Scifi*>(gROOT->GetListOfGlobals()->FindObject("Scifi"));
    }

    if (mufi) {
        fMuFilter = mufi;
    } else if (!fMuFilter && gROOT && gROOT->GetListOfGlobals()) {
        fMuFilter = dynamic_cast<MuFilter*>(gROOT->GetListOfGlobals()->FindObject("MuFilter"));
    }

    // Suppress TGeoNavigator warnings during full volume traversal
    int prevErrorIgnore = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kFatal;

    // 1. Cache SciFi channel geometries
    if (fScifi) {
        fChannelMap.clear();
        fChannelMap.reserve(15360);

        TVector3 left, right;
        // Iterate over stations 1..5, planes 0..1 (horizontal=0, vertical=1)
        for (int station = 1; station <= 5; ++station) {
            for (int plane = 0; plane <= 1; ++plane) {
                bool is_vert = (plane == 1);
                // 4 mats per plane
                for (int mat = 0; mat < 4; ++mat) {
                    // 3 SiPMs per mat, 128 channels each
                    for (int sipm = 0; sipm < 3; ++sipm) {
                        for (int ch = 0; ch < 128; ++ch) {
                            int channel_id = sipm * 128 + ch;
                            int det_id = station * 1000000 + plane * 100000 + mat * 10000 + channel_id;

                            try {
                                left.SetXYZ(-9999.0, -9999.0, -9999.0);
                                right.SetXYZ(-9999.0, -9999.0, -9999.0);
                                fScifi->GetSiPMPosition(det_id, left, right);
                                if (left.X() != -9999.0) {
                                    ChannelGeometry cg;
                                    cg.xA = static_cast<float>(left.X());
                                    cg.yA = static_cast<float>(left.Y());
                                    cg.zA = static_cast<float>(left.Z());
                                    cg.xB = static_cast<float>(right.X());
                                    cg.yB = static_cast<float>(right.Y());
                                    cg.zB = static_cast<float>(right.Z());
                                    cg.sipm_x = cg.xA;
                                    cg.sipm_y = cg.yA;
                                    cg.z_mid = 0.5f * (cg.zA + cg.zB);
                                    cg.is_vertical = is_vert;

                                    fChannelMap[det_id] = cg;
                                }
                            } catch (...) {
                                continue;
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Cache MuFilter channel geometries
    if (fMuFilter) {
        fMufiChannelMap.clear();
        fMufiChannelMap.reserve(600);
        TVector3 left, right;

        // Veto: up to 3 planes (0, 1, 2), up to 10 bars each (10000 + p*1000 + b)
        for (int p = 0; p < 3; ++p) {
            for (int b = 0; b < 10; ++b) {
                int det_id = 10000 + p * 1000 + b;
                try {
                    left.SetXYZ(-9999.0, -9999.0, -9999.0);
                    right.SetXYZ(-9999.0, -9999.0, -9999.0);
                    fMuFilter->GetPosition(det_id, left, right);
                    if (left.X() != -9999.0) {
                        ChannelGeometry cg;
                        cg.xA = static_cast<float>(left.X());
                        cg.yA = static_cast<float>(left.Y());
                        cg.zA = static_cast<float>(left.Z());
                        cg.xB = static_cast<float>(right.X());
                        cg.yB = static_cast<float>(right.Y());
                        cg.zB = static_cast<float>(right.Z());
                        cg.z_mid = 0.5f * (cg.zA + cg.zB);
                        cg.is_vertical = (std::abs(right.Y() - left.Y()) > std::abs(right.X() - left.X()));
                        fMufiChannelMap[det_id] = cg;
                    }
                } catch (...) {}
            }
        }

        // US: 5 planes (0..4), up to 10 bars each (20000 + p*1000 + b)
        for (int p = 0; p < 5; ++p) {
            for (int b = 0; b < 10; ++b) {
                int det_id = 20000 + p * 1000 + b;
                try {
                    left.SetXYZ(-9999.0, -9999.0, -9999.0);
                    right.SetXYZ(-9999.0, -9999.0, -9999.0);
                    fMuFilter->GetPosition(det_id, left, right);
                    if (left.X() != -9999.0) {
                        ChannelGeometry cg;
                        cg.xA = static_cast<float>(left.X());
                        cg.yA = static_cast<float>(left.Y());
                        cg.zA = static_cast<float>(left.Z());
                        cg.xB = static_cast<float>(right.X());
                        cg.yB = static_cast<float>(right.Y());
                        cg.zB = static_cast<float>(right.Z());
                        cg.z_mid = 0.5f * (cg.zA + cg.zB);
                        cg.is_vertical = (std::abs(right.Y() - left.Y()) > std::abs(right.X() - left.X()));
                        fMufiChannelMap[det_id] = cg;
                    }
                } catch (...) {}
            }
        }

        // DS: 4 planes (0..3), up to 120 bars each (30000 + p*1000 + b)
        for (int p = 0; p < 4; ++p) {
            for (int b = 0; b < 120; ++b) {
                int det_id = 30000 + p * 1000 + b;
                try {
                    left.SetXYZ(-9999.0, -9999.0, -9999.0);
                    right.SetXYZ(-9999.0, -9999.0, -9999.0);
                    fMuFilter->GetPosition(det_id, left, right);
                    if (left.X() != -9999.0) {
                        ChannelGeometry cg;
                        cg.xA = static_cast<float>(left.X());
                        cg.yA = static_cast<float>(left.Y());
                        cg.zA = static_cast<float>(left.Z());
                        cg.xB = static_cast<float>(right.X());
                        cg.yB = static_cast<float>(right.Y());
                        cg.zB = static_cast<float>(right.Z());
                        cg.z_mid = 0.5f * (cg.zA + cg.zB);
                        cg.is_vertical = (std::abs(right.Y() - left.Y()) > std::abs(right.X() - left.X()));
                        fMufiChannelMap[det_id] = cg;
                    }
                } catch (...) {}
            }
        }
    }

    gErrorIgnoreLevel = prevErrorIgnore;
    fHasGeometryCache = !fChannelMap.empty();
}

bool DigiValidationProcessor::getChannelGeometry(int detID, TVector3& left, TVector3& right, bool& isVertical) const {
    if (detID >= 100000) {
        // SciFi channel: strictly read from pre-cached geometry
        auto it = fChannelMap.find(detID);
        if (it != fChannelMap.end()) {
            const auto& cg = it->second;
            left.SetXYZ(cg.xA, cg.yA, cg.zA);
            right.SetXYZ(cg.xB, cg.yB, cg.zB);
            isVertical = cg.is_vertical;
            return true;
        }
        return false;
    } else {
        // MuFilter channel: strictly read from pre-cached geometry
        auto it = fMufiChannelMap.find(detID);
        if (it != fMufiChannelMap.end()) {
            const auto& cg = it->second;
            left.SetXYZ(cg.xA, cg.yA, cg.zA);
            right.SetXYZ(cg.xB, cg.yB, cg.zB);
            isVertical = cg.is_vertical;
            return true;
        }
        return false;
    }
}

float DigiValidationProcessor::computeDistToChannel(sndRecoTrack* trk, int detID) const {
    if (!trk) return std::numeric_limits<float>::quiet_NaN();

    TVector3 left, right;
    bool isVertical = false;
    if (!getChannelGeometry(detID, left, right, isVertical)) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    float z_channel = 0.5f * (left.Z() + right.Z());
    TVector3 start = trk->getStart();
    TVector3 mom = trk->getTrackMom();
    float pz = (std::abs(mom.Z()) > 1e-10f) ? static_cast<float>(mom.Z()) : 1e-10f;
    float t = (z_channel - static_cast<float>(start.Z())) / pz;
    float x_track = static_cast<float>(start.X()) + t * static_cast<float>(mom.X());
    float y_track = static_cast<float>(start.Y()) + t * static_cast<float>(mom.Y());

    if (isVertical) {
        float x_channel = 0.5f * (static_cast<float>(left.X()) + static_cast<float>(right.X()));
        return std::abs(x_track - x_channel);
    } else {
        float y_channel = 0.5f * (static_cast<float>(left.Y()) + static_cast<float>(right.Y()));
        return std::abs(y_track - y_channel);
    }
}

float DigiValidationProcessor::computeDoca(sndRecoTrack* trk, int detID) const {
    if (!trk) return std::numeric_limits<float>::quiet_NaN();

    TVector3 left, right;
    bool isVertical = false;
    if (!getChannelGeometry(detID, left, right, isVertical)) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    TVector3 pos = trk->getStart();
    TVector3 mom = trk->getTrackMom();
    TVector3 pq = left - pos;
    TVector3 uCrossv = (right - left).Cross(mom);
    double mag = uCrossv.Mag();
    if (mag < 1e-10) return 999.0f;

    double doca = pq.Dot(uCrossv) / mag;
    return static_cast<float>(std::abs(doca));
}

float DigiValidationProcessor::computeDistToChannel(sndRecoTrack* trk, const sndScifiHit* hit) const {
    return hit ? computeDistToChannel(trk, hit->GetDetectorID()) : std::numeric_limits<float>::quiet_NaN();
}

float DigiValidationProcessor::computeDistToChannel(sndRecoTrack* trk, const MuFilterHit* hit) const {
    return hit ? computeDistToChannel(trk, hit->GetDetectorID()) : std::numeric_limits<float>::quiet_NaN();
}

float DigiValidationProcessor::computeDoca(sndRecoTrack* trk, const sndScifiHit* hit) const {
    return hit ? computeDoca(trk, hit->GetDetectorID()) : std::numeric_limits<float>::quiet_NaN();
}

float DigiValidationProcessor::computeDoca(sndRecoTrack* trk, const MuFilterHit* hit) const {
    return hit ? computeDoca(trk, hit->GetDetectorID()) : std::numeric_limits<float>::quiet_NaN();
}

void DigiValidationProcessor::findBestTracks(
    const TClonesArray* tracks,
    sndRecoTrack*& bestScifiTrack,
    sndRecoTrack*& bestDsTrack
) const {
    bestScifiTrack = nullptr;
    bestDsTrack = nullptr;
    float bestScifiChi2 = 1e9f;
    float bestDsChi2 = 1e9f;

    if (!tracks) return;
    int n = tracks->GetEntries();
    for (int i = 0; i < n; ++i) {
        auto* trk = dynamic_cast<sndRecoTrack*>(tracks->At(i));
        if (!trk) continue;
        int t = trk->getTrackType();
        float c2 = trk->getChi2Ndf();
        bool flag = trk->getTrackFlag();

        // SciFi track: match configured type (default 11) or fallback 1
        if (t == fConfig.scifi_track_type || (t == 1 && !bestScifiTrack)) {
            if (flag && c2 < bestScifiChi2) {
                bestScifiChi2 = c2;
                bestScifiTrack = trk;
            } else if (!bestScifiTrack) {
                bestScifiTrack = trk;
                bestScifiChi2 = c2;
            }
        }
        // DS track: match configured type (default 13) or fallback 3
        if (t == fConfig.ds_track_type || (t == 3 && !bestDsTrack)) {
            if (flag && c2 < bestDsChi2) {
                bestDsChi2 = c2;
                bestDsTrack = trk;
            } else if (!bestDsTrack) {
                bestDsTrack = trk;
                bestDsChi2 = c2;
            }
        }
    }
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

    // 1. Identify best tracks
    sndRecoTrack* scifiTrack = nullptr;
    sndRecoTrack* dsTrack    = nullptr;
    findBestTracks(tracks, scifiTrack, dsTrack);

    if (tracks) {
        int n_tr = tracks->GetEntries();
        s.n_tracks = static_cast<double>(n_tr);
        for (int i = 0; i < n_tr; ++i) {
            auto* trk = static_cast<sndRecoTrack*>(tracks->At(i));
            if (!trk || !trk->getTrackFlag()) continue;
            float c2 = trk->getChi2Ndf();
            if (c2 >= 0.0f) {
                s.track_chi2.push_back(static_cast<double>(c2));
                s.track_slope_xz.push_back(static_cast<double>(trk->getSlopeXZ()));
                s.track_slope_yz.push_back(static_cast<double>(trk->getSlopeYZ()));
            }
        }
    }

    s.has_scifi_track = (scifiTrack != nullptr);
    s.has_ds_track    = (dsTrack != nullptr);
    s.has_both_tracks = (scifiTrack != nullptr && dsTrack != nullptr);

    if (scifiTrack) {
        float c2 = scifiTrack->getChi2Ndf();
        if (c2 >= 0.0f && c2 < fConfig.chi2_max) {
            s.has_clean_track = true;
            s.track_chi2_ndf_clean = static_cast<double>(c2);
            s.track_slope_xz_clean = scifiTrack->getSlopeXZ();
            s.track_slope_yz_clean = scifiTrack->getSlopeYZ();

            TVector3 start = scifiTrack->getStart();
            TVector3 mom   = scifiTrack->getTrackMom();
            double pz = (std::abs(mom.Z()) > 1e-10) ? mom.Z() : 1e-10;
            double t300 = (300.0 - start.Z()) / pz;
            s.track_x_300 = start.X() + t300 * mom.X();
            s.track_y_300 = start.Y() + t300 * mom.Y();
        }
    }
    if (dsTrack) {
        s.ds_track_chi2_ndf = static_cast<double>(dsTrack->getChi2Ndf());
        s.ds_track_slope_xz = static_cast<double>(dsTrack->getSlopeXZ());
        s.ds_track_slope_yz = static_cast<double>(dsTrack->getSlopeYZ());
    }

    // 2. SciFi Hits: Distance from SciFi track & Selection cut
    std::unordered_map<int, double> hit_qdc_map;
    if (scifiHits) {
        int n_sf = scifiHits->GetEntries();
        s.n_scifi_hits_all = static_cast<double>(n_sf);
        hit_qdc_map.reserve(n_sf);

        TVector3 trk_start, trk_mom;
        double trk_pz = 1.0;
        if (scifiTrack) {
            trk_start = scifiTrack->getStart();
            trk_mom   = scifiTrack->getTrackMom();
            trk_pz    = (std::abs(trk_mom.Z()) > 1e-10) ? trk_mom.Z() : 1e-10;
        }

        for (int i = 0; i < n_sf; ++i) {
            auto* h = static_cast<sndScifiHit*>(scifiHits->At(i));
            if (!h || !h->isValid()) continue;

            double qdc = h->GetSignal(0);
            int det_id = h->GetDetectorID();
            int st = h->GetStation();
            bool is_vert = h->isVertical();

            // Measure distance from SciFi track to activated channel
            float dist = 999.0f;
            float doca = 999.0f;
            if (scifiTrack) {
                dist = computeDistToChannel(scifiTrack, det_id);
                doca = computeDoca(scifiTrack, det_id);

                if (!std::isnan(dist) && !std::isinf(dist)) {
                    s.dist_scifi.push_back(static_cast<double>(dist));
                    if (is_vert) {
                        s.dist_scifi_vert.push_back(static_cast<double>(dist));
                    } else {
                        s.dist_scifi_horiz.push_back(static_cast<double>(dist));
                    }
                }
                if (!std::isnan(doca) && !std::isinf(doca)) {
                    s.doca_scifi.push_back(static_cast<double>(doca));
                }
            }

            // Cut: Only keep SciFi hits close to the SciFi track (e.g. <= 1 mm = 0.1 cm)
            bool pass_hit_cut = scifiTrack && (!std::isnan(dist) && dist <= fConfig.scifi_max_dist);
            if (pass_hit_cut) {
                s.n_scifi_hits += 1.0;
                s.scifi_sum_qdc += qdc;

                if (st == 1)      s.n_scifi_st1 += 1.0;
                else if (st == 2) s.n_scifi_st2 += 1.0;
                else if (st == 3) s.n_scifi_st3 += 1.0;
                else if (st == 4) s.n_scifi_st4 += 1.0;
                else if (st == 5) s.n_scifi_st5 += 1.0;

                if (st >= 1 && st <= 5) {
                    s.station_hits[st - 1] += 1.0;
                    s.station_qdc[st - 1] += qdc;
                }

                hit_qdc_map[det_id] = qdc;

                // SiPM attenuation distance (distance along fiber from hit position to SiPM readout)
                if (s.has_clean_track) {
                    double dist_to_sipm = 0.0;
                    auto it = fChannelMap.find(det_id);
                    if (it != fChannelMap.end()) {
                        const auto& cg = it->second;
                        double t = (cg.z_mid - trk_start.Z()) / trk_pz;
                        double x_pos = trk_start.X() + t * trk_mom.X();
                        double y_pos = trk_start.Y() + t * trk_mom.Y();

                        if (cg.is_vertical) {
                            dist_to_sipm = std::abs(cg.sipm_y - y_pos);
                        } else {
                            dist_to_sipm = std::abs(cg.sipm_x - x_pos);
                        }
                    } else {
                        // Fallback using nominal geometry
                        int s_idx = std::clamp(st, 1, 5);
                        double z_st = 300.0 + (s_idx - 1) * 8.5;
                        double t = (z_st - trk_start.Z()) / trk_pz;
                        double x_pos = trk_start.X() + t * trk_mom.X();
                        double y_pos = trk_start.Y() + t * trk_mom.Y();

                        dist_to_sipm = is_vert ? std::abs(54.5 - y_pos) : std::abs(-46.0 - x_pos);
                    }

                    s.hit_qdc.push_back(qdc);
                    s.hit_distance.push_back(dist_to_sipm);
                    s.hit_station.push_back(static_cast<double>(st));
                    s.hit_is_vertical.push_back(is_vert ? 1.0 : 0.0);

                    if (!is_vert) {
                        s.hit_qdc_horiz.push_back(qdc);
                        s.hit_dist_horiz.push_back(dist_to_sipm);
                    } else {
                        s.hit_qdc_vert.push_back(qdc);
                        s.hit_dist_vert.push_back(dist_to_sipm);
                    }
                }
            }
        }
        if (s.n_scifi_hits > 0.0) {
            s.scifi_mean_qdc = s.scifi_sum_qdc / s.n_scifi_hits;
        }
    }

    // 3. Clusters: only constructed from track-associated hits
    if (clusters) {
        int n_cl = clusters->GetEntries();
        for (int i = 0; i < n_cl; ++i) {
            auto* cl = static_cast<sndCluster*>(clusters->At(i));
            if (!cl) continue;
            int n_ch = 0;
            double qdc_sum = 0.0;
            int first = cl->GetFirst();
            int orig_n = cl->GetN();

            for (int ch = first; ch < first + orig_n; ++ch) {
                auto it = hit_qdc_map.find(ch);
                if (it != hit_qdc_map.end()) {
                    qdc_sum += it->second;
                    n_ch++;
                }
            }
            if (n_ch > 0) {
                s.n_scifi_clusters += 1.0;
                s.cluster_size.push_back(static_cast<double>(n_ch));
                s.cluster_qdc.push_back(qdc_sum);
                s.scifi_cluster_sum_qdc += qdc_sum;
            }
        }
    }

    // 4. MuFilter Hits: Veto close to SciFi track, US & DS close to DS track
    if (mufiHits) {
        int n_mf = mufiHits->GetEntries();
        s.n_mufi_hits_all = static_cast<double>(n_mf);

        for (int i = 0; i < n_mf; ++i) {
            auto* h = static_cast<MuFilterHit*>(mufiHits->At(i));
            if (!h || !h->isValid()) continue;

            int sys = h->GetSystem(); // 1=Veto, 2=US, 3=DS
            int det_id = h->GetDetectorID();
            double hit_qdc = 0.0;
            std::map<Int_t, Float_t> sigs = h->GetAllSignals(true, true, false);
            for (const auto& kv : sigs) {
                hit_qdc += kv.second;
            }

            if (sys == 1) {
                // Veto hits: check distance to SciFi track
                float dist = 999.0f;
                float doca = 999.0f;
                if (scifiTrack) {
                    dist = computeDistToChannel(scifiTrack, det_id);
                    doca = computeDoca(scifiTrack, det_id);

                    if (!std::isnan(dist) && !std::isinf(dist)) {
                        s.dist_veto.push_back(static_cast<double>(dist));
                    }
                    if (!std::isnan(doca) && !std::isinf(doca)) {
                        s.doca_veto.push_back(static_cast<double>(doca));
                    }
                }

                if (scifiTrack && !std::isnan(dist) && dist <= fConfig.veto_max_dist) {
                    s.n_mufi_veto_hits += 1.0;
                    s.mufi_sum_qdc += hit_qdc;
                }
            } else if (sys == 2) {
                // US hits: check distance to DS track
                float dist = 999.0f;
                float doca = 999.0f;
                if (dsTrack) {
                    dist = computeDistToChannel(dsTrack, det_id);
                    doca = computeDoca(dsTrack, det_id);

                    if (!std::isnan(dist) && !std::isinf(dist)) {
                        s.dist_us.push_back(static_cast<double>(dist));
                    }
                    if (!std::isnan(doca) && !std::isinf(doca)) {
                        s.doca_us.push_back(static_cast<double>(doca));
                    }
                }

                if (dsTrack && !std::isnan(dist) && dist <= fConfig.us_max_dist) {
                    s.n_mufi_us_hits += 1.0;
                    s.mufi_sum_qdc += hit_qdc;
                }
            } else if (sys == 3) {
                // DS hits: check distance to DS track
                float dist = 999.0f;
                float doca = 999.0f;
                if (dsTrack) {
                    dist = computeDistToChannel(dsTrack, det_id);
                    doca = computeDoca(dsTrack, det_id);

                    if (!std::isnan(dist) && !std::isinf(dist)) {
                        s.dist_ds.push_back(static_cast<double>(dist));
                    }
                    if (!std::isnan(doca) && !std::isinf(doca)) {
                        s.doca_ds.push_back(static_cast<double>(doca));
                    }
                }

                if (dsTrack && !std::isnan(dist) && dist <= fConfig.ds_max_dist) {
                    s.n_mufi_ds_hits += 1.0;
                    s.mufi_sum_qdc += hit_qdc;
                }
            }
        }
        s.n_mufi_hits = s.n_mufi_veto_hits + s.n_mufi_us_hits + s.n_mufi_ds_hits;
    }

    return s;
}

} // namespace snd::trident
