#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include "TClonesArray.h"
#include "TVector3.h"
#include "sndRecoTrack.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"
#include "SNDLHCEventHeader.h"
#include "TSeqCollection.h"
#include "TObjArray.h"
#include <algorithm>
#include "TROOT.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "Scifi.h"
#include "MuFilter.h"

namespace snd::trident {

struct IP1Filter {
    bool operator()(const SNDLHCEventHeader& header) const {
        return const_cast<SNDLHCEventHeader&>(header).isIP1();
    }
};



class BaseSpatialAnisotropyCalculator {
    protected:
        mutable std::unordered_map<int, TVector3> fPositions;
        virtual void extractPositions(const TClonesArray& hits, std::vector<TVector3>& out_positions) const = 0;

    public:
        virtual ~BaseSpatialAnisotropyCalculator() = default;

        double calculateAnisotropy(const std::vector<TVector3>& positions) const {
            if (positions.size() < 3) return -1.0;

            TVector3 mean(0., 0., 0.);
            for (const auto& pos : positions) mean += pos;
            mean *= (1.0 / positions.size());

            TMatrixDSym cov(3);
            cov.Zero();
            for (const auto& pos : positions) {
                const TVector3 d = pos - mean;
                cov(0, 0) += d.X() * d.X();
                cov(0, 1) += d.X() * d.Y();
                cov(0, 2) += d.X() * d.Z();
                cov(1, 1) += d.Y() * d.Y();
                cov(1, 2) += d.Y() * d.Z();
                cov(2, 2) += d.Z() * d.Z();
            }
            cov(1, 0) = cov(0, 1);
            cov(2, 0) = cov(0, 2);
            cov(2, 1) = cov(1, 2);
            cov *= (1.0 / positions.size());

            const TVectorD ev = TMatrixDSymEigen(cov).GetEigenValues();
            std::vector<double> sorted = {ev[0], ev[1], ev[2]};
            std::sort(sorted.begin(), sorted.end(), std::greater<double>());

            const double tot_var = sorted[0] + sorted[1] + sorted[2];
            if (tot_var <= 0.0) return -1.0;

            return sorted[0] / tot_var;
        }

        /// Functor call operator
        double operator()(const TClonesArray& hits) const {
            const int n_hits = hits.GetEntries();
            if (n_hits < 3) return -1.0;

            std::vector<TVector3> positions;
            positions.reserve(n_hits);

            extractPositions(hits, positions);
            return calculateAnisotropy(positions);
        }
    };


class SciFiAnisotropyCalculator : public BaseSpatialAnisotropyCalculator {
    private:
        Scifi* fScifiDet{nullptr};

        void initCache() {
            if (!fScifiDet) return;
            TVector3 a, b;
            fPositions.reserve(16000);

            for (int station = 1; station <= 5; ++station) {
                for (int plane : {0, 100000}) {
                    for (int mat = 0; mat < 4; ++mat) {
                        for (int sipm = 0; sipm < 4; ++sipm) {
                            for (int ch = 0; ch < 128; ++ch) {
                                int det_id = station * 1000000 + plane + mat * 10000 + sipm * 1000 + ch;
                                try {
                                    fScifiDet->GetSiPMPosition(det_id, a, b);
                                    fPositions[det_id] = 0.5 * (a + b);
                                } catch (...) {}
                            }
                        }
                    }
                }
            }
        }

    protected:
        void extractPositions(
            const TClonesArray& hits,
            std::vector<TVector3>& out_positions
        ) const override {
            const int n_hits = hits.GetEntries();
            for (int i = 0; i < n_hits; ++i) {
                auto* hit = static_cast<sndScifiHit*>(hits.At(i));
                if (!hit || !hit->isValid()) continue;

                auto it = fPositions.find(hit->GetDetectorID());
                if (it != fPositions.end()) {
                    out_positions.push_back(it->second);
                }
            }
        }

    public:
        SciFiAnisotropyCalculator(Scifi* scifi = nullptr) : fScifiDet(scifi) {
            if (!fScifiDet && gROOT && gROOT->GetListOfGlobals()) {
                fScifiDet = dynamic_cast<Scifi*>(gROOT->GetListOfGlobals()->FindObject("Scifi"));
            }
            if (fScifiDet) {
                initCache();
            }
        }
    };




class MuFilterAnisotropyCalculator : public BaseSpatialAnisotropyCalculator {
    private:
        MuFilter* fMuFilterDet{nullptr};
        int fTargetSystem{2}; // 1 = Veto, 2 = US, 3 = DS, -1 = All

        inline void cacheChannel(int det_id) const {
            if (!fMuFilterDet) return;
            if (fPositions.find(det_id) != fPositions.end()) return;

            TVector3 L, R;
            try {
                fMuFilterDet->GetPosition(det_id, L, R);
                fPositions[det_id] = 0.5 * (L + R);
            } catch (...) {}
        }

        void initCache() {
            if (!fMuFilterDet) return;

            if (fTargetSystem == -1)     fPositions.reserve(600);
            else if (fTargetSystem == 3) fPositions.reserve(500);
            else if (fTargetSystem == 2) fPositions.reserve(70);
            else if (fTargetSystem == 1) fPositions.reserve(30);

            // Veto (Planes 0 and 1, 7 bars each: 10000..10006, 11000..11006; Plane 2 if present)
            if (fTargetSystem == 1 || fTargetSystem == -1) {
                for (int p = 0; p < 3; ++p)
                    for (int b = 0; b < 7; ++b)
                        cacheChannel(10000 + p * 1000 + b);
            }
            // US (Stations 0 to 4, 10 bars each: 20000..24009)
            if (fTargetSystem == 2 || fTargetSystem == -1) {
                for (int s = 0; s < 5; ++s)
                    for (int b = 0; b < 10; ++b)
                        cacheChannel(20000 + s * 1000 + b);
            }
            // DS (Stations 0-2: 60 hor [0..59] + 60 ver [60..119]; Station 3: 60 ver [60..119])
            if (fTargetSystem == 3 || fTargetSystem == -1) {
                for (int s = 0; s < 3; ++s) {
                    for (int b = 0; b < 120; ++b)
                        cacheChannel(30000 + s * 1000 + b);
                }
                for (int b = 60; b < 120; ++b) {
                    cacheChannel(33000 + b);
                }
            }
        }

    protected:
        void extractPositions(
            const TClonesArray& hits,
            std::vector<TVector3>& out_positions
        ) const override {
            const int n_hits = hits.GetEntries();
            for (int i = 0; i < n_hits; ++i) {
                auto* hit = static_cast<MuFilterHit*>(hits.At(i));
                if (!hit || !hit->isValid()) continue;

                if (fTargetSystem != -1 && hit->GetSystem() != fTargetSystem) continue;

                const int det_id = hit->GetDetectorID();
                auto it = fPositions.find(det_id);

                if (it != fPositions.end()) {
                    out_positions.push_back(it->second);
                } else if (fMuFilterDet) {
                    cacheChannel(det_id);
                    it = fPositions.find(det_id);
                    if (it != fPositions.end()) {
                        out_positions.push_back(it->second);
                    }
                }
            }
        }

    public:
        MuFilterAnisotropyCalculator(MuFilter* mufilter = nullptr, int target_system = 2)
            : fMuFilterDet(mufilter), fTargetSystem(target_system)
        {
            if (!fMuFilterDet && gROOT && gROOT->GetListOfGlobals()) {
                fMuFilterDet = dynamic_cast<MuFilter*>(
                    gROOT->GetListOfGlobals()->FindObject("MuFilter")
                );
                if (!fMuFilterDet) {
                    fMuFilterDet = dynamic_cast<MuFilter*>(
                        gROOT->GetListOfGlobals()->FindObject("MuFi")
                    );
                }
            }
            if (fMuFilterDet) {
                initCache();
            }
        }

        void setTargetSystem(int target_system) {
            fTargetSystem = target_system;
            initCache();
        }
    };


struct MuonCalibrationConfig {
    double chi2_max{20.0};          // General / fallback chi2/ndf threshold
    double chi2_max_scifi{20.0};    // SciFi track chi2/ndf upper limit
    double chi2_max_ds{20.0};       // DS track chi2/ndf upper limit
    double max_slope{0.05};         // Max angular slope (< 0.05 rad)
    double fiducial_margin{1.5};    // cm margin from detector borders
    int scifi_track_type{11};       // SciFi track type (11 = SciFi Hough)
    int ds_track_type{13};          // DS track type (13 = DS Hough)
    double z_match{430.0};          // Fiducial plane Z position [cm]
    double pos_match_max{3.0};      // Max transverse distance at z_match [cm]
    double angle_match_max{0.015};  // Max angle difference [rad]
    int scifi_hits_min{10};         // Typical single MIP across 10 planes
    int scifi_hits_max{35};
    int ds_hits_min{2};             // Muon filter penetration (DS)
    double ds_match_slope_max{0.04};// Max slope difference between SF and DS tracks
    double scifi_qdc_min{0.0};      // Offline SciFi threshold in p.e./QDC (0.0 = use isValid())
};

struct MuonCalibrationMetrics {
    // 1. Classification & Quality Cuts
    bool is_clean{false};
    bool pass_cut_single_scifi_track{false};
    bool pass_cut_ds_track{false};
    bool pass_cut_chi2_scifi{false};
    bool pass_cut_chi2_ds{false};
    bool pass_cut_chi2{false};
    bool pass_cut_slope_scifi{false};
    bool pass_cut_slope_ds{false};
    bool pass_cut_slope{false};
    bool pass_cut_fiducial_430{false};
    bool pass_cut_fiducial{false};
    bool pass_cut_pos_match_430{false};
    bool pass_cut_angle_match{false};
    bool pass_cut_ds_match{false};
    bool pass_cut_scifi_hits{false};
    bool pass_cut_ds_hits{false};
    int n_tracks{0};
    int n_scifi_tracks{0};
    int n_ds_tracks{0};

    // SciFi Track properties
    float track_chi2_ndf{-1.0f};
    float track_chi2{0.0f};
    int track_ndf{0};
    bool track_flag{false};
    int track_type{0};
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

    // DS Track properties
    float ds_track_chi2_ndf{-1.0f};
    float ds_track_chi2{0.0f};
    int ds_track_ndf{0};
    bool ds_track_flag{false};
    int ds_track_type{0};
    float ds_track_slope_xz{0.0f};
    float ds_track_slope_yz{0.0f};
    float ds_track_angle_xz{0.0f};
    float ds_track_angle_yz{0.0f};

    // DS & SF Matching Kinematics
    float diff_slope_xz{0.0f};
    float diff_slope_yz{0.0f};
    float diff_angle_xz{0.0f};
    float diff_angle_yz{0.0f};
    float diff_angle_3d{0.0f};
    float scifi_x_430{0.0f};
    float scifi_y_430{0.0f};
    float ds_x_430{0.0f};
    float ds_y_430{0.0f};
    float diff_x_430{0.0f};
    float diff_y_430{0.0f};
    float diff_pos_430{0.0f};

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
    double veto_sum_qdc{0.0};
    int us_nhits{0};
    double us_sum_qdc{0.0};

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
    TVector3 extrapolateToZ(sndRecoTrack* track, double z) const;

private:
    MuonCalibrationConfig fConfig;
};

} // namespace snd::trident
