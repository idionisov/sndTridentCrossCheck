#ifndef DIGI_VALIDATION_PROCESSOR_H
#define DIGI_VALIDATION_PROCESSOR_H

#include <vector>
#include <unordered_map>
#include "TClonesArray.h"
#include "TVector3.h"
#include "Scifi.h"
#include "MuFilter.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"
#include "sndCluster.h"
#include "sndRecoTrack.h"

namespace snd::trident {

struct DigiValidationConfig {
    double scifi_max_dist{0.1};   // Max distance for SciFi hits to SciFi track [cm] (1 mm = 0.1 cm)
    double veto_max_dist{3.0};    // Max distance for Veto hits to SciFi track [cm] (3 cm)
    double us_max_dist{3.0};      // Max distance for US hits to DS track [cm] (3 cm)
    double ds_max_dist{0.3};      // Max distance for DS hits to DS track [cm] (3 mm = 0.3 cm)
    int scifi_track_type{11};     // SciFi track type (11 = Hough SciFi)
    int ds_track_type{13};        // DS track type (13 = Hough DS)
    double chi2_max{20.0};        // Max chi2/ndf
};

struct ChannelGeometry {
    float xA{0.0f};
    float yA{0.0f};
    float zA{0.0f};
    float xB{0.0f};
    float yB{0.0f};
    float zB{0.0f};
    float sipm_x{0.0f};
    float sipm_y{0.0f};
    float z_mid{0.0f};
    bool is_vertical{false};
};

struct DigiValidationSummary {
    // Event-level scalar quantities (track-associated)
    double n_scifi_hits{0.0};
    double n_scifi_hits_all{0.0};
    double n_scifi_st1{0.0};
    double n_scifi_st2{0.0};
    double n_scifi_st3{0.0};
    double n_scifi_st4{0.0};
    double n_scifi_st5{0.0};
    double scifi_sum_qdc{0.0};
    double scifi_mean_qdc{0.0};

    double n_scifi_clusters{0.0};
    std::vector<double> cluster_size;
    std::vector<double> cluster_qdc;
    double scifi_cluster_sum_qdc{0.0};

    double n_mufi_hits{0.0};
    double n_mufi_hits_all{0.0};
    double n_mufi_veto_hits{0.0};
    double n_mufi_us_hits{0.0};
    double n_mufi_ds_hits{0.0};
    double mufi_sum_qdc{0.0};

    double n_tracks{0.0};
    bool has_scifi_track{false};
    bool has_ds_track{false};
    bool has_both_tracks{false};
    bool has_clean_track{false};
    double track_chi2_ndf_clean{0.0};
    double track_slope_xz_clean{0.0};
    double track_slope_yz_clean{0.0};
    double track_x_300{0.0};
    double track_y_300{0.0};

    double ds_track_chi2_ndf{0.0};
    double ds_track_slope_xz{0.0};
    double ds_track_slope_yz{0.0};

    double event_weight{1.0};

    // Vector observables for SciFi hit QDC vs distance to SiPM (for hits passing cut)
    std::vector<double> hit_qdc;
    std::vector<double> hit_distance;
    std::vector<double> hit_station;
    std::vector<double> hit_is_vertical;

    std::vector<double> hit_qdc_horiz;
    std::vector<double> hit_dist_horiz;

    std::vector<double> hit_qdc_vert;
    std::vector<double> hit_dist_vert;

    // Multi-track observables
    std::vector<double> track_chi2;
    std::vector<double> track_slope_xz;
    std::vector<double> track_slope_yz;

    // Multiplicity & QDC per station
    std::vector<double> station_numbers{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> station_hits{0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> station_qdc{0.0, 0.0, 0.0, 0.0, 0.0};

    // Track-to-channel distance distributions
    std::vector<double> dist_scifi;
    std::vector<double> dist_scifi_horiz;
    std::vector<double> dist_scifi_vert;
    std::vector<double> doca_scifi;

    std::vector<double> dist_veto;
    std::vector<double> doca_veto;

    std::vector<double> dist_us;
    std::vector<double> doca_us;

    std::vector<double> dist_ds;
    std::vector<double> doca_ds;
};

class DigiValidationProcessor {
public:
    DigiValidationProcessor(
        Scifi* scifi = nullptr,
        MuFilter* mufi = nullptr,
        const DigiValidationConfig& config = DigiValidationConfig()
    );
    ~DigiValidationProcessor() = default;

    DigiValidationProcessor(const DigiValidationProcessor&) = default;
    DigiValidationProcessor& operator=(const DigiValidationProcessor&) = default;
    DigiValidationProcessor(DigiValidationProcessor&&) = default;
    DigiValidationProcessor& operator=(DigiValidationProcessor&&) = default;

    void initCache(Scifi* scifi, MuFilter* mufi = nullptr);
    size_t getCachedChannelCount() const { return fChannelMap.size(); }
    size_t getCachedMufiChannelCount() const { return fMufiChannelMap.size(); }

    void setConfig(const DigiValidationConfig& config) { fConfig = config; }
    const DigiValidationConfig& getConfig() const { return fConfig; }

    float computeDistToChannel(sndRecoTrack* trk, int detID) const;
    float computeDoca(sndRecoTrack* trk, int detID) const;
    float computeDistToChannel(sndRecoTrack* trk, const sndScifiHit* hit) const;
    float computeDistToChannel(sndRecoTrack* trk, const MuFilterHit* hit) const;
    float computeDoca(sndRecoTrack* trk, const sndScifiHit* hit) const;
    float computeDoca(sndRecoTrack* trk, const MuFilterHit* hit) const;

    DigiValidationSummary process(
        const TClonesArray* scifiHits,
        const TClonesArray* clusters,
        const TClonesArray* mufiHits,
        const TClonesArray* tracks,
        double event_weight = 1.0
    ) const;

    // Single unambiguous call operator for ROOT RDataFrame
    DigiValidationSummary operator()(
        const TClonesArray& scifiHits,
        const TClonesArray& clusters,
        const TClonesArray& mufiHits,
        const TClonesArray& tracks
    ) const {
        return process(&scifiHits, &clusters, &mufiHits, &tracks, 1.0);
    }

private:
    void findBestTracks(
        const TClonesArray* tracks,
        sndRecoTrack*& bestScifiTrack,
        sndRecoTrack*& bestDsTrack
    ) const;

    bool getChannelGeometry(int detID, TVector3& left, TVector3& right, bool& isVertical) const;

    DigiValidationConfig fConfig;
    Scifi* fScifi{nullptr};
    MuFilter* fMuFilter{nullptr};
    mutable std::unordered_map<int, ChannelGeometry> fChannelMap;
    mutable std::unordered_map<int, ChannelGeometry> fMufiChannelMap;
    bool fHasGeometryCache{false};
};

} // namespace snd::trident

#endif // DIGI_VALIDATION_PROCESSOR_H
