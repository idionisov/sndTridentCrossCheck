#ifndef DIGI_VALIDATION_PROCESSOR_H
#define DIGI_VALIDATION_PROCESSOR_H

#include "TClonesArray.h"
#include "sndCluster.h"
#include "sndScifiHit.h"
#include "sndRecoTrack.h"
#include "MuFilterHit.h"
#include "ShipMCTrack.h"
#include "TVector3.h"
#include "Scifi.h"

#include <vector>
#include <unordered_map>
#include <cmath>
#include <string>

namespace snd::trident {

struct ChannelGeometry {
    float xA{0.0f};
    float yA{0.0f};
    float zA{0.0f};
    float xB{0.0f};
    float yB{0.0f};
    float zB{0.0f};
    float z_mid{0.0f};
    float sipm_x{0.0f};
    float sipm_y{0.0f};
    float sipm_z{0.0f};
    bool is_vertical{false};
};

struct DigiValidationSummary {
    // Event-level scalar quantities
    double n_scifi_hits{0.0};
    double n_scifi_st1{0.0};
    double n_scifi_st2{0.0};
    double n_scifi_st3{0.0};
    double n_scifi_st4{0.0};
    double n_scifi_st5{0.0};
    double scifi_sum_qdc{0.0};
    double scifi_mean_qdc{0.0};

    double n_scifi_clusters{0.0};
    double scifi_cluster_sum_qdc{0.0};

    double n_mufi_hits{0.0};
    double n_mufi_veto_hits{0.0};
    double n_mufi_us_hits{0.0};
    double n_mufi_ds_hits{0.0};
    double mufi_sum_qdc{0.0};

    double n_tracks{0.0};
    bool has_clean_track{false};
    double track_chi2_ndf_clean{0.0};
    double track_slope_xz_clean{0.0};
    double track_slope_yz_clean{0.0};
    double track_x_300{0.0};
    double track_y_300{0.0};

    double event_weight{1.0};

    // Vector observables for RDataFrame 1D & 2D projections
    std::vector<double> hit_qdc;
    std::vector<double> hit_distance;
    std::vector<double> hit_station;
    std::vector<double> hit_is_vertical;

    std::vector<double> hit_qdc_horiz;
    std::vector<double> hit_dist_horiz;

    std::vector<double> hit_qdc_vert;
    std::vector<double> hit_dist_vert;

    std::vector<double> cluster_size;
    std::vector<double> cluster_qdc;

    std::vector<double> track_chi2;
    std::vector<double> track_slope_xz;
    std::vector<double> track_slope_yz;

    std::vector<double> station_numbers{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> station_hits{0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> station_qdc{0.0, 0.0, 0.0, 0.0, 0.0};
};

class DigiValidationProcessor {
public:
    DigiValidationProcessor(Scifi* scifi = nullptr);
    ~DigiValidationProcessor() = default;

    void initCache(Scifi* scifi);
    size_t getCachedChannelCount() const { return fChannelMap.size(); }

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
    std::unordered_map<int, ChannelGeometry> fChannelMap;
    bool fHasGeometryCache{false};
};

} // namespace snd::trident

#endif // DIGI_VALIDATION_PROCESSOR_H
