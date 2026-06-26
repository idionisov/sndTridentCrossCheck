import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ROOT

# Load libraries and declare C++ helper once
if not hasattr(ROOT, 'computeSciFiParams'):
    if not hasattr(ROOT, 'sndScifiHit'):
        for library in ["libBase", "libShipData", "libshipLHC", "libsnd_analysis_tools"]:
            ROOT.gSystem.Load(library)
    sndsw_path = os.environ.get('SNDSW_ROOT', '')
    if sndsw_path:
        ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndSciFiTools.h"')
    
    ROOT.gInterpreter.Declare("""
    #include "sndScifiHit.h"
    #include "TClonesArray.h"
    #include "sndSciFiTools.h"
    #include <vector>
    #include <algorithm>
    #include <cmath>

    struct SciFiParams {
        int sf_nhits;
        double hit_w_density;
        double sf_total_qdc;
        int max_sf_nhits;
        double max_sf_qdc;
    };

    SciFiParams computeSciFiParams(TClonesArray* scifi_hits, double radius, bool min_check, double min_hit_density) {
        SciFiParams params = {0, 0.0, 0.0, 0, 0.0};
        if (!scifi_hits) return params;
        
        int entries = scifi_hits->GetEntries();
        params.sf_nhits = entries;
        
        int nhits_per_plane[2][5] = {{0}}; // [0]: horizontal, [1]: vertical
        double qdc_per_plane[2][5] = {{0.0}};
        
        for (int i = 0; i < entries; ++i) {
            sndScifiHit* hit = (sndScifiHit*)scifi_hits->At(i);
            if (!hit || !hit->isValid()) continue;
            
            double qdc = hit->GetSignal();
            params.sf_total_qdc += qdc;
            
            int is_vertical = hit->isVertical() ? 1 : 0;
            int station = hit->GetStation(); // 1 to 5
            if (station >= 1 && station <= 5) {
                nhits_per_plane[is_vertical][station - 1] += 1;
                qdc_per_plane[is_vertical][station - 1] += qdc;
            }
            
            params.hit_w_density += snd::analysis_tools::densityScifi(hit->GetChannelID(), *scifi_hits, radius, min_hit_density, min_check);
        }
        
        for (int orient = 0; orient < 2; ++orient) {
            for (int plane = 0; plane < 5; ++plane) {
                if (nhits_per_plane[orient][plane] > params.max_sf_nhits) {
                    params.max_sf_nhits = nhits_per_plane[orient][plane];
                }
                if (qdc_per_plane[orient][plane] > params.max_sf_qdc) {
                    params.max_sf_qdc = qdc_per_plane[orient][plane];
                }
            }
        }
        
        return params;
    }

    bool validateHit_local(sndScifiHit *aHit, int ref_station, bool ref_orientation) {
        if (!aHit || !aHit->isValid()) return false;
        if (aHit->GetStation() != ref_station) return false;
        if (aHit->isVertical() != ref_orientation) return false;
        return true;
    }

    double qdcDensityScifi_local(int reference_SiPM, TClonesArray* scifi_hits, int radius) {
        double qdc_density = 0.0;
        bool orientation = (int(reference_SiPM / 100000) % 10 == 1);
        int ref_station = reference_SiPM / 1000000;
        int referenceChannel = snd::analysis_tools::calculateSiPMNumber(reference_SiPM);
        
        int entries = scifi_hits->GetEntries();
        for (int i = 0; i < entries; ++i) {
            sndScifiHit* hit = (sndScifiHit*)scifi_hits->At(i);
            if (!validateHit_local(hit, ref_station, orientation)) continue;
            int hitChannel = snd::analysis_tools::calculateSiPMNumber(hit->GetChannelID());
            if (radius == -1) {
                qdc_density += hit->GetSignal();
            } else {
                if (hitChannel > referenceChannel + radius) break;
                if (std::abs(referenceChannel - hitChannel) <= radius) {
                    qdc_density += hit->GetSignal();
                }
            }
        }
        return qdc_density;
    }

    struct EventSciFiInfo {
        int sf_nhits;
        double sum_hit_w_density;
        double max_hit_w_density;
        double sum_qdc_w_density;
        double max_qdc_w_density;
        double sum_qdc;
        double max_qdc;
        int max_sf_nhits_per_plane;
        double max_sf_qdc_per_plane;
        std::vector<double> hit_w_densities;
        std::vector<double> qdc_w_densities;
        std::vector<double> qdcs;
    };

    EventSciFiInfo computeEventSciFiInfo(TClonesArray* scifi_hits, double radius, bool min_check, double min_hit_density) {
        EventSciFiInfo info = {0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, {}, {}, {}};
        if (!scifi_hits) return info;
        
        int entries = scifi_hits->GetEntries();
        info.sf_nhits = entries;
        info.hit_w_densities.resize(entries, 0.0);
        info.qdc_w_densities.resize(entries, 0.0);
        info.qdcs.resize(entries, 0.0);
        
        int nhits_per_plane[2][5] = {{0}};
        double qdc_per_plane[2][5] = {{0.0}};
        
        for (int i = 0; i < entries; ++i) {
            sndScifiHit* hit = (sndScifiHit*)scifi_hits->At(i);
            if (!hit) continue;
            
            double qdc = hit->GetSignal();
            info.qdcs[i] = qdc;
            
            if (!hit->isValid()) continue;
            
            info.sum_qdc += qdc;
            if (qdc > info.max_qdc) {
                info.max_qdc = qdc;
            }
            
            int is_vertical = hit->isVertical() ? 1 : 0;
            int station = hit->GetStation();
            if (station >= 1 && station <= 5) {
                nhits_per_plane[is_vertical][station - 1] += 1;
                qdc_per_plane[is_vertical][station - 1] += qdc;
            }
            
            double hit_w = (double)snd::analysis_tools::densityScifi(hit->GetChannelID(), *scifi_hits, radius, min_hit_density, min_check);
            info.hit_w_densities[i] = hit_w;
            info.sum_hit_w_density += hit_w;
            if (hit_w > info.max_hit_w_density) {
                info.max_hit_w_density = hit_w;
            }
            
            double qdc_w = qdcDensityScifi_local(hit->GetChannelID(), scifi_hits, radius);
            info.qdc_w_densities[i] = qdc_w;
            info.sum_qdc_w_density += qdc_w;
            if (qdc_w > info.max_qdc_w_density) {
                info.max_qdc_w_density = qdc_w;
            }
        }
        
        for (int orient = 0; orient < 2; ++orient) {
            for (int plane = 0; plane < 5; ++plane) {
                if (nhits_per_plane[orient][plane] > info.max_sf_nhits_per_plane) {
                    info.max_sf_nhits_per_plane = nhits_per_plane[orient][plane];
                }
                if (qdc_per_plane[orient][plane] > info.max_sf_qdc_per_plane) {
                    info.max_sf_qdc_per_plane = qdc_per_plane[orient][plane];
                }
            }
        }
        
        return info;
    }
    """)

def get_scifi_hit_params(scifi_hits, radius=40, min_check=False, min_hit_density=1000000):
    res = ROOT.computeSciFiParams(scifi_hits, radius, min_check, min_hit_density)
    return res.sf_nhits, res.hit_w_density, res.sf_total_qdc, res.max_sf_nhits, res.max_sf_qdc

def get_event_scifi_info(scifi_hits, radius=40, min_check=False, min_hit_density=1000000):
    res = ROOT.computeEventSciFiInfo(scifi_hits, radius, min_check, min_hit_density)
    return {
        'sf_nhits': res.sf_nhits,
        'sum_hit_weight_density': res.sum_hit_w_density,
        'max_hit_weight_density': res.max_hit_w_density,
        'sum_qdc_weight_density': res.sum_qdc_w_density,
        'max_qdc_weight_density': res.max_qdc_w_density,
        'sum_qdc': res.sum_qdc,
        'max_qdc': res.max_qdc,
        'max_scifi_nhits_per_plane': res.max_sf_nhits_per_plane,
        'max_scifi_qdc_per_plane': res.max_sf_qdc_per_plane,
        'hit_w_densities': list(res.hit_w_densities),
        'qdc_w_densities': list(res.qdc_w_densities),
        'qdcs': list(res.qdcs)
    }

def get_scifi_hit_density(scifi_hits, radius=40, min_check=False, min_hit_density=1000000):
    return get_scifi_hit_params(scifi_hits, radius, min_check, min_hit_density)[1]

def get_scifi_total_qdc(scifi_hits):
    return get_scifi_hit_params(scifi_hits)[2]

def get_scifi_total_nhits(scifi_hits):
    return scifi_hits.GetEntries()

def get_scifi_max_nhits_per_plane(scifi_hits):
    return get_scifi_hit_params(scifi_hits)[3]

def get_scifi_max_qdc_per_plane(scifi_hits):
    return get_scifi_hit_params(scifi_hits)[4]


def flush_to_parquet(data, batch_idx, cols, base_filename, verbose=False):
    df = pd.DataFrame(data, columns=cols)
    table = pa.Table.from_pandas(df)
    
    path = Path(base_filename)
    if path.suffix.lower() == '.parquet':
        filename = f"{path.with_suffix('')}_{batch_idx}.parquet"
    else:
        filename = f"{path}_{batch_idx}.parquet"
        
    pq.write_table(table, filename, compression='snappy')

    if verbose:
        print(f"Batch {batch_idx} written to {filename}")
