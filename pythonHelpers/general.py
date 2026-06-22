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
    """)

def get_scifi_hit_params(scifi_hits, radius=40, min_check=False, min_hit_density=1000000):
    res = ROOT.computeSciFiParams(scifi_hits, radius, min_check, min_hit_density)
    return res.sf_nhits, res.hit_w_density, res.sf_total_qdc, res.max_sf_nhits, res.max_sf_qdc

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
