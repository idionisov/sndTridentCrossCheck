import ROOT
import os
import pandas as pd
from collections import defaultdict

# Setup paths and includes
base_dir = "/eos/experiment/sndlhc/convertedData/physics"
sndsw_path = os.environ['SNDSW_ROOT']
ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndTchainGetter.h"')
ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndGeometryGetter.h"')

df_white = pd.read_csv("/afs/cern.ch/work/i/idioniso/sndTridentValidation/htcondor/gallery/gallery_events.csv")
run_map = defaultdict(lambda: defaultdict(list))

for _, row in df_white.iterrows():
    r, e = int(row['run']), int(row['event_number'])
    f_num = e // 1000000
    run_map[r][f_num].append(e)

data_dict = { 'run': [], 'event_number': [], 'shower_start_sf': [], 'sum_hit_density_w_sf': [], 
             'sum_qdc_sf': [], 'max_qdc_per_sf_plane': [], 'n_hits_sf': [], 
             'max_n_hits_per_sf_plane': [] }

last_run = -1
scifi, mufilter, conf = None, None, None




def veto_is_activated(mf_hits: ROOT.TClonesArray) -> bool:
    for mf_hit in mf_hits:
        if mf_hit.GetSystem()==1:
            return True
    return False




def get_sum_hit_weight_density(sf_hits: ROOT.TClonesArray, radius: float = 40, min_check: bool = False, min_hit_density: int = 1000000):
    density = 0
    for sf in sf_hits:
        id_ = sf.GetChannelID()
        density += ROOT.snd.analysis_tools.densityScifi(id_, sf_hits, radius, min_hit_density, min_check)
    return density

def get_sf_qdc(sf_hits: ROOT.TClonesArray) -> list:
    qdc = [0, 0, 0, 0, 0]
    for sf_hit in sf_hits:
        plane = sf_hit.GetStation()-1
        qdc[plane] += sf_hit.GetSignal()
    return qdc



for run in sorted(run_map.keys()):
    if run != last_run:
        scifi, mufilter = ROOT.snd.analysis_tools.GetGeometry(run)
        conf = ROOT.snd.Configuration(ROOT.snd.Configuration.GetOption(run), scifi, mufilter)
        last_run = run

    year = 2022 if run < 6000 else 2023
    run_path = os.path.join(base_dir, str(year), f"run_{run:06d}")

    for f_num in sorted(run_map[run].keys()):
        file_path = os.path.join(run_path, f"sndsw_raw-{f_num:04d}.root")
        
        if not os.path.exists(file_path):
            print(f"Warning: File missing {file_path}")
            continue

        print(f"Processing Run {run} | File {f_num} | Events: {len(run_map[run][f_num])}")
        
        ch = ROOT.TChain("rawConv")
        ch.Add(file_path)
        ch.GetEvent(0)
        scifi.InitEvent(ch.EventHeader)
        mufilter.InitEvent(ch.EventHeader)

        for event_num in run_map[run][f_num]:
            entry_in_file = event_num % 1000000
            if ch.GetEntry(entry_in_file) <= 0:
                continue
            
            eh = ch.EventHeader
            scifi.InitEvent(eh)
            mufilter.InitEvent(eh)

            mf_hits = ch.Digi_MuFilterHits
            activated_veto = veto_is_activated(mf_hits)
            if not activated_veto:
                continue

            sf_hits = ch.Digi_ScifiHits

            h_hits = ROOT.std.vector('int')(5)
            v_hits = ROOT.std.vector('int')(5)

            scifi_planes = ROOT.snd.analysis_tools.FillScifi(conf, sf_hits, scifi)
            us_planes = ROOT.snd.analysis_tools.FillUS(conf, mf_hits, mufilter)
            ROOT.snd.analysis_tools.getSciFiHitsPerStation(sf_hits, h_hits, v_hits)

            h_hits_max = max(h_hits)
            v_hits_max = max(v_hits)
            n_hits_max = max(h_hits_max, v_hits_max)
            n_hits_tot = max(sum(h_hits), sum(v_hits))

            qdc_list = get_sf_qdc(sf_hits)
            qdc_tot = sum(qdc_list)
            qdc_max = max(qdc_list)

            scifi_sh_start = ROOT.snd.analysis_tools.GetScifiShowerStart(scifi_planes)
                
            hit_density_w = get_sum_hit_weight_density(sf_hits)

            data_dict['run'].append(run)
            data_dict['event_number'].append(event_num)
            data_dict['shower_start_sf'].append(scifi_sh_start)
            data_dict['sum_hit_density_w_sf'].append(hit_density_w)
            data_dict['sum_qdc_sf'].append(qdc_tot)
            data_dict['max_qdc_per_sf_plane'].append(qdc_max)
            data_dict['n_hits_sf'].append(n_hits_tot)
            data_dict['max_n_hits_per_sf_plane'].append(n_hits_max)


pd.DataFrame(data_dict).to_csv("gallery_results.csv", index=False)
