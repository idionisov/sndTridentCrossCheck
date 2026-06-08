import ROOT
import os
import pandas as pd

year = 2023
run = 6640
input_dir = f"/eos/experiment/sndlhc/users/odurhan/multi_muon_search/{year}/{run}/"


sndsw_path = os.environ['SNDSW_ROOT']
ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndTchainGetter.h"')
ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndGeometryGetter.h"')


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

def veto_is_activated(mf_hits: ROOT.TClonesArray) -> bool:
    for mf_hit in mf_hits:
        if mf_hit.GetSystem()==1:
            return True
    return False


ch = ROOT.TChain("rawConv")
n_files = ch.Add(f"{input_dir}/*.root")


scifi, mufilter = ROOT.snd.analysis_tools.GetGeometry(run)
conf = ROOT.snd.Configuration(ROOT.snd.Configuration.GetOption(run), scifi, mufilter)

ch.GetEvent(0)
scifi.InitEvent(ch.EventHeader)
mufilter.InitEvent(ch.EventHeader)



df = {
    'event_number': [],
    'shower_start_sf': [],
    'sum_hit_density_w': [],
    'sum_qdc_sf': [],
    'max_sf_plane_qdc': [],
    'n_sf_hits': [],
    'max_plane_n_sf_hits': [],
    'veto_is_activated': []
}


for i, event in enumerate(ch):      
    eh = event.EventHeader
    scifi.InitEvent(eh)
    mufilter.InitEvent(eh)

    event_num = eh.GetEventNumber()
    sf_hits = event.Digi_ScifiHits
    mf_hits = event.Digi_MuFilterHits

    activated_veto = veto_is_activated(mf_hits)

    h_hits = ROOT.std.vector('int')(5)
    v_hits = ROOT.std.vector('int')(5)

    scifi_planes = ROOT.snd.analysis_tools.FillScifi(conf, event.Digi_ScifiHits, scifi)
    us_planes = ROOT.snd.analysis_tools.FillUS(conf, event.Digi_MuFilterHits, mufilter)
    ROOT.snd.analysis_tools.getSciFiHitsPerStation(event.Digi_ScifiHits, h_hits, v_hits)

    h_hits_max = max(h_hits)
    v_hits_max = max(v_hits)
    n_hits_max = max(h_hits_max, v_hits_max)
    n_hits_tot = max(sum(h_hits), sum(v_hits))

    qdc_list = get_sf_qdc(event.Digi_ScifiHits)
    qdc_tot = sum(qdc_list)
    qdc_max = max(qdc_list)

    scifi_sh_start = ROOT.snd.analysis_tools.GetScifiShowerStart(scifi_planes)
        
    hit_density_w = get_sum_hit_weight_density(event.Digi_ScifiHits)

    df['event_number'].append(event_num)
    df['shower_start_sf'].append(scifi_sh_start)
    df['sum_hit_density_w'].append(hit_density_w)
    df['sum_qdc_sf'].append(qdc_tot)
    df['max_sf_plane_qdc'].append(qdc_max)
    df['n_sf_hits'].append(n_hits_tot)
    df['max_plane_n_sf_hits'].append(n_hits_max)
    df['veto_is_activated'].append(activated_veto)


df = pd.DataFrame(df)
print(df)
