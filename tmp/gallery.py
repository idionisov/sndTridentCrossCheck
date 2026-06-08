import ROOT
import os
import pandas as pd
import ddfUtils.root as ddfRoot

run = 6640
year = 2023
input_dir = f"/eos/experiment/sndlhc/convertedData/physics/{year}/run_{run:06d}"

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

h = {
    'shower_start_sf': ROOT.TH1I("h1d_shower_start_sf", ";shower_start_sf;", 6, -1, 5),
    'sum_hit_density_w_sf': ROOT.TH1F("h1d_sum_hdw_sf", ";sum_hit_density_w_sf;", 1000, 0, 350000),
    'sum_qdc_sf': ROOT.TH1F("h1d_sum_qdc_sf", ";sum_qdc_sf;", 1000, 0, 20000),
    'max_qdc_per_sf_plane': ROOT.TH1F("h1d_max_qdc_sf", ";max_qdc_sf_per_sf_plane;", 1000, 0, 15000),
    'n_hits_sf': ROOT.TH1I("h1d_n_hits_sf", ";n_hits_sf;", 1100, 0, 3300),
    'max_n_hits_per_sf_plane': ROOT.TH1I("h1d_max_n_hits_sf", ";max_n_hits_per_sf_plane;", 1100, 0, 1100)
}



print(f"Processing Year {year} | Run {run}...")

ch = ROOT.TChain("rawConv")
n_files = ch.Add(f"{input_dir}/*.root")

if n_files == 0:
    print(f"  -> No ROOT files found in {input_dir}, skipping.")
    exit(-1)

scifi, mufilter = ROOT.snd.analysis_tools.GetGeometry(run)
conf = ROOT.snd.Configuration(ROOT.snd.Configuration.GetOption(run), scifi, mufilter)

ch.GetEvent(0)
scifi.InitEvent(ch.EventHeader)
mufilter.InitEvent(ch.EventHeader)

for i, event in enumerate(ch):      
    eh = event.EventHeader
    event_num = eh.GetEventNumber()

    scifi.InitEvent(eh)
    mufilter.InitEvent(eh)

    sf_hits = event.Digi_ScifiHits
    mf_hits = event.Digi_MuFilterHits

    activated_veto = veto_is_activated(mf_hits)

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

    # Append to dictionary
    h['shower_start_sf'].Fill(scifi_sh_start)
    h['sum_hit_density_w_sf'].Fill(hit_density_w)
    h['sum_qdc_sf'].Fill(qdc_tot)
    h['max_qdc_per_sf_plane'].Fill(qdc_max)
    h['n_hits_sf'].Fill(n_hits_tot)
    h['max_n_hits_per_sf_plane'].Fill(n_hits_max)

ddfRoot.save_to_root(h, fout="all_6640.root")
