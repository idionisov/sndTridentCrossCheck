import ROOT
import os
import pandas as pd

years = [2022, 2023]
# base_dir = "/eos/experiment/sndlhc/users/odurhan/multi_muon_search"
base_dir = "/eos/experiment/sndlhc/convertedData/physics"

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

# Initialize the dictionary with the new 'run' column
data_dict = {
    'run': [],
    'event_number': [],
    'shower_start_sf': [],
    'sum_hit_density_w_sf': [],
    'sum_qdc_sf': [],
    'max_qdc_per_sf_plane': [],
    'n_hits_sf': [],
    'max_n_hits_per_sf_plane': [],
    'veto_is_activated': []
}

whitelist_df = pd.read_csv("/afs/cern.ch/work/i/idioniso/sndTridentValidation/htcondor/gallery/gallery_events.csv")
whitelist = set(zip(whitelist_df['run'], whitelist_df['event_number']))

# Loop over years
for year in years:
    year_dir = os.path.join(base_dir, str(year))
    if not os.path.exists(year_dir):
        continue
    
    # Loop over run directories in the year directory
    for run_str in sorted(os.listdir(year_dir)):
        input_dir = os.path.join(year_dir, run_str)
        
        # Ensure it's a directory to avoid crashing on hidden files
        if not os.path.isdir(input_dir):
            continue
            
        if "_" in run_str:
            run = int(run_str.split("_")[-1])
        else:
            run = int(run_str)
        if run not in whitelist_df["run"].unique():
            continue


        print(f"Processing Year {year} | Run {run}...")

        # Setup TChain for this specific run
        ch = ROOT.TChain("rawConv")
        n_files = ch.Add(f"{input_dir}/*.root")
        
        # Skip if no root files are found
        if n_files == 0:
            print(f"  -> No ROOT files found in {input_dir}, skipping.")
            continue

        # Initialize Geometry and Config for this run
        scifi, mufilter = ROOT.snd.analysis_tools.GetGeometry(run)
        conf = ROOT.snd.Configuration(ROOT.snd.Configuration.GetOption(run), scifi, mufilter)

        # Grab the first event to initialize the headers
        ch.GetEvent(0)
        scifi.InitEvent(ch.EventHeader)
        mufilter.InitEvent(ch.EventHeader)

        # Event Loop
        for i, event in enumerate(ch):      
            eh = event.EventHeader
            event_num = eh.GetEventNumber()

            if whitelist is not None and (run, event_num) not in whitelist:
                continue

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
            data_dict['run'].append(run)
            data_dict['event_number'].append(event_num)
            data_dict['shower_start_sf'].append(scifi_sh_start)
            data_dict['sum_hit_density_w_sf'].append(hit_density_w)
            data_dict['sum_qdc_sf'].append(qdc_tot)
            data_dict['max_qdc_per_sf_plane'].append(qdc_max)
            data_dict['n_hits_sf'].append(n_hits_tot)
            data_dict['max_n_hits_per_sf_plane'].append(n_hits_max)
            data_dict['veto_is_activated'].append(activated_veto)

# Create the final DataFrame
df = pd.DataFrame(data_dict)
print(df)
