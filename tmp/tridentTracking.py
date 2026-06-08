import os
import random
import ROOT

# Enable batch mode to suppress graphics windows
ROOT.gROOT.SetBatch(True)

import argparse
import csv
import datetime
import time
from array import array

import numpy as np
import pandas as pd
import rootUtils as ut

# Load SND@LHC specific libraries
for lib in ["libBase", "libShipData", "libshipLHC"]:
    ROOT.gSystem.Load(lib)

import SndlhcGeo
import SndlhcMuonReco
import SndlhcTracking

sndsw_path = os.environ['SNDSW_ROOT']
ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndSciFiTools.h"')

# Global display objects
h = {}
proj = {1: 'xz', 2: 'yz'}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file',  type=str, required=True, help='Path to input file.')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='Output CSV filename')
    parser.add_argument('-or', '--output-root', type=str, help='Output ROOT filename for selected events')
    parser.add_argument('-s', '--start-event', type=int, help='Start event number.', default=0)
    parser.add_argument('-n', '--n-events', type=int, help='Number of events.', default=1000000)
    parser.add_argument('-f', '--fraction', type=float, default=1.0, help='Fraction of events to process')
    return parser.parse_args()

def get_geofile(year: int) -> str:
    if year == 2022: return "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root"
    elif year == 2023: return "/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V3_2023.root"
    elif year == 2024: return "/eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V12_2024.root"
    elif year == 2025: return "/eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V8_2025.root"
    return None

def veto_is_activated(mf_hits: ROOT.TClonesArray) -> bool:
    for mf_hit in mf_hits:
        if mf_hit.GetSystem() == 1: return True
    return False

def get_sum_hit_weight_density(sf_hits: ROOT.TClonesArray, radius: float = 40, min_check: bool = False, min_hit_density: int = 1000000):
    density = 0
    for sf in sf_hits:
        id_ = sf.GetChannelID()
        density += ROOT.snd.analysis_tools.densityScifi(id_, sf_hits, radius, min_hit_density, min_check)
    return density

def get_sf_qdc(sf_hits: ROOT.TClonesArray) -> float:
    qdc = 0
    for sf_hit in sf_hits: qdc += sf_hit.GetSignal()
    return qdc

def initialize_display(geo):
    """Setup the canvas and background histograms for event display."""
    ut.bookCanvas(h, key='simpleDisplay', title='simple event display', nx=1200, ny=1016, cx=1, cy=2)
    zStart, zEnd = 250., 600.
    xStart, yStart = -100., -30.
    h['xmin'], h['xmax'] = xStart, xStart + 110.
    h['ymin'], h['ymax'] = yStart, yStart + 110.
    h['zmin'], h['zmax'] = zStart, zEnd
    ut.bookHist(h, 'xz', '; z [cm]; x [cm]', 500, h['zmin'], h['zmax'], 100, h['xmin'], h['xmax'])
    ut.bookHist(h, 'yz', '; z [cm]; y [cm]', 500, h['zmin'], h['zmax'], 100, h['ymin'], h['ymax'])
    h['xz'].SetStats(0); h['yz'].SetStats(0)

def drawDetectors(geo):
    """Ported from 2dEventDisplay.py to draw geometry boxes."""
    mi = geo.snd_geo.MuFilter
    si = geo.snd_geo.Scifi
    em = geo.snd_geo.EmulsionDet
    nav = ROOT.gGeoManager.GetCurrentNavigator()

    nodes = {'volMuFilter_1/volFeBlockEnd_1':ROOT.kGreen-6}
    for i in range(mi.NVetoPlanes):
       nodes['volVeto_1/volVetoPlane_{}_{}'.format(i, i)]=ROOT.kRed
       for j in range(mi.NVetoBars):
          if i<2: nodes['volVeto_1/volVetoPlane_{}_{}/volVetoBar_1{}{:0>3d}'.format(i, i, i, j)]=ROOT.kRed
          if i==2: nodes['volVeto_1/volVetoPlane_{}_{}/volVetoBar_ver_1{}{:0>3d}'.format(i, i, i, j)]=ROOT.kRed
       if i<2: nodes['volVeto_1/subVetoBox_{}'.format(i)]=ROOT.kGray+1
       if i==2: nodes['volVeto_1/subVeto3Box_{}'.format(i)]=ROOT.kGray+1
    for i in range(si.nscifi):
       nodes['volTarget_1/ScifiVolume{}_{}000000'.format(i+1, i+1)]=ROOT.kBlue+1
       nodes['volTarget_1/volFeTarget{}_1'.format(i+1)]=ROOT.kGreen-6
    for i in range(em.wall):
       nodes['volTarget_1/volWallborder_{}'.format(i)]=ROOT.kGray
    for i in range(mi.NDownstreamPlanes):
       nodes['volMuFilter_1/volMuDownstreamDet_{}_{}'.format(i, i+mi.NVetoPlanes+mi.NUpstreamPlanes)]=ROOT.kBlue+1
       for j in range(mi.NDownstreamBars):
          nodes['volMuFilter_1/volMuDownstreamDet_{}_{}/volMuDownstreamBar_ver_3{}{:0>3d}'.format(i, i+mi.NVetoPlanes+mi.NUpstreamPlanes, i, j+mi.NDownstreamBars)]=ROOT.kBlue+1
          if i < 3:
             nodes['volMuFilter_1/volMuDownstreamDet_{}_{}/volMuDownstreamBar_hor_3{}{:0>3d}'.format(i, i+mi.NVetoPlanes+mi.NUpstreamPlanes, i, j)]=ROOT.kBlue+1
    for i in range(mi.NDownstreamPlanes):
       nodes['volMuFilter_1/subDSBox_{}'.format(i+mi.NVetoPlanes+mi.NUpstreamPlanes)]=ROOT.kGray+1
    for i in range(mi.NUpstreamPlanes):
       nodes['volMuFilter_1/subUSBox_{}'.format(i+mi.NVetoPlanes)]=ROOT.kGray+1
       nodes['volMuFilter_1/volMuUpstreamDet_{}_{}'.format(i, i+mi.NVetoPlanes)]=ROOT.kBlue+1
       for j in range(mi.NUpstreamBars):
          nodes['volMuFilter_1/volMuUpstreamDet_{}_{}/volMuUpstreamBar_2{}00{}'.format(i, i+mi.NVetoPlanes, i, j)]=ROOT.kBlue+1
       nodes['volMuFilter_1/volFeBlock_{}'.format(i)]=ROOT.kGreen-6
    for i in range(mi.NVetoPlanes+mi.NUpstreamPlanes,mi.NVetoPlanes+mi.NUpstreamPlanes+mi.NDownstreamPlanes):
       nodes['volMuFilter_1/volFeBlock_{}'.format(i)]=ROOT.kGreen-6

    passNodes = {'Block', 'Wall', 'FeTarget'}
    xNodes = {'UpstreamBar', 'VetoBar', 'hor'}
    proj_map = {'X':0,'Y':1}

    for node_ in nodes:
       node = '/cave_1/Detector_0/'+node_
       for p in ['X', 'Y']:
          if not nav.CheckPath(node): continue
          nav.cd(node)
          N = nav.GetCurrentNode()
          S = N.GetVolume().GetShape()
          dx,dy,dz = S.GetDX(),S.GetDY(),S.GetDZ()
          ox,oy,oz = S.GetOrigin()[0],S.GetOrigin()[1],S.GetOrigin()[2]
          P, M = {}, {}
          if p=='X' and (not any(xNode in node for xNode in xNodes) or 'VetoBar_ver' in node):
             P['LB'] = array('d',[-dx+ox,oy,-dz+oz]); P['LT'] = array('d',[dx+ox,oy,-dz+oz])
             P['RB'] = array('d',[-dx+ox,oy,dz+oz]);  P['RT'] = array('d',[dx+ox,oy,dz+oz])
          elif p=='Y' and 'ver' not in node:
             P['LB'] = array('d',[ox,-dy+oy,-dz+oz]); P['LT'] = array('d',[ox,dy+oy,-dz+oz])
             P['RB'] = array('d',[ox,-dy+oy,dz+oz]);  P['RT'] = array('d',[ox,dy+oy,dz+oz])
          else: continue

          for C in P:
             M[C] = array('d',[0,0,0])
             nav.LocalToMaster(P[C],M[C])

          poly = ROOT.TPolyLine()
          c = proj_map[p]
          poly.SetPoint(0,M['LB'][2],M['LB'][c]); poly.SetPoint(1,M['LT'][2],M['LT'][c])
          poly.SetPoint(2,M['RT'][2],M['RT'][c]); poly.SetPoint(3,M['RB'][2],M['RB'][c])
          poly.SetPoint(4,M['LB'][2],M['LB'][c])
          poly.SetLineColor(nodes[node_]); poly.SetLineWidth(1)

          h['simpleDisplay'].cd(c+1)
          if any(passNode in node for passNode in passNodes):
             poly.SetFillColorAlpha(nodes[node_], 0.5)
             poly.Draw('f&&same')
          poly.Draw('same')
          h[f"{node}_{p}"] = poly

def draw_event(event, geo, hough_lines):
    """Draw SciFi & MuFilter hits and hough lines on the canvas."""
    for p in proj:
        h['simpleDisplay'].cd(p)
        h[proj[p]].Draw('b')

    drawDetectors(geo)

    hit_graphs = {
        'X': {'Scifi': ROOT.TGraphErrors(), 'MuFilter': ROOT.TGraphErrors()},
        'Y': {'Scifi': ROOT.TGraphErrors(), 'MuFilter': ROOT.TGraphErrors()}
    }
    counts = {'X': {'Scifi': 0, 'MuFilter': 0}, 'Y': {'Scifi': 0, 'MuFilter': 0}}

    # SciFi Hits
    si = geo.snd_geo.Scifi
    for sfHit in event.Digi_ScifiHits:
        if not sfHit.isValid(): continue
        A, B = ROOT.TVector3(), ROOT.TVector3()
        geo.modules['Scifi'].GetSiPMPosition(sfHit.GetDetectorID(), A, B)
        Z = A.Z()
        orient = 'X' if sfHit.isVertical() else 'Y'
        g, idx = hit_graphs[orient]['Scifi'], counts[orient]['Scifi']
        g.SetPoint(idx, Z, A.X() if orient == 'X' else A.Y())
        g.SetPointError(idx, si.scifimat_z/2, si.channel_width/2)
        counts[orient]['Scifi'] += 1

    # MuFilter Hits
    mi = geo.snd_geo.MuFilter
    for muHit in event.Digi_MuFilterHits:
        if not muHit.isValid(): continue
        A, B = ROOT.TVector3(), ROOT.TVector3()
        geo.modules['MuFilter'].GetPosition(muHit.GetDetectorID(), A, B)
        Z = (A.Z() + B.Z()) / 2
        orient = 'X' if muHit.isVertical() else 'Y'
        g, idx = hit_graphs[orient]['MuFilter'], counts[orient]['MuFilter']
        system = muHit.GetSystem()
        if system == 1: # Veto
            dx, dy, dz = (mi.Veto3BarX/2 if hasattr(mi, "Veto3BarX") else mi.VetoBarX/2), mi.VetoBarY/2, mi.VetoBarZ/2
        elif system == 2: # US
            dx, dy, dz = mi.UpstreamBarX/2, mi.UpstreamBarY/2, mi.UpstreamBarZ/2
        else: # DS
            dx, dy, dz = mi.DownstreamBarX_ver/2, mi.DownstreamBarY/2, mi.DownstreamBarZ/2
        g.SetPoint(idx, Z, A.X() if orient == 'X' else A.Y())
        g.SetPointError(idx, dz, dx if orient == 'X' else dy)
        counts[orient]['MuFilter'] += 1

    for orient, p_idx in [('X', 1), ('Y', 2)]:
        h['simpleDisplay'].cd(p_idx)
        gs = hit_graphs[orient]['Scifi']
        gs.SetMarkerStyle(20); gs.SetMarkerSize(0.8); gs.SetMarkerColor(ROOT.kBlue+2)
        gs.Draw('sameP'); h[f'hits_sf_{orient}'] = gs
        gm = hit_graphs[orient]['MuFilter']
        gm.SetMarkerStyle(21); gm.SetMarkerSize(0.8); gm.SetMarkerColor(ROOT.kRed+1)
        gm.Draw('sameP'); h[f'hits_mu_{orient}'] = gm
        p_name = 'XZ' if orient == 'X' else 'YZ'
        for i, (slope, intercept) in enumerate(hough_lines[p_name]):
            zmin, zmax = h['zmin'], h['zmax']
            y1, y2 = slope * zmin + intercept, slope * zmax + intercept
            line = ROOT.TLine(zmin, y1, zmax, y2)
            line.SetLineColor(ROOT.kCyan+2); line.SetLineWidth(2)
            line.Draw("same"); h[f'line_{p_name}_{i}'] = line

def getHoughLines(ht_task, event, geo):
    """ Returns (n_tracks, max_dangle, line_params_dict) """
    hit_collection = {"pos": [[], [], []], "d": [[], [], []], "vert": [], "system": [], "detectorID": []}
    a, b = ROOT.TVector3(), ROOT.TVector3()
    for sfHit in event.Digi_ScifiHits:
        if not sfHit.isValid(): continue
        geo.modules['Scifi'].GetSiPMPosition(sfHit.GetDetectorID(), a, b)
        hit_collection["pos"][0].append(a.X()); hit_collection["pos"][1].append(a.Y()); hit_collection["pos"][2].append(a.Z())
        hit_collection["d"][0].append(ht_task.Scifi_dx); hit_collection["d"][1].append(ht_task.Scifi_dy); hit_collection["d"][2].append(ht_task.Scifi_dz)
        hit_collection["vert"].append(sfHit.isVertical()); hit_collection["system"].append(0); hit_collection["detectorID"].append(sfHit.GetDetectorID())

    if not hit_collection['pos'][0]: return 0, 0.0, {'XZ': [], 'YZ': []}
    for k in ["pos", "d"]: hit_collection[k] = np.array(hit_collection[k], dtype=np.float32)
    hit_collection["vert"] = np.array(hit_collection["vert"], dtype=np.bool_); hit_collection["system"] = np.array(hit_collection["system"], dtype=np.int32)
    hit_collection["detectorID"] = np.array(hit_collection["detectorID"], dtype=np.int32)
    counts, slopes, lines = {'XZ': 0, 'YZ': 0}, {'XZ': [], 'YZ': []}, {'XZ': [], 'YZ': []}
    for proj_name in ['XZ', 'YZ']:
        is_v, ax = (proj_name == 'XZ'), (0 if proj_name == 'XZ' else 1)
        h_obj = ht_task.h_ZX if is_v else ht_task.h_ZY
        proj_used = np.zeros(len(hit_collection['pos'][0]), dtype=np.bool_)
        for i_muon in range(ht_task.max_reco_muons):
            m = np.logical_and(hit_collection['vert'] == is_v, ~proj_used)
            if not np.any(m): break
            res = h_obj.fit_randomize(np.dstack([hit_collection['pos'][2][m], hit_collection['pos'][ax][m]])[0],
                                      np.dstack([hit_collection['d'][2][m], hit_collection['d'][ax][m]])[0],
                                      ht_task.n_random, False, False)
            if res[0] in [-1, -999]: break
            hits_rel = SndlhcMuonReco.hit_finder(res[0], res[1], np.dstack([hit_collection['pos'][2][m], hit_collection['pos'][ax][m]]),
                                               np.dstack([hit_collection['d'][2][m], hit_collection['d'][ax][m]]), ht_task.tolerance)
            if len(hits_rel) == 0: break
            if SndlhcMuonReco.numPlanesHit(hit_collection['system'][m][hits_rel], hit_collection['detectorID'][m][hits_rel]) >= ht_task.min_planes_hit:
                counts[proj_name] += 1; slopes[proj_name].append(res[0]); lines[proj_name].append(res); proj_used[np.where(m)[0][hits_rel]] = True
            else: break
    max_dangle_XZ = np.ptp(np.arctan(slopes['XZ'])) if counts['XZ'] >= 2 else 0.0
    max_dangle_YZ = np.ptp(np.arctan(slopes['YZ'])) if counts['YZ'] >= 2 else 0.0
    if counts['XZ'] > counts['YZ']: return counts['XZ'], max_dangle_XZ, lines
    elif counts['YZ'] > counts['XZ']: return counts['YZ'], max_dangle_YZ, lines
    else: return counts['XZ'], max(max_dangle_XZ, max_dangle_YZ), lines

def main():
    args = get_args()
    t0 = time.time()
    f_in = ROOT.TFile.Open(args.input_file if args.input_file.endswith(".root") else args.input_file + ".root")
    tree_in = f_in.Get("rawConv"); tree_in.GetEvent(0)
    year = datetime.datetime.fromtimestamp(tree_in.EventHeader.GetUTCtimestamp()).year
    geo = SndlhcGeo.GeoInterface(get_geofile(year))
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    for m_name in ['Scifi', 'MuFilter']:
        obj = lsOfGlobals.FindObject(m_name)
        if obj: lsOfGlobals.Remove(obj)
        lsOfGlobals.Add(geo.modules[m_name])
    initialize_display(geo)
    run = ROOT.FairRunAna()
    ioman = ROOT.FairRootManager.Instance(); ioman.SetTreeName("rawConv")
    run.SetSource(ROOT.FairFileSource(f_in)); sink = ROOT.FairRootFileSink(ROOT.TMemFile('dummy','CREATE')); run.SetSink(sink)
    xrdb = ROOT.FairRuntimeDb.instance()
    xrdb.getContainer("FairBaseParSet").setStatic(); xrdb.getContainer("FairGeoParSet").setStatic()
    sf_task = SndlhcMuonReco.MuonReco(); run.AddTask(sf_task)
    sf_task.SetParFile(f"/afs/cern.ch/user/i/idioniso/snd_master/sndsw/trackingParams.xml")
    sf_task.SetHoughSpaceFormat("linearSlopeIntercept"); sf_task.SetTrackingCase('passing_mu_Sf'); run.Init()
    tree_in = ioman.GetInTree()
    if tree_in.GetBranch('Digi_MuFilterHit'): tree_in.Digi_MuFilterHits = tree_in.Digi_MuFilterHit
    f_out_root, tree_out = None, None
    if args.output_root:
        f_out_root = ROOT.TFile(args.output_root if args.output_root.endswith(".root") else args.output_root + ".root", "RECREATE")
        tree_out = tree_in.CloneTree(0)
    df = pd.read_csv("/afs/cern.ch/work/i/idioniso/sndTridentValidation/htcondor/tridents.csv")
    daq_run = tree_in.EventHeader.GetRunId()
    WHITELIST =  [40494705, 42654342, 42779710, 44090196, 45071967, 45273862, 45777264, 45829061, 46370701, 47186301, 49321027, 49835804, 50827835, 51934379, 52432412, 52476304, 53580626, 53815548, 55650753, 56093454, 56632346, 57279696, 58561242, 59234686, 61085082, 61112250, 61785031, 64265831, 64449161, 65116147, 65411337, 65431051, 65904880, 65956245, 66591491, 67132461, 67225718, 68056969, 68687707, 69580838, 69923664, 70272461, 70273425, 70556208, 70886216, 71633702, 71998173, 72663359, 72889207, 73716952, 73807459, 73908606, 74222812, 75166684, 75368180, 75842405, 76781076, 80554030, 85769372, 86435914, 87264719, 87747393, 92998879, 93490327, 93651815, 93806772, 94293041, 94806364, 95233375, 95611465, 95955082, 96084799, 96755615, 97532628, 98883980, 98895508, 99152766, 99436481, 103647057, 103817761, 104244404, 104513758, 104948623, 107199750, 107783796, 109036526, 109429153, 109578358, 109680713, 111422117, 113056913, 113436635, 115350216, 116090319, 116500525, 116643555, 116670128, 116738793, 117178921, 117323508, 119229104, 120309792, 120707975, 121726815, 122317572, 123340759, 124194176, 125140906, 125704009, 126841256, 127320752, 127355203, 128051713, 129993469, 131901260, 133301464, 134029038, 134241152, 134490696, 134987630, 137643293, 138494074, 140021271, 141838864, 142174812, 142874511, 143203840, 143508457, 145105137, 146941968, 147672959, 147707163, 149346933, 149609396, 150556659, 152919556, 153969669, 154437652, 156422425, 156635358, 156804137, 157242705, 158371960, 158438181, 159038098, 159917148, 161285394, 165325145, 165662260, 165982211, 166403543, 167006118, 167121211, 167990284, 168484779, 168532617, 168668960, 169009271, 171029610, 171134611, 171219248, 171993934, 172495486, 173771479, 174340531, 174889937, 176964532, 177351627, 178110245, 178652432, 179053288, 179840573, 180718415, 183252121, 183862566, 183939945, 185822946, 185872868, 187378229, 187548063, 188795643, 189021483, 189821989, 190299489, 190716300, 191075839, 191298675, 192364136, 192592870, 193058423, 194434799, 194534261, 195650312, 196067754, 197094657, 197753718, 198641297, 199557231, 199621757, 199908550, 200106407, 200445824, 201990837, 202298237, 203902706, 206870530, 206923122, 207138945, 207521736, 208221757, 208495583, 208774038, 209894637, 211496144, 211982295, 212265880, 212909071, 214055525, 214424986, 214645436, 215415024, 215666108, 216363643, 216919237, 217365434, 217493056, 218558222, 218930065, 219394086, 220034161, 220056893, 220255965, 221512291, 221520327, 222074698, 222311617, 222593933, 224476291, 224558824, 225109609, 226844402, 228267411, 228776199, 229836796, 230688132, 233318389, 233797231, 233969514, 235670156, 235736563, 236418139, 236716622, 237594737, 237751003, 237845981, 237874406, 238270356, 238331038, 239956146, 241136308, 242752402, 243469118, 243894540, 244326260, 245891688, 245948882, 246785581, 247534565, 248130095, 249130718, 249597177, 251243086, 251443130, 251848878, 252820632, 255266315, 255396147, 255564347, 255572132, 255667749, 256262133, 256385405, 256460296, 256727545, 256908326, 259496714, 259569379, 260625198, 260763693, 261429243, 261461085, 261672215, 262321179, 263577818, 263950327, 264506400, 266758090, 267324648, 267745017, 268479569, 269010447, 271114761, 271294562, 276384258, 277538777, 279101354, 279879631, 279911929, 281264385, 281955469, 282307870, 285731550, 288122455, 288307130, 288804298, 289188326, 290274434, 291740152, 293050322, 293258307, 293280467, 294490083, 296928538, 297186128, 297930160, 297989287, 302487472, 303146709, 306293456, 306372121, 307723816, 308692919, 309614890, 310043005, 313022659, 314585734, 314792888, 315274399, 316899157, 321206640, 322875180, 324525792, 327619890, 328984313, 329489631, 330987145, 331878688, 332039047, 332981735, 333696055, 333820028, 335315021, 335840888, 336630802, 338121234, 342381621, 343502777, 344061676, 348469113, 348627887, 349032529, 349400906, 349489827, 349992797, 350831633, 351257586, 351491741, 354699007, 355757990, 356394207, 357521263, 359176822, 359606932, 360150622, 360652936, 363174876, 363378483, 363400031]
    n_break = min(args.n_events, tree_in.GetEntries() - args.start_event); pr = 0

    with open(
        args.output_file if args.output_file.endswith(".csv") else args.output_file + ".csv", mode='w', newline=''
    ) as csv_file:
        fieldnames = ['run', 'event_number', 'activated_veto', 'n_tracks', 'max_dangle', 'sum_hit_weight_density', 'sum_qdc']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i_entry in range(args.start_event, args.start_event + n_break):
            tree_in.GetEvent(i_entry); event_number = tree_in.EventHeader.GetEventNumber()
            if event_number not in WHITELIST:
                continue
            if random.random() >= args.fraction:
                continue


            if ((i_entry - args.start_event) * 100 / n_break) >= pr:
                h_time, rem = divmod(int(time.time() - t0), 3600); m, s = divmod(rem, 60)
                print(f"[{int(pr)}%] \t {i_entry-args.start_event:,}/{n_break:,} \t {h_time:02d}:{m:02d}:{s:02d}"); pr += 1
            if tree_in.EventHeader.ClassName() == 'SNDLHCEventHeader': geo.modules['Scifi'].InitEvent(tree_in.EventHeader)
            n_tracks, max_dangle, lines = getHoughLines(sf_task, tree_in, geo)
            if n_tracks >= 3:
                print(f"Match: Event {event_number}, Lines {n_tracks}, DAngle {max_dangle:.4f}")
                if tree_out:
                    tree_out.Fill(); draw_event(tree_in, geo, lines)
                    c = h['simpleDisplay']; c.SetName(f"c_Run{daq_run}_{event_number}"); c.SetTitle(f"Event {event_number} - {n_tracks} Lines")
                    f_out_root.cd(); c.Write()

            if n_tracks >= 2:
                writer.writerow({'run': tree_in.EventHeader.GetRunId(), 'event_number': event_number, 'activated_veto': veto_is_activated(tree_in.Digi_MuFilterHits), 'n_tracks': n_tracks, 'max_dangle': max_dangle, 'sum_hit_weight_density': get_sum_hit_weight_density(tree_in.Digi_ScifiHits), 'sum_qdc': get_sf_qdc(tree_in.Digi_ScifiHits)})
                csv_file.flush()
    if f_out_root: f_out_root.cd(); tree_out.Write(); f_out_root.Close()
    print(f"Finished in {time.time() - t0:.2f}s.")

if __name__ == "__main__": main()
