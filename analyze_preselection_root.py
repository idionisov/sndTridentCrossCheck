#!/usr/bin/env python3
"""
================================================================================
SND@LHC Comprehensive Pre-Selection Observable & Canvas Generator
================================================================================
High-throughput pre-tracking distribution extractor supporting 1D, 2D, and
TProfile observables across:
1. Monte Carlo Trident Signal (Rock, Target, Muon System, Total Combined)
2. Monte Carlo Trident Background (unboosted single-muon events)
3. Passing Muon Simulation (optional, via '--pmu-input')
4. Real Experimental Collision Data (optional, via '--data-input')

Uses:
- Fast C++ JIT compilation for metric calculation
- Separate modular configuration file ('config/preselection_histograms_config.py')
- Multiprocessing (spawn context) over ROOT files
- Automated dual-pad TCanvas rendering with efficiency & rejection curves
- Robust hierarchical directory creation and atomic file writing (safe on EOS)

Author: SND@LHC Collaboration
================================================================================
"""

import os
import sys
import glob
import time
import math
import argparse
import tempfile
import shutil
import multiprocessing as mp
from typing import Dict, Any, List, Tuple

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.preselection_histograms_config import HIST_CONFIGS_1D, HIST_CONFIGS_2D, PROFILE_CONFIGS

# Load SND@LHC analysis libraries
for lib in ["libBase", "libShipData", "libshipLHC", "libsnd_analysis_tools"]:
    ROOT.gSystem.Load(lib)

sndsw_path = os.environ.get("SNDSW_ROOT", "")
if sndsw_path:
    scifi_tools_h = os.path.join(sndsw_path, "analysis/tools/sndSciFiTools.h")
    if os.path.exists(scifi_tools_h):
        ROOT.gInterpreter.ProcessLine(f'#include "{scifi_tools_h}"')

# Declare Fast C++ Metric Extractor with 1D arrays and safe null handling
if not hasattr(ROOT, "extractAllPreselectionMetricsFast"):
    ROOT.gInterpreter.Declare("""
    #include "sndScifiHit.h"
    #include "MuFilterHit.h"
    #include "TClonesArray.h"
    #include "sndSciFiTools.h"
    #include <vector>
    #include <algorithm>
    #include <cmath>
    #include <set>
    #include <cstring>

    struct AllPreselectionMetrics {
        // 1. SciFi Global
        int scifi_nhits;
        double scifi_sum_qdc;
        double scifi_max_qdc;
        double scifi_mean_qdc;
        int scifi_planes_hit;
        int scifi_stations_hit;

        // 2. SciFi Density Weights
        double scifi_max_hit_density;
        double scifi_sum_hit_density;
        double scifi_max_qdc_density;
        double scifi_sum_qdc_density;

        // 3. SciFi Stations (1 to 5)
        int scifi_nhits_st[5];
        double scifi_qdc_st[5];

        // 4. SciFi Individual Planes (10 planes: H and V for stations 1..5)
        int scifi_nhits_plane_h[5];
        int scifi_nhits_plane_v[5];
        double scifi_qdc_plane_h[5];
        double scifi_qdc_plane_v[5];
        int scifi_max_nhits_plane;
        double scifi_max_qdc_plane;

        // 5. SciFi Ratios & Asymmetries
        double scifi_qdc_ratio_down_up;
        double scifi_nhits_ratio_down_up;
        double scifi_qdc_ratio_st5_st1;
        double scifi_qdc_asym_xy;
        double scifi_nhits_asym_xy;

        // 6. Veto (System 1)
        int veto_nhits;
        double veto_sum_qdc;
        double veto_max_qdc;
        int veto_planes_hit;

        // 7. Upstream MuFilter (US, System 2, 5 stations)
        int us_nhits;
        double us_sum_qdc;
        double us_max_qdc;
        int us_planes_hit;
        int us_nhits_st[5];
        double us_qdc_st[5];

        // 8. Downstream MuFilter (DS, System 3, 4 stations, 7 planes)
        int ds_nhits;
        double ds_sum_qdc;
        double ds_max_qdc;
        int ds_planes_hit;
        int ds_stations_hit;
        int ds_deepest_station;
        int ds_deepest_plane;
        int ds_max_nhits_plane;
        double ds_max_qdc_plane;

        // 9. DS Individual Planes (Stations 1-3 H+V, Station 4 V)
        int ds_nhits_plane_h[4];
        int ds_nhits_plane_v[4];
        double ds_qdc_plane_h[4];
        double ds_qdc_plane_v[4];

        // 10. DS Ratios & Penetration
        double ds_qdc_ratio_back_front;
        double ds_nhits_ratio_back_front;
        double ds_qdc_ratio_ds4_ds1;

        // 11. Cross-System Global
        int total_nhits;
        double total_sum_qdc;
        double ratio_ds_to_scifi_qdc;
        double ratio_ds_to_scifi_nhits;
        double ratio_us_to_scifi_qdc;
        double ratio_mufi_to_scifi_qdc;
    };

    bool validateHitFast(sndScifiHit *aHit, int ref_station, bool ref_orientation) {
        if (!aHit || !aHit->isValid()) return false;
        if (aHit->GetStation() != ref_station) return false;
        if (aHit->isVertical() != ref_orientation) return false;
        return true;
    }

    double qdcDensityFast(int reference_SiPM, TClonesArray* scifi_hits, int radius) {
        if (!scifi_hits) return 0.0;
        double qdc_density = 0.0;
        bool orientation = (int(reference_SiPM / 100000) % 10 == 1);
        int ref_station = reference_SiPM / 1000000;
        int referenceChannel = snd::analysis_tools::calculateSiPMNumber(reference_SiPM);

        int entries = scifi_hits->GetEntries();
        for (int i = 0; i < entries; ++i) {
            sndScifiHit* hit = (sndScifiHit*)scifi_hits->At(i);
            if (!hit || !validateHitFast(hit, ref_station, orientation)) continue;
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

    AllPreselectionMetrics extractAllPreselectionMetricsFast(TClonesArray* scifi_hits, TClonesArray* mufi_hits, double density_radius = 40.0) {
        AllPreselectionMetrics m;
        std::memset(&m, 0, sizeof(AllPreselectionMetrics));

        // A. Process SciFi Hits
        if (scifi_hits) {
            int n_sf = scifi_hits->GetEntries();
            m.scifi_nhits = n_sf;
            std::set<int> sf_planes, sf_stations;
            double qdc_x = 0.0, qdc_y = 0.0;
            int hits_x = 0, hits_y = 0;

            for (int i = 0; i < n_sf; ++i) {
                sndScifiHit* h = (sndScifiHit*)scifi_hits->At(i);
                if (!h || !h->isValid()) continue;

                double qdc = h->GetSignal();
                m.scifi_sum_qdc += qdc;
                if (qdc > m.scifi_max_qdc) m.scifi_max_qdc = qdc;

                int st = h->GetStation(); // 1..5
                int is_vert = h->isVertical() ? 1 : 0; // 1 = XZ, 0 = YZ

                if (st >= 1 && st <= 5) {
                    sf_stations.insert(st);
                    sf_planes.insert((st - 1) * 2 + is_vert);
                    m.scifi_nhits_st[st - 1] += 1;
                    m.scifi_qdc_st[st - 1] += qdc;
                    if (is_vert == 1) {
                        m.scifi_nhits_plane_v[st - 1] += 1;
                        m.scifi_qdc_plane_v[st - 1] += qdc;
                        qdc_x += qdc; hits_x += 1;
                    } else {
                        m.scifi_nhits_plane_h[st - 1] += 1;
                        m.scifi_qdc_plane_h[st - 1] += qdc;
                        qdc_y += qdc; hits_y += 1;
                    }
                }

                double hit_w = (double)snd::analysis_tools::densityScifi(h->GetChannelID(), *scifi_hits, density_radius, 1000000.0, false);
                m.scifi_sum_hit_density += hit_w;
                if (hit_w > m.scifi_max_hit_density) m.scifi_max_hit_density = hit_w;

                double qdc_w = qdcDensityFast(h->GetChannelID(), scifi_hits, density_radius);
                m.scifi_sum_qdc_density += qdc_w;
                if (qdc_w > m.scifi_max_qdc_density) m.scifi_max_qdc_density = qdc_w;
            }

            if (n_sf > 0) m.scifi_mean_qdc = m.scifi_sum_qdc / n_sf;
            m.scifi_planes_hit = sf_planes.size();
            m.scifi_stations_hit = sf_stations.size();

            for (int s = 0; s < 5; ++s) {
                if (m.scifi_nhits_plane_h[s] > m.scifi_max_nhits_plane) m.scifi_max_nhits_plane = m.scifi_nhits_plane_h[s];
                if (m.scifi_nhits_plane_v[s] > m.scifi_max_nhits_plane) m.scifi_max_nhits_plane = m.scifi_nhits_plane_v[s];
                if (m.scifi_qdc_plane_h[s] > m.scifi_max_qdc_plane) m.scifi_max_qdc_plane = m.scifi_qdc_plane_h[s];
                if (m.scifi_qdc_plane_v[s] > m.scifi_max_qdc_plane) m.scifi_max_qdc_plane = m.scifi_qdc_plane_v[s];
            }

            double up_qdc = m.scifi_qdc_st[0] + m.scifi_qdc_st[1];
            double down_qdc = m.scifi_qdc_st[3] + m.scifi_qdc_st[4];
            m.scifi_qdc_ratio_down_up = (down_qdc) / (up_qdc + 1e-4);

            double up_hits = m.scifi_nhits_st[0] + m.scifi_nhits_st[1];
            double down_hits = m.scifi_nhits_st[3] + m.scifi_nhits_st[4];
            m.scifi_nhits_ratio_down_up = (down_hits) / (up_hits + 1e-4);

            m.scifi_qdc_ratio_st5_st1 = (m.scifi_qdc_st[4]) / (m.scifi_qdc_st[0] + 1e-4);

            if (qdc_x + qdc_y > 0) m.scifi_qdc_asym_xy = (qdc_x - qdc_y) / (qdc_x + qdc_y);
            if (hits_x + hits_y > 0) m.scifi_nhits_asym_xy = double(hits_x - hits_y) / double(hits_x + hits_y);
        }

        // B. Process MuFilter Hits (Veto, US, DS)
        if (mufi_hits) {
            int n_mf = mufi_hits->GetEntries();
            std::set<int> veto_planes, us_planes, ds_planes, ds_stations;

            for (int i = 0; i < n_mf; ++i) {
                MuFilterHit* h = (MuFilterHit*)mufi_hits->At(i);
                if (!h || !h->isValid()) continue;

                int sys = h->GetSystem(); // 1=Veto, 2=US, 3=DS
                int det_id = h->GetDetectorID();
                int plane_idx = (det_id % 10000) / 1000;
                double qdc = h->GetSignal();
                bool is_vert = h->isVertical();
                int o_idx = is_vert ? 1 : 0;

                if (sys == 1) { // Veto
                    m.veto_nhits += 1;
                    m.veto_sum_qdc += qdc;
                    if (qdc > m.veto_max_qdc) m.veto_max_qdc = qdc;
                    veto_planes.insert(plane_idx);
                }
                else if (sys == 2) { // US (5 stations)
                    m.us_nhits += 1;
                    m.us_sum_qdc += qdc;
                    if (qdc > m.us_max_qdc) m.us_max_qdc = qdc;
                    if (plane_idx >= 0 && plane_idx < 5) {
                        m.us_nhits_st[plane_idx] += 1;
                        m.us_qdc_st[plane_idx] += qdc;
                        us_planes.insert(plane_idx);
                    }
                }
                else if (sys == 3) { // DS (4 stations, 7 planes)
                    m.ds_nhits += 1;
                    m.ds_sum_qdc += qdc;
                    if (qdc > m.ds_max_qdc) m.ds_max_qdc = qdc;
                    if (plane_idx >= 0 && plane_idx < 4) {
                        ds_stations.insert(plane_idx + 1);
                        if ((plane_idx + 1) > m.ds_deepest_station) m.ds_deepest_station = plane_idx + 1;

                        int global_ds_plane = (plane_idx < 3) ? (plane_idx * 2 + o_idx) : 6;
                        ds_planes.insert(global_ds_plane);
                        if ((global_ds_plane + 1) > m.ds_deepest_plane) m.ds_deepest_plane = global_ds_plane + 1;

                        if (is_vert) {
                            m.ds_nhits_plane_v[plane_idx] += 1;
                            m.ds_qdc_plane_v[plane_idx] += qdc;
                        } else {
                            m.ds_nhits_plane_h[plane_idx] += 1;
                            m.ds_qdc_plane_h[plane_idx] += qdc;
                        }
                    }
                }
            }

            m.veto_planes_hit = veto_planes.size();
            m.us_planes_hit = us_planes.size();
            m.ds_planes_hit = ds_planes.size();
            m.ds_stations_hit = ds_stations.size();

            for (int s = 0; s < 4; ++s) {
                if (m.ds_nhits_plane_h[s] > m.ds_max_nhits_plane) m.ds_max_nhits_plane = m.ds_nhits_plane_h[s];
                if (m.ds_nhits_plane_v[s] > m.ds_max_nhits_plane) m.ds_max_nhits_plane = m.ds_nhits_plane_v[s];
                if (m.ds_qdc_plane_h[s] > m.ds_max_qdc_plane) m.ds_max_qdc_plane = m.ds_qdc_plane_h[s];
                if (m.ds_qdc_plane_v[s] > m.ds_max_qdc_plane) m.ds_max_qdc_plane = m.ds_qdc_plane_v[s];
            }

            double front_ds_qdc = (m.ds_qdc_plane_h[0] + m.ds_qdc_plane_v[0] + m.ds_qdc_plane_h[1] + m.ds_qdc_plane_v[1]);
            double back_ds_qdc  = (m.ds_qdc_plane_h[2] + m.ds_qdc_plane_v[2] + m.ds_qdc_plane_v[3]);
            m.ds_qdc_ratio_back_front = (back_ds_qdc) / (front_ds_qdc + 1e-4);

            double front_ds_hits = (m.ds_nhits_plane_h[0] + m.ds_nhits_plane_v[0] + m.ds_nhits_plane_h[1] + m.ds_nhits_plane_v[1]);
            double back_ds_hits  = (m.ds_nhits_plane_h[2] + m.ds_nhits_plane_v[2] + m.ds_nhits_plane_v[3]);
            m.ds_nhits_ratio_back_front = (back_ds_hits) / (front_ds_hits + 1e-4);

            double ds1_qdc = m.ds_qdc_plane_h[0] + m.ds_qdc_plane_v[0];
            double ds4_qdc = m.ds_qdc_plane_v[3];
            m.ds_qdc_ratio_ds4_ds1 = (ds4_qdc) / (ds1_qdc + 1e-4);
        }

        // C. Cross-System Global Metrics
        m.total_nhits = m.scifi_nhits + m.us_nhits + m.ds_nhits + m.veto_nhits;
        m.total_sum_qdc = m.scifi_sum_qdc + m.us_sum_qdc + m.ds_sum_qdc + m.veto_sum_qdc;

        m.ratio_ds_to_scifi_qdc = m.ds_sum_qdc / (m.scifi_sum_qdc + 1e-4);
        m.ratio_ds_to_scifi_nhits = double(m.ds_nhits) / double(m.scifi_nhits + 1e-4);
        m.ratio_us_to_scifi_qdc = m.us_sum_qdc / (m.scifi_sum_qdc + 1e-4);
        m.ratio_mufi_to_scifi_qdc = (m.us_sum_qdc + m.ds_sum_qdc) / (m.scifi_sum_qdc + 1e-4);

        return m;
    }
    """)

def get_or_create_dir(parent_tfile_or_dir, path_str: str):
    """
    Safely navigates or creates nested directory hierarchy without slashes bug in ROOT.
    """
    parts = [p for p in path_str.split("/") if p]
    curr = parent_tfile_or_dir
    for part in parts:
        sub = curr.GetDirectory(part)
        if not sub:
            sub = curr.mkdir(part)
        curr = sub
    return curr

def metrics_to_dict(m) -> Dict[str, float]:
    """Flattens the C++ AllPreselectionMetrics struct into a fast Python lookup dict."""
    d = {
        "scifi_nhits": int(m.scifi_nhits),
        "scifi_sum_qdc": float(m.scifi_sum_qdc),
        "scifi_max_qdc": float(m.scifi_max_qdc),
        "scifi_mean_qdc": float(m.scifi_mean_qdc),
        "scifi_planes_hit": int(m.scifi_planes_hit),
        "scifi_stations_hit": int(m.scifi_stations_hit),
        "scifi_max_hit_density": float(m.scifi_max_hit_density),
        "scifi_sum_hit_density": float(m.scifi_sum_hit_density),
        "scifi_max_qdc_density": float(m.scifi_max_qdc_density),
        "scifi_sum_qdc_density": float(m.scifi_sum_qdc_density),
        "scifi_max_nhits_plane": int(m.scifi_max_nhits_plane),
        "scifi_max_qdc_plane": float(m.scifi_max_qdc_plane),
        "scifi_qdc_ratio_down_up": float(m.scifi_qdc_ratio_down_up),
        "scifi_nhits_ratio_down_up": float(m.scifi_nhits_ratio_down_up),
        "scifi_qdc_ratio_st5_st1": float(m.scifi_qdc_ratio_st5_st1),
        "scifi_qdc_asym_xy": float(m.scifi_qdc_asym_xy),
        "scifi_nhits_asym_xy": float(m.scifi_nhits_asym_xy),
        "veto_nhits": int(m.veto_nhits),
        "veto_sum_qdc": float(m.veto_sum_qdc),
        "veto_planes_hit": int(m.veto_planes_hit),
        "veto_max_qdc": float(m.veto_max_qdc),
        "us_nhits": int(m.us_nhits),
        "us_sum_qdc": float(m.us_sum_qdc),
        "us_planes_hit": int(m.us_planes_hit),
        "us_max_qdc": float(m.us_max_qdc),
        "ds_nhits": int(m.ds_nhits),
        "ds_sum_qdc": float(m.ds_sum_qdc),
        "ds_max_qdc": float(m.ds_max_qdc),
        "ds_planes_hit": int(m.ds_planes_hit),
        "ds_stations_hit": int(m.ds_stations_hit),
        "ds_deepest_station": int(m.ds_deepest_station),
        "ds_deepest_plane": int(m.ds_deepest_plane),
        "ds_max_nhits_plane": int(m.ds_max_nhits_plane),
        "ds_max_qdc_plane": float(m.ds_max_qdc_plane),
        "ds_qdc_ratio_back_front": float(m.ds_qdc_ratio_back_front),
        "ds_nhits_ratio_back_front": float(m.ds_nhits_ratio_back_front),
        "ds_qdc_ratio_ds4_ds1": float(m.ds_qdc_ratio_ds4_ds1),
        "total_nhits": int(m.total_nhits),
        "total_sum_qdc": float(m.total_sum_qdc),
        "ratio_ds_to_scifi_qdc": float(m.ratio_ds_to_scifi_qdc),
        "ratio_ds_to_scifi_nhits": float(m.ratio_ds_to_scifi_nhits),
        "ratio_us_to_scifi_qdc": float(m.ratio_us_to_scifi_qdc),
        "ratio_mufi_to_scifi_qdc": float(m.ratio_mufi_to_scifi_qdc),
    }

    # Station Profiles
    for s in range(5):
        d[f"scifi_nhits_st{s+1}"] = int(m.scifi_nhits_st[s])
        d[f"scifi_qdc_st{s+1}"] = float(m.scifi_qdc_st[s])
        d[f"scifi_nhits_st{s+1}_h"] = int(m.scifi_nhits_plane_h[s])
        d[f"scifi_nhits_st{s+1}_v"] = int(m.scifi_nhits_plane_v[s])
        d[f"scifi_qdc_st{s+1}_h"] = float(m.scifi_qdc_plane_h[s])
        d[f"scifi_qdc_st{s+1}_v"] = float(m.scifi_qdc_plane_v[s])
        d[f"us_nhits_st{s+1}"] = int(m.us_nhits_st[s])
        d[f"us_qdc_st{s+1}"] = float(m.us_qdc_st[s])

    # DS Planes (7 Planes)
    d["ds_nhits_st1_h"] = int(m.ds_nhits_plane_h[0]); d["ds_nhits_st1_v"] = int(m.ds_nhits_plane_v[0])
    d["ds_nhits_st2_h"] = int(m.ds_nhits_plane_h[1]); d["ds_nhits_st2_v"] = int(m.ds_nhits_plane_v[1])
    d["ds_nhits_st3_h"] = int(m.ds_nhits_plane_h[2]); d["ds_nhits_st3_v"] = int(m.ds_nhits_plane_v[2])
    d["ds_nhits_st4_v"] = int(m.ds_nhits_plane_v[3])

    d["ds_qdc_st1_h"] = float(m.ds_qdc_plane_h[0]); d["ds_qdc_st1_v"] = float(m.ds_qdc_plane_v[0])
    d["ds_qdc_st2_h"] = float(m.ds_qdc_plane_h[1]); d["ds_qdc_st2_v"] = float(m.ds_qdc_plane_v[1])
    d["ds_qdc_st3_h"] = float(m.ds_qdc_plane_h[2]); d["ds_qdc_st3_v"] = float(m.ds_qdc_plane_v[2])
    d["ds_qdc_st4_v"] = float(m.ds_qdc_plane_v[3])

    return d

def create_efficiency_graph(h_raw, cut_dir="<="):
    """Computes TGraphAsymmErrors for cumulative efficiency with binomial error propagation."""
    gr = ROOT.TGraphAsymmErrors()
    ROOT.SetOwnership(gr, False)
    tot = h_raw.Integral()
    if tot <= 0: return gr

    nb = h_raw.GetNbinsX()
    var_tot = sum([h_raw.GetBinError(i)**2 for i in range(1, nb + 1)])

    for b in range(1, nb + 1):
        x = h_raw.GetBinLowEdge(b) + h_raw.GetBinWidth(b) if cut_dir == "<=" else h_raw.GetBinLowEdge(b)
        if cut_dir == "<=":
            pass_val = h_raw.Integral(1, b)
            var_pass = sum([h_raw.GetBinError(i)**2 for i in range(1, b + 1)])
        else:
            pass_val = h_raw.Integral(b, nb)
            var_pass = sum([h_raw.GetBinError(i)**2 for i in range(b, nb + 1)])

        eff = pass_val / tot
        var_eff = (var_pass * (1.0 - 2.0 * eff) + (eff**2) * var_tot) / (tot**2) if tot > 0 else 0.0
        err = math.sqrt(max(0.0, var_eff))
        pt = gr.GetN()
        gr.SetPoint(pt, x, eff)
        gr.SetPointError(pt, 0.0, 0.0, min(eff, err), min(1.0 - eff, err))
    return gr

def create_rejection_graph(h_raw, cut_dir="<="):
    """Computes TGraphAsymmErrors for cumulative rejection (1 - FPR)."""
    gr = ROOT.TGraphAsymmErrors()
    ROOT.SetOwnership(gr, False)
    tot = h_raw.Integral()
    if tot <= 0: return gr

    nb = h_raw.GetNbinsX()
    var_tot = sum([h_raw.GetBinError(i)**2 for i in range(1, nb + 1)])

    for b in range(1, nb + 1):
        x = h_raw.GetBinLowEdge(b) + h_raw.GetBinWidth(b) if cut_dir == "<=" else h_raw.GetBinLowEdge(b)
        if cut_dir == "<=":
            pass_val = h_raw.Integral(1, b)
            var_pass = sum([h_raw.GetBinError(i)**2 for i in range(1, b + 1)])
        else:
            pass_val = h_raw.Integral(b, nb)
            var_pass = sum([h_raw.GetBinError(i)**2 for i in range(b, nb + 1)])

        fpr = pass_val / tot
        rej = 1.0 - fpr
        var_fpr = (var_pass * (1.0 - 2.0 * fpr) + (fpr**2) * var_tot) / (tot**2) if tot > 0 else 0.0
        err = math.sqrt(max(0.0, var_fpr))
        pt = gr.GetN()
        gr.SetPoint(pt, x, rej)
        gr.SetPointError(pt, 0.0, 0.0, min(rej, err), min(1.0 - rej, err))
    return gr

def create_superimposed_1d_canvas(var_name, h_rock, h_target, h_mufi, h_bkg, h_data=None, h_pmu=None, cut_dir="<="):
    """Generates dual-pad publication canvas with log-y distributions and efficiency/rejection curve."""
    c = ROOT.TCanvas(f"c_{var_name}", f"Preselection: {var_name}", 880, 800)
    ROOT.SetOwnership(c, False)

    pad1 = ROOT.TPad(f"pad1_{var_name}", "Top", 0.0, 0.30, 1.0, 1.0)
    pad2 = ROOT.TPad(f"pad2_{var_name}", "Bottom", 0.0, 0.0, 1.0, 0.30)
    ROOT.SetOwnership(pad1, False)
    ROOT.SetOwnership(pad2, False)

    pad1.SetBottomMargin(0.03); pad1.SetTopMargin(0.08); pad1.SetLeftMargin(0.12); pad1.SetRightMargin(0.05); pad1.SetLogy(1); pad1.Draw()
    pad2.SetTopMargin(0.03); pad2.SetBottomMargin(0.28); pad2.SetLeftMargin(0.12); pad2.SetRightMargin(0.05); pad2.Draw()

    # 1. Draw top pad
    pad1.cd()
    raw_dict = {
        "rock": (h_rock, ROOT.kRed + 1, "Rock Trident Signal", "FL"),
        "target": (h_target, ROOT.kAzure + 1, "Target Trident Signal", "FL"),
        "mufi": (h_mufi, ROOT.kTeal + 2, "Muon System Trident Signal", "FL"),
        "bkg": (h_bkg, ROOT.kGray + 2, "Trident MC Bkg", "L"),
    }
    if h_data and h_data.GetEntries() > 0:
        raw_dict["data"] = (h_data, ROOT.kBlack, f"Data (N={h_data.GetEntries():,.0f})", "EP")
    if h_pmu and h_pmu.GetEntries() > 0:
        raw_dict["pmu"] = (h_pmu, ROOT.kOrange + 2, f"Passing #mu MC (N={h_pmu.GetEntries():,.0f})", "L")

    samples = {}
    for k, (h_orig, color, label, draw_opt) in raw_dict.items():
        if h_orig and h_orig.GetEntries() > 0:
            h_cl = h_orig.Clone(f"{h_orig.GetName()}_norm_{var_name}")
            ROOT.SetOwnership(h_cl, False)
            if h_cl.Integral() > 0: h_cl.Scale(1.0 / h_cl.Integral())
            h_cl.SetLineColor(color); h_cl.SetLineWidth(2); h_cl.SetStats(0)
            if "F" in draw_opt: h_cl.SetFillColorAlpha(color, 0.25)
            if k == "bkg": h_cl.SetLineStyle(2)
            if k == "pmu": h_cl.SetLineStyle(7)
            if k == "data": h_cl.SetMarkerColor(ROOT.kBlack); h_cl.SetMarkerStyle(20); h_cl.SetMarkerSize(0.85)
            samples[k] = (h_cl, h_orig, color, label, draw_opt)

    max_y = max([s[0].GetMaximum() for s in samples.values()]) * 6.0 if samples else 1.0
    for s in samples.values():
        s[0].SetMaximum(max(1.0, max_y)); s[0].SetMinimum(1e-4)

    first_draw = True
    for k in ["bkg", "pmu", "rock", "target", "mufi", "data"]:
        if k in samples:
            h_s, _, _, _, _ = samples[k]
            opt = "HIST" if k != "data" else "E1"
            if not first_draw:
                opt += " SAME"
            h_s.Draw(opt)
            first_draw = False

    leg = ROOT.TLegend(0.48, 0.62, 0.93, 0.89)
    ROOT.SetOwnership(leg, False)
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.031)
    for k in ["data", "rock", "target", "mufi", "bkg", "pmu"]:
        if k in samples:
            h_cl, h_orig, _, label, draw_opt = samples[k]
            ent_str = f" (N={h_orig.GetEntries():,.0f})" if k != "data" else ""
            leg.AddEntry(h_cl, f"{label}{ent_str}", draw_opt)
    leg.Draw()

    # 2. Draw bottom pad (Efficiency / Rejection)
    pad2.cd(); pad2.SetGridy(1)
    ref_h = None
    for cand in [h_rock, h_target, h_mufi, h_bkg, h_data, h_pmu]:
        if cand and cand.GetNbinsX() > 0:
            ref_h = cand
            break

    if ref_h:
        frame = ROOT.TH1D(f"frame_{var_name}", f";{ref_h.GetXaxis().GetTitle()};Eff / Rej", ref_h.GetNbinsX(), ref_h.GetXaxis().GetXmin(), ref_h.GetXaxis().GetXmax())
        ROOT.SetOwnership(frame, False)
        frame.SetStats(0); frame.GetYaxis().SetRangeUser(-0.05, 1.15); frame.GetYaxis().SetNdivisions(505)
        frame.GetYaxis().SetTitleSize(0.09); frame.GetYaxis().SetTitleOffset(0.55); frame.GetYaxis().SetLabelSize(0.08)
        frame.GetXaxis().SetTitleSize(0.10); frame.GetXaxis().SetTitleOffset(1.1); frame.GetXaxis().SetLabelSize(0.08)
        frame.Draw("AXIS")

    gr_rock = create_efficiency_graph(h_rock, cut_dir); gr_rock.SetLineColor(ROOT.kRed + 1); gr_rock.SetLineWidth(2)
    if gr_rock.GetN() > 0: gr_rock.Draw("L SAME")

    gr_target = create_efficiency_graph(h_target, cut_dir); gr_target.SetLineColor(ROOT.kAzure + 1); gr_target.SetLineWidth(2)
    if gr_target.GetN() > 0: gr_target.Draw("L SAME")

    gr_mufi = create_efficiency_graph(h_mufi, cut_dir); gr_mufi.SetLineColor(ROOT.kTeal + 2); gr_mufi.SetLineWidth(2)
    if gr_mufi.GetN() > 0: gr_mufi.Draw("L SAME")

    h_eval_bkg = h_bkg; bkg_col = ROOT.kGray + 2
    if h_data and h_data.GetEntries() > 0: h_eval_bkg = h_data; bkg_col = ROOT.kBlack
    elif h_pmu and h_pmu.GetEntries() > 0: h_eval_bkg = h_pmu; bkg_col = ROOT.kOrange + 2

    gr_bkg = create_rejection_graph(h_eval_bkg, cut_dir); gr_bkg.SetLineColor(bkg_col); gr_bkg.SetLineWidth(2); gr_bkg.SetLineStyle(2)
    if gr_bkg.GetN() > 0: gr_bkg.Draw("L SAME")

    leg2 = ROOT.TLegend(0.48, 0.32, 0.93, 0.65)
    ROOT.SetOwnership(leg2, False)
    leg2.SetBorderSize(0); leg2.SetFillStyle(0); leg2.SetTextSize(0.065)
    if gr_rock.GetN() > 0: leg2.AddEntry(gr_rock, f"Rock Eff ({cut_dir})", "L")
    if gr_target.GetN() > 0: leg2.AddEntry(gr_target, f"Target Eff ({cut_dir})", "L")
    if gr_mufi.GetN() > 0: leg2.AddEntry(gr_mufi, f"MuFilter Eff ({cut_dir})", "L")
    if gr_bkg.GetN() > 0: leg2.AddEntry(gr_bkg, f"Bkg Rej ({cut_dir})", "L")
    leg2.Draw()

    c.cd()
    c.Modified()
    c.Update()
    return c

def process_single_file_worker(args_tuple):
    """Worker function to process one ROOT file into a temporary output ROOT file."""
    in_file, out_file, stream_type, max_events, radius = args_tuple
    t0 = time.time()

    ROOT.TH1.AddDirectory(False)
    try:
        f_in = ROOT.TFile.Open(in_file, "READ")
    except Exception as e:
        print(f"Error opening '{in_file}': {e}")
        return None

    if not f_in or f_in.IsZombie():
        print(f"Error opening '{in_file}'")
        if f_in: f_in.Close()
        return None

    tree = f_in.Get("cbmsim") or f_in.Get("rawConv")
    if not tree:
        f_in.Close()
        return None

    # Book local 1D, 2D, and Profile histograms
    h1_dict, h2_dict, prof_dict = {}, {}, {}
    cats = ["rock", "target", "mufi", "allsig", "bkg"] if stream_type == "trident_mc" else [stream_type]

    for cat in cats:
        h1_dict[cat] = {}
        for var_name, title, nbins, xmin, xmax, _, _ in HIST_CONFIGS_1D:
            h = ROOT.TH1D(f"h_{cat}_{var_name}", f"{cat.capitalize()}: {title}", nbins, xmin, xmax)
            h.SetDirectory(0); h.Sumw2()
            h1_dict[cat][var_name] = h

        h2_dict[cat] = {}
        for h2_name, title, nx, xmin, xmax, ny, ymin, ymax, _, _, _ in HIST_CONFIGS_2D:
            h2 = ROOT.TH2D(f"h2_{cat}_{h2_name}", f"{cat.capitalize()}: {title}", nx, xmin, xmax, ny, ymin, ymax)
            h2.SetDirectory(0); h2.Sumw2()
            h2_dict[cat][h2_name] = h2

        prof_dict[cat] = {}
        for prof_name, title, nx, xmin, xmax, _ in PROFILE_CONFIGS:
            prof = ROOT.TProfile(f"prof_{cat}_{prof_name}", f"{cat.capitalize()}: {title}", nx, xmin, xmax)
            prof.SetDirectory(0); prof.Sumw2()
            prof_dict[cat][prof_name] = prof

    n_entries = tree.GetEntries()
    n_scan = n_entries if (max_events <= 0 or max_events > n_entries) else max_events
    sig_count, bkg_count = 0, 0

    for i in range(n_scan):
        tree.GetEntry(i)

        is_sig = False
        reg_id = -1
        vtx_z = -9999.0
        mc_wt = 1.0

        if stream_type == "trident_mc":
            if hasattr(tree, "is_signal"):
                try: is_sig = bool(tree.is_signal)
                except Exception: is_sig = False
            elif hasattr(tree, "MCTrack") and tree.MCTrack and tree.MCTrack.GetEntries() > 0:
                is_sig = True
            else:
                is_sig = True

            if hasattr(tree, "region_type"):
                try: reg_id = int(tree.region_type)
                except Exception: reg_id = -1

            if hasattr(tree, "vtx_z"):
                try: vtx_z = float(tree.vtx_z)
                except Exception: vtx_z = -9999.0
            elif hasattr(tree, "MCTrack") and tree.MCTrack and tree.MCTrack.GetEntries() > 0:
                try: vtx_z = float(tree.MCTrack.At(0).GetStartZ())
                except Exception: vtx_z = -9999.0

            if hasattr(tree, "mc_weight"):
                try: mc_wt = float(tree.mc_weight)
                except Exception: mc_wt = 1.0
            elif hasattr(tree, "MCEventHeader") and tree.MCEventHeader:
                try: mc_wt = float(tree.MCEventHeader.GetWeight())
                except Exception: mc_wt = 1.0

            # Determine signal region
            if reg_id == 1:
                is_rock = is_sig; is_target = False; is_mufi = False
            elif reg_id == 2:
                is_rock = False; is_target = is_sig; is_mufi = False
            elif reg_id == 3:
                is_rock = False; is_target = False; is_mufi = is_sig
            else:
                is_rock = is_sig and (vtx_z < 260.0 and vtx_z > -9000.0)
                is_target = is_sig and (260.0 <= vtx_z < 355.0)
                is_mufi = is_sig and (vtx_z >= 355.0)
                if is_sig and not (is_rock or is_target or is_mufi):
                    is_target = True
        elif stream_type == "pmu":
            if hasattr(tree, "mc_weight"):
                try: mc_wt = float(tree.mc_weight)
                except Exception: mc_wt = 1.0
            elif hasattr(tree, "MCEventHeader") and tree.MCEventHeader:
                try: mc_wt = float(tree.MCEventHeader.GetWeight())
                except Exception: mc_wt = 1.0
            else:
                mc_wt = 1.0
            is_rock, is_target, is_mufi = False, False, False
        else: # data
            mc_wt = 1.0
            is_rock, is_target, is_mufi = False, False, False

        if is_sig: sig_count += 1
        else: bkg_count += 1

        scifi_hits = getattr(tree, "Digi_ScifiHits", getattr(tree, "Digits_Scifi", None))
        mufi_hits = getattr(tree, "Digi_MuFilterHits", getattr(tree, "Digi_MuFilterHit", getattr(tree, "Digits_MuFilter", None)))

        m = ROOT.extractAllPreselectionMetricsFast(scifi_hits, mufi_hits, radius)
        f_dict = metrics_to_dict(m)

        # Helper to fill a category
        def fill_category(cat_name, wt):
            for v_name, _, _, _, _, _, _ in HIST_CONFIGS_1D:
                h1_dict[cat_name][v_name].Fill(f_dict[v_name], wt)

            for h2_name, _, _, _, _, _, _, _, x_var, y_var, _ in HIST_CONFIGS_2D:
                h2_dict[cat_name][h2_name].Fill(f_dict[x_var], f_dict[y_var], wt)

            # Profiles
            for s in range(5):
                prof_dict[cat_name]["prof_scifi_qdc_vs_station"].Fill(s + 1, f_dict[f"scifi_qdc_st{s+1}"], wt)
                prof_dict[cat_name]["prof_scifi_nhits_vs_station"].Fill(s + 1, f_dict[f"scifi_nhits_st{s+1}"], wt)

            for s in range(4):
                qdc_ds_st = f_dict.get(f"ds_qdc_st{s+1}_h", 0.0) + f_dict.get(f"ds_qdc_st{s+1}_v", 0.0)
                hits_ds_st = f_dict.get(f"ds_nhits_st{s+1}_h", 0) + f_dict.get(f"ds_nhits_st{s+1}_v", 0)
                prof_dict[cat_name]["prof_ds_qdc_vs_station"].Fill(s + 1, qdc_ds_st, wt)
                prof_dict[cat_name]["prof_ds_nhits_vs_station"].Fill(s + 1, hits_ds_st, wt)

            # Full 24-plane profile
            # Veto 1-2 (bin 1-2)
            prof_dict[cat_name]["prof_detector_longitudinal_qdc"].Fill(1, f_dict["veto_sum_qdc"] * 0.5, wt)
            prof_dict[cat_name]["prof_detector_longitudinal_qdc"].Fill(2, f_dict["veto_sum_qdc"] * 0.5, wt)
            # SciFi 1-10 (bin 3-12)
            for s in range(5):
                prof_dict[cat_name]["prof_detector_longitudinal_qdc"].Fill(3 + s*2, f_dict[f"scifi_qdc_st{s+1}_h"], wt)
                prof_dict[cat_name]["prof_detector_longitudinal_qdc"].Fill(4 + s*2, f_dict[f"scifi_qdc_st{s+1}_v"], wt)
            # US 1-5 (bin 13-17)
            for s in range(5):
                prof_dict[cat_name]["prof_detector_longitudinal_qdc"].Fill(13 + s, f_dict[f"us_qdc_st{s+1}"], wt)
            # DS 1-7 (bin 18-24)
            ds_plane_qdcs = [
                f_dict["ds_qdc_st1_h"], f_dict["ds_qdc_st1_v"],
                f_dict["ds_qdc_st2_h"], f_dict["ds_qdc_st2_v"],
                f_dict["ds_qdc_st3_h"], f_dict["ds_qdc_st3_v"],
                f_dict["ds_qdc_st4_v"]
            ]
            for p_idx, q_val in enumerate(ds_plane_qdcs):
                prof_dict[cat_name]["prof_detector_longitudinal_qdc"].Fill(18 + p_idx, q_val, wt)

        if stream_type == "trident_mc":
            if is_sig:
                fill_category("allsig", mc_wt)
                if is_rock: fill_category("rock", mc_wt)
                elif is_target: fill_category("target", mc_wt)
                elif is_mufi: fill_category("mufi", mc_wt)
            else:
                fill_category("bkg", mc_wt)
        else:
            fill_category(stream_type, mc_wt)

    f_in.Close()

    # Save worker temp output using hierarchical directory helper
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    f_out = ROOT.TFile.Open(out_file, "RECREATE")
    for cat in cats:
        d1 = get_or_create_dir(f_out, f"1D/{cat}")
        d1.cd()
        for v_name, _, _, _, _, _, _ in HIST_CONFIGS_1D:
            h1_dict[cat][v_name].Write()

        d2 = get_or_create_dir(f_out, f"2D/{cat}")
        d2.cd()
        for h2_name, _, _, _, _, _, _, _, _, _, _ in HIST_CONFIGS_2D:
            h2_dict[cat][h2_name].Write()

        dp = get_or_create_dir(f_out, f"Profiles/{cat}")
        dp.cd()
        for prof_name, _, _, _, _, _ in PROFILE_CONFIGS:
            prof_dict[cat][prof_name].Write()

    f_out.Close()
    elapsed = time.time() - t0
    print(f"  [{stream_type.upper()}] Done '{os.path.basename(in_file)}': {n_scan:,} ev in {elapsed:.1f}s")
    return n_scan, sig_count, bkg_count

def process_batch(file_list, stream_type, max_events, workers, radius, temp_dir):
    tasks = []
    temp_files = []
    for idx, f in enumerate(file_list):
        base = os.path.splitext(os.path.basename(f))[0]
        out_p = os.path.join(temp_dir, f"{stream_type}_{idx}_{base}.root")
        temp_files.append(out_p)
        tasks.append((f, out_p, stream_type, max_events, radius))

    tot_ev, tot_sig, tot_bkg = 0, 0, 0
    if workers > 1 and len(tasks) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(tasks)), maxtasksperchild=10) as pool:
            res = pool.map(process_single_file_worker, tasks)
        for r in res:
            if r: tot_ev += r[0]; tot_sig += r[1]; tot_bkg += r[2]
    else:
        for t in tasks:
            r = process_single_file_worker(t)
            if r: tot_ev += r[0]; tot_sig += r[1]; tot_bkg += r[2]

    return temp_files, tot_ev, tot_sig, tot_bkg

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive SND@LHC Preselection Observable & Canvas Generator"
    )
    parser.add_argument("-i", "--input", dest="input_pattern", default="/eos/user/i/idioniso/sndMuTri/data/trimuon_boost100/trimuon_digCPP-*.root", help="Input MC Trident ROOT files (wildcard or path)")
    parser.add_argument("--data-input", "--data", dest="data_pattern", default="", help="Optional Collision Data ROOT files (e.g. '$SND_DATA/.../sndsw_raw-*.root')")
    parser.add_argument("--pmu-input", "--pmu", dest="pmu_pattern", default="", help="Optional Passing Muon ROOT files (e.g. '.../pmu_*.root')")
    parser.add_argument("-o", "--output", dest="output_file", default="plots/preselection.root", help="Master output ROOT file")
    parser.add_argument("-n", "--max-events", dest="max_events", type=int, default=0, help="Max events per file (0 = all)")
    parser.add_argument("-j", "--workers", dest="workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("-r", "--radius", dest="radius", type=float, default=40.0, help="Hit/QDC density search radius (default: 40 SiPM channels)")

    args = parser.parse_args()

    mc_files = sorted(glob.glob(args.input_pattern)) or ([args.input_pattern] if os.path.exists(args.input_pattern) else [])
    data_files = sorted(glob.glob(args.data_pattern)) if args.data_pattern else []
    pmu_files = sorted(glob.glob(args.pmu_pattern)) if args.pmu_pattern else []

    if not mc_files and not data_files and not pmu_files:
        print("Error: No valid input files found for MC, Data, or Passing Muons.")
        sys.exit(1)

    print("=" * 80)
    print("SND@LHC COMPREHENSIVE PRESELECTION OBSERVABLE GENERATOR")
    print("=" * 80)
    print(f"MC Trident Files   : {len(mc_files)} files ({args.input_pattern if mc_files else 'None'})")
    print(f"Collision Data     : {len(data_files)} files ({args.data_pattern if data_files else 'None'})")
    print(f"Passing Muons      : {len(pmu_files)} files ({args.pmu_pattern if pmu_files else 'None'})")
    print(f"Total 1D Observables: {len(HIST_CONFIGS_1D)}")
    print(f"Total 2D Histograms : {len(HIST_CONFIGS_2D)}")
    print(f"Total Profiles      : {len(PROFILE_CONFIGS)}")
    print(f"Worker Processes   : {args.workers}")
    print(f"Output File        : {args.output_file}")
    print("=" * 80)

    t0_all = time.time()
    temp_dir = tempfile.mkdtemp(prefix="snd_presel_all_obs_")

    try:
        # 1. Process MC Tridents
        mc_temps, mc_ev, mc_sig, mc_bkg = [], 0, 0, 0
        if mc_files:
            print(f"\n--- [1/3] Processing {len(mc_files)} MC Trident Files ---")
            mc_temps, mc_ev, mc_sig, mc_bkg = process_batch(mc_files, "trident_mc", args.max_events, args.workers, args.radius, temp_dir)

        # 2. Process Data
        data_temps, data_ev = [], 0
        if data_files:
            print(f"\n--- [2/3] Processing {len(data_files)} Real Collision Data Files ---")
            data_temps, data_ev, _, _ = process_batch(data_files, "data", args.max_events, args.workers, args.radius, temp_dir)

        # 3. Process PMU
        pmu_temps, pmu_ev = [], 0
        if pmu_files:
            print(f"\n--- [3/3] Processing {len(pmu_files)} Passing Muon Files ---")
            pmu_temps, pmu_ev, _, _ = process_batch(pmu_files, "pmu", args.max_events, args.workers, args.radius, temp_dir)

        # 4. Master Accumulation & Final ROOT File Creation (via Local Temp File for Safe Atomic EOS Writing)
        print(f"\n--- Accumulating Histograms & Rendering Superimposed Canvases ---")
        ROOT.TH1.AddDirectory(False)

        local_master_path = os.path.join(temp_dir, "preselection_master.root")
        f_master = ROOT.TFile.Open(local_master_path, "RECREATE")

        all_cats = []
        if mc_files: all_cats.extend(["rock", "target", "mufi", "allsig", "bkg"])
        if data_files: all_cats.append("data")
        if pmu_files: all_cats.append("pmu")

        master_h1, master_h2, master_prof = {}, {}, {}
        for cat in all_cats:
            master_h1[cat] = {}
            for v_name, title, nbins, xmin, xmax, _, _ in HIST_CONFIGS_1D:
                h = ROOT.TH1D(f"h_{cat}_{v_name}", f"{cat.capitalize()}: {title}", nbins, xmin, xmax)
                h.SetDirectory(0); h.Sumw2()
                master_h1[cat][v_name] = h

            master_h2[cat] = {}
            for h2_name, title, nx, xmin, xmax, ny, ymin, ymax, _, _, _ in HIST_CONFIGS_2D:
                h2 = ROOT.TH2D(f"h2_{cat}_{h2_name}", f"{cat.capitalize()}: {title}", nx, xmin, xmax, ny, ymin, ymax)
                h2.SetDirectory(0); h2.Sumw2()
                master_h2[cat][h2_name] = h2

            master_prof[cat] = {}
            for prof_name, title, nx, xmin, xmax, _ in PROFILE_CONFIGS:
                prof = ROOT.TProfile(f"prof_{cat}_{prof_name}", f"{cat.capitalize()}: {title}", nx, xmin, xmax)
                prof.SetDirectory(0); prof.Sumw2()
                master_prof[cat][prof_name] = prof

        def accumulate_from_temps(temp_list, cat_map):
            for f_path in temp_list:
                f_t = ROOT.TFile.Open(f_path, "READ")
                if not f_t or f_t.IsZombie(): continue
                for cat_target, cat_src in cat_map.items():
                    for v_name, _, _, _, _, _, _ in HIST_CONFIGS_1D:
                        h_in = f_t.Get(f"1D/{cat_src}/h_{cat_src}_{v_name}")
                        if h_in: master_h1[cat_target][v_name].Add(h_in)

                    for h2_name, _, _, _, _, _, _, _, _, _, _ in HIST_CONFIGS_2D:
                        h2_in = f_t.Get(f"2D/{cat_src}/h2_{cat_src}_{h2_name}")
                        if h2_in: master_h2[cat_target][h2_name].Add(h2_in)

                    for prof_name, _, _, _, _, _ in PROFILE_CONFIGS:
                        p_in = f_t.Get(f"Profiles/{cat_src}/prof_{cat_src}_{prof_name}")
                        if p_in: master_prof[cat_target][prof_name].Add(p_in)
                f_t.Close()

        # Accumulate all streams
        if mc_files: accumulate_from_temps(mc_temps, {"rock": "rock", "target": "target", "mufi": "mufi", "allsig": "allsig", "bkg": "bkg"})
        if data_files: accumulate_from_temps(data_temps, {"data": "data"})
        if pmu_files: accumulate_from_temps(pmu_temps, {"pmu": "pmu"})

        # Category display names and directory mapping
        cat_dir_map = {
            "rock": "Histograms/Rock_Signal",
            "target": "Histograms/Target_Signal",
            "mufi": "Histograms/MuonSystem_Signal",
            "allsig": "Histograms/All_Signal",
            "bkg": "Histograms/Background",
            "data": "Histograms/Data",
            "pmu": "Histograms/Passing_Muon"
        }

        # Write Histograms to Master File
        for cat in all_cats:
            dir_std = cat_dir_map.get(cat, f"Histograms/{cat.capitalize()}")
            d_std = get_or_create_dir(f_master, dir_std)
            d_std.cd()
            for v_name, _, _, _, _, _, _ in HIST_CONFIGS_1D:
                master_h1[cat][v_name].Write()

            d_2d = get_or_create_dir(f_master, f"Histograms/2D/{cat.capitalize()}")
            d_2d.cd()
            for h2_name, _, _, _, _, _, _, _, _, _, _ in HIST_CONFIGS_2D:
                master_h2[cat][h2_name].Write()

            d_p = get_or_create_dir(f_master, f"Histograms/Profiles/{cat.capitalize()}")
            d_p.cd()
            for prof_name, _, _, _, _, _ in PROFILE_CONFIGS:
                master_prof[cat][prof_name].Write()

        # Write Superimposed Master 1D Canvases
        if mc_files:
            d_canvases_1d = get_or_create_dir(f_master, "Superimposed_Canvases/1D_Distributions")
            for v_name, _, _, _, _, cut_dir, _ in HIST_CONFIGS_1D:
                c = create_superimposed_1d_canvas(
                    v_name,
                    master_h1["rock"][v_name],
                    master_h1["target"][v_name],
                    master_h1["mufi"][v_name],
                    master_h1["bkg"][v_name],
                    master_h1["data"][v_name] if data_files else None,
                    master_h1["pmu"][v_name] if pmu_files else None,
                    cut_dir
                )
                d_canvases_1d.cd()
                c.Write()

        # Write Superimposed Profiles Canvases
        d_canvases_prof = get_or_create_dir(f_master, "Superimposed_Canvases/Profiles_Comparison")
        for prof_name, title, _, _, _, _ in PROFILE_CONFIGS:
            c_prof = ROOT.TCanvas(f"c_{prof_name}", f"Profile Comparison: {prof_name}", 800, 600)
            c_prof.cd()
            c_prof.SetGrid(1, 1)

            colors = {"rock": ROOT.kRed + 1, "target": ROOT.kAzure + 1, "mufi": ROOT.kTeal + 2, "allsig": ROOT.kMagenta + 1, "bkg": ROOT.kGray + 2, "data": ROOT.kBlack, "pmu": ROOT.kOrange + 2}
            leg_p = ROOT.TLegend(0.60, 0.68, 0.92, 0.89)
            leg_p.SetBorderSize(0); leg_p.SetFillStyle(0)

            has_drawn = False
            first = True
            for cat in all_cats:
                p = master_prof[cat].get(prof_name, None)
                if p and p.GetEntries() > 0:
                    h_proj = p.ProjectionX(f"{prof_name}_{cat}_proj", "E")
                    h_proj.SetDirectory(0)
                    h_proj.SetLineColor(colors.get(cat, ROOT.kBlack))
                    h_proj.SetMarkerColor(colors.get(cat, ROOT.kBlack))
                    h_proj.SetMarkerStyle(20)
                    h_proj.SetMarkerSize(0.8)
                    h_proj.SetLineWidth(2)
                    h_proj.SetStats(0)
                    draw_opt = "E1" if first else "E1 SAME"
                    h_proj.Draw(draw_opt)
                    leg_p.AddEntry(h_proj, cat.capitalize(), "lep")
                    first = False
                    has_drawn = True
            if has_drawn:
                leg_p.Draw()
                d_canvases_prof.cd()
                c_prof.Write()

        f_master.Close()

        # Atomically copy master file to final destination
        out_dir = os.path.dirname(os.path.abspath(args.output_file))
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(local_master_path, args.output_file)

        print(f"\n[SUCCESS] Master output saved: '{args.output_file}' ({os.path.getsize(args.output_file)/(1024*1024):.2f} MB)")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed_all = time.time() - t0_all
    print("=" * 80)
    print(f"Total Processing Time: {elapsed_all:.1f} s")
    if mc_files: print(f"MC Events Scanned    : {mc_ev:,} ({mc_sig:,} signal, {mc_bkg:,} bkg)")
    if data_files: print(f"Data Events Scanned  : {data_ev:,}")
    if pmu_files: print(f"Passing Muons Scanned: {pmu_ev:,}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
