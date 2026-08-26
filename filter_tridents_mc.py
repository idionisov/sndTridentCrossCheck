#!/usr/bin/env python3
"""
filter_tridents_mc.py
---------------------
High-throughput standalone filter script for extracting and storing Trident events
(mu + Z -> mu + mu+ + mu- + Z) from Monte Carlo datasets into new ROOT files with
equivalent structure, rich physics observables, and fiducial area constraints.

Features:
- Pure signal event extraction or full dataset tagging with --store-all-events.
- Configurable Fiducial Area plane (Z, X_min, X_max, Y_min, Y_max) for 3-muon containment.
- High performance ROOT RDataFrame JIT-compiled C++ filtering.
- Fully configurable Z regions (Rock, Target, Muon System) via command line.
- Comprehensive physics & kinematics calculation (Invariant mass, opening angle, momenta).
- Compact event summary storage using categorical enums and bitmasks.
- Automatically un-biases Geant4 cross-section biasing weights (mc_weight).
- Preserves identical 'cbmsim' TTree structure with all original branches.
- Copies detector geometry ('ShipGeo') and all metadata if present.
- Supports single files or wildcard patterns with per-file output naming.
"""

import os
import sys
import glob
import time
import argparse
import ROOT

def init_root_jit(z_min=-1e9, z_max=1e9, target_z_min=280.0, target_z_max=360.0,
                  num_threads=6, process_mask=7, boost_factor=100.0,
                  use_fiducial=False, fid_z=320.0, fid_xmin=-42.0, fid_xmax=-11.0,
                  fid_ymin=18.0, fid_ymax=49.0):
    """
    Initializes ROOT multi-threading and declares JIT C++ filter functions.
    """
    if num_threads > 0:
        ROOT.EnableImplicitMT(num_threads)

    sndsw_path = os.environ.get("SNDSW_ROOT", "")
    if sndsw_path:
        ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndSciFiTools.h"')
        ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndTchainGetter.h"')
        ROOT.gInterpreter.ProcessLine(f'#include "{sndsw_path}/analysis/tools/sndGeometryGetter.h"')

    use_fid_str = "true" if use_fiducial else "false"

    cxx_code = f"""
#ifndef FILTER_TRIDENTS_MC_H
#define FILTER_TRIDENTS_MC_H

#include "sndScifiHit.h"
#include "ShipMCTrack.h"
#include <TClonesArray.h>
#include <atomic>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <vector>
#include <cmath>
#include <algorithm>

namespace TridentFilter {{
    std::atomic<long long> rdf_events{{0}};
    std::atomic<long long> total_rdf_events{{0}};
    std::chrono::steady_clock::time_point start_time;
    std::mutex cout_mutex;
    
    enum ProcessType : int {{
        kUnknown = -1,
        kMuonToMuonPair = 0,    // Genuine trident: mu + N -> mu + mu+ mu- + N
        kGammaToMuPair = 1,     // Muon brem gamma-conversion: gamma + N -> N + mu+ mu-
        kAnnihiToMuPair = 2     // Positron annihilation: e+ e- -> mu+ mu-
    }};

    enum RegionType : int {{
        kRegionUnknown = 0,
        kRegionRock = 1,        // Z < target_z_min
        kRegionTarget = 2,      // target_z_min <= Z < target_z_max
        kRegionMuonSystem = 3   // Z >= target_z_max
    }};

    enum FlagBits : unsigned int {{
        kIsGenuine        = 1 << 0,  // 0x01: Genuine trident
        kIsGammaConv      = 1 << 1,  // 0x02: Photon conversion
        kIsPositronAnn    = 1 << 2,  // 0x04: Positron annihilation
        kIsPrimaryMuon    = 1 << 3,  // 0x08: Initiated by primary beam muon (MotherId == -1)
        kIsSecondaryMuon  = 1 << 4,  // 0x10: Initiated by secondary shower muon
        kIsDirectBrem     = 1 << 5,  // 0x20: Converting photon directly emitted by muon (cascade depth == 1)
        kInRock           = 1 << 6,  // 0x40: Vertex in rock (Z < target_z_min)
        kInTarget         = 1 << 7,  // 0x80: Vertex in target (target_z_min <= Z < target_z_max)
        kInMuonSystem     = 1 << 8,  // 0x100: Vertex in muon system (Z >= target_z_max)
        kInFiducial       = 1 << 9   // 0x200: All 3 trident muons crossed the fiducial area
    }};

    double g_z_min = {z_min};
    double g_z_max = {z_max};
    double g_target_z_min = {target_z_min};
    double g_target_z_max = {target_z_max};
    int g_process_mask = {process_mask}; // bitmask: 1=genuine (1<<0), 2=gamma (1<<1), 4=annihil (1<<2)
    double g_boost_factor = {boost_factor}; // Geant4 cross-section biasing factor for muToMuonPairProd
    
    bool g_use_fiducial = {use_fid_str};
    double g_fid_z = {fid_z};
    double g_fid_xmin = {fid_xmin};
    double g_fid_xmax = {fid_xmax};
    double g_fid_ymin = {fid_ymin};
    double g_fid_ymax = {fid_ymax};

    void configure(double z_min, double z_max, double target_z_min, double target_z_max,
                   int process_mask, double boost_factor,
                   bool use_fiducial, double fid_z, double fid_xmin, double fid_xmax, double fid_ymin, double fid_ymax) {{
        g_z_min = z_min;
        g_z_max = z_max;
        g_target_z_min = target_z_min;
        g_target_z_max = target_z_max;
        g_process_mask = process_mask;
        g_boost_factor = boost_factor;
        g_use_fiducial = use_fiducial;
        g_fid_z = fid_z;
        g_fid_xmin = fid_xmin;
        g_fid_xmax = fid_xmax;
        g_fid_ymin = fid_ymin;
        g_fid_ymax = fid_ymax;
    }}

    void reset_progress(long long total = 0) {{
        rdf_events = 0;
        total_rdf_events = total;
        start_time = std::chrono::steady_clock::now();
    }}

    bool print_progress() {{
        long long c = ++rdf_events;
        long long total = total_rdf_events.load();

        if (c % 100000 == 0 || (total > 0 && c == total)) {{
            auto now = std::chrono::steady_clock::now();
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

            long long hours = elapsed_sec / 3600;
            long long minutes = (elapsed_sec % 3600) / 60;
            long long seconds = elapsed_sec % 60;

            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "  Processed " << c;
            if (total > 0) {{
                double pct = (100.0 * c) / total;
                std::cout << " / " << total << " (" << std::fixed << std::setprecision(1) << pct << "%)";
            }}
            std::cout << " | Elapsed: "
                      << std::setfill('0') << std::setw(2) << hours << ":"
                      << std::setfill('0') << std::setw(2) << minutes << ":"
                      << std::setfill('0') << std::setw(2) << seconds
                      << std::endl;
        }}
        return true;
    }}

    struct TridentInfo {{
        bool is_signal;
        bool is_fiducial;           // True if all 3 muons cross the fiducial plane within (xmin, xmax, ymin, ymax)
        int proc_type;              // 0: Genuine, 1: GammaToMuPair, 2: AnnihiToMuPair
        int region_type;            // 1: Rock, 2: Target, 3: MuonSystem
        unsigned int trident_flags; // Packed bitmask of FlagBits
        double mc_weight;           // Un-biased primary MC event weight
        
        // Interaction vertex (cm)
        double vtx_x;
        double vtx_y;
        double vtx_z;
        
        // Track indices in MCTrack
        int rad_muon_id;            // Radiating muon track index (-1 if primary beam at generator)
        int mu_minus_id;            // Pair-produced mu- track index
        int mu_plus_id;             // Pair-produced mu+ track index
        
        // Radiating/Incoming Muon kinematics
        double p_mu_in;
        double px_mu_in;
        double py_mu_in;
        double pz_mu_in;
        double pt_mu_in;
        
        // Pair-produced mu- kinematics
        double p_mu_minus;
        double px_mu_minus;
        double py_mu_minus;
        double pz_mu_minus;
        double pt_mu_minus;
        double eta_mu_minus;
        double phi_mu_minus;
        
        // Pair-produced mu+ kinematics
        double p_mu_plus;
        double px_mu_plus;
        double py_mu_plus;
        double pz_mu_plus;
        double pt_mu_plus;
        double eta_mu_plus;
        double phi_mu_plus;
        
        // Pair composite kinematics
        double inv_mass_2mu;        // Invariant mass of mu+ mu- pair (GeV/c^2)
        double opening_angle_mrad;  // Opening angle between mu+ and mu- (mrad)
        double pt_2mu;              // Transverse momentum of mu+ mu- pair (GeV)
        double energy_asym;         // (E(mu+) - E(mu-)) / (E(mu+) + E(mu-))
        
        // Ancestry & shower info
        int cascade_depth;          // Shower steps between radiating muon and pair production
        int emission_proc;          // Geant4 emission process ID (5=pair, 8=brem, 9=delta, 10=annihilation)

        // Extrapolated positions at fiducial plane Z = g_fid_z (cm)
        double fid_x_minus;
        double fid_y_minus;
        double fid_x_plus;
        double fid_y_plus;
        double fid_x_rad;
        double fid_y_rad;
    }};

    inline bool check_muon_fiducial(const ShipMCTrack* tr, double z_plane, double xmin, double xmax, double ymin, double ymax, double& x_out, double& y_out) {{
        if (!tr) {{
            x_out = -9999.0;
            y_out = -9999.0;
            return false;
        }}
        double z0 = tr->GetStartZ();
        double pz = tr->GetPz();
        if (pz <= 0.0) {{
            x_out = tr->GetStartX();
            y_out = tr->GetStartY();
            return false;
        }}
        double dz = z_plane - z0;
        x_out = tr->GetStartX() + (tr->GetPx() / pz) * dz;
        y_out = tr->GetStartY() + (tr->GetPy() / pz) * dz;

        return (x_out >= xmin && x_out <= xmax && y_out >= ymin && y_out <= ymax);
    }}

    TridentInfo compute_trident_info(const TClonesArray& mc_tracks) {{
        int n_tracks = mc_tracks.GetEntriesFast();
        const double m_mu = 0.1056583755; // Muon mass in GeV
        
        for (int i = 0; i < n_tracks; ++i) {{
            auto* tr1 = static_cast<ShipMCTrack*>(mc_tracks.At(i));
            if (!tr1) continue;
            
            // Look for mu- or mu+ created via pair production (kPPair=5) or positron annihilation (kPAnnihilation=10, 11, 12)
            int p1 = tr1->GetProcID();
            if (p1 != 5 && p1 != 10 && p1 != 11 && p1 != 12) continue;
            int pdg1 = tr1->GetPdgCode();
            if (std::abs(pdg1) != 13) continue;
            
            // Find its partner
            for (int j = i + 1; j < n_tracks; ++j) {{
                auto* tr2 = static_cast<ShipMCTrack*>(mc_tracks.At(j));
                if (!tr2) continue;
                
                if (tr2->GetProcID() == p1 && tr2->GetMotherId() == tr1->GetMotherId()) {{
                    int pdg2 = tr2->GetPdgCode();
                    
                    // Check if they are opposite sign muons
                    if ((pdg1 == 13 && pdg2 == -13) || (pdg1 == -13 && pdg2 == 13)) {{
                        double z_v = tr1->GetStartZ();
                        
                        // Apply Z region filter cut
                        if (z_v >= g_z_min && z_v < g_z_max) {{
                            int m_id = tr1->GetMotherId();
                            int proc_type = kUnknown;
                            int rad_mu_id = -1;
                            double rad_p = 0.0, rad_px = 0.0, rad_py = 0.0, rad_pz = 0.0;
                            int em_proc = p1;
                            bool is_sec = true;
                            bool is_prim_init = false;
                            bool is_dir_brem = false;
                            int depth = 0;
                            
                            if (m_id == -1) {{
                                // Primary muon directly produced the pair
                                proc_type = kMuonToMuonPair;
                                rad_mu_id = -1;
                                rad_p = tr1->GetP();
                                rad_px = tr1->GetPx();
                                rad_py = tr1->GetPy();
                                rad_pz = tr1->GetPz();
                                em_proc = 5; // kPPair
                                is_sec = false;
                                is_prim_init = true;
                                is_dir_brem = true;
                                depth = 0;
                            }} else if (m_id >= 0 && m_id < n_tracks) {{
                                auto* mother = static_cast<ShipMCTrack*>(mc_tracks.At(m_id));
                                if (mother) {{
                                    int m_pdg = mother->GetPdgCode();
                                    
                                    if (std::abs(m_pdg) == 13) {{
                                        // Genuine Trident: Immediate mother is the muon
                                        proc_type = kMuonToMuonPair;
                                        rad_mu_id = m_id;
                                        rad_p = mother->GetP();
                                        rad_px = mother->GetPx();
                                        rad_py = mother->GetPy();
                                        rad_pz = mother->GetPz();
                                        em_proc = 5; // kPPair
                                        is_sec = (mother->GetMotherId() != -1);
                                        is_prim_init = (mother->GetMotherId() == -1);
                                        is_dir_brem = true;
                                        depth = 0;
                                    }} else if (m_pdg == 22 || std::abs(m_pdg) == 11) {{
                                        // Photon conversion (22) or Positron annihilation (11)
                                        proc_type = (m_pdg == 22) ? kGammaToMuPair : kAnnihiToMuPair;
                                        
                                        // Walk up ancestry backwards to find the radiating muon
                                        int curr = m_id;
                                        int child_of_muon = -1;
                                        
                                        while (curr >= 0 && curr < n_tracks) {{
                                            auto* anc = static_cast<ShipMCTrack*>(mc_tracks.At(curr));
                                            if (!anc) break;
                                            
                                            depth++;
                                            if (std::abs(anc->GetPdgCode()) == 13) {{
                                                rad_mu_id = curr;
                                                rad_p = anc->GetP();
                                                rad_px = anc->GetPx();
                                                rad_py = anc->GetPy();
                                                rad_pz = anc->GetPz();
                                                is_sec = (anc->GetMotherId() != -1);
                                                is_prim_init = (anc->GetMotherId() == -1);
                                                depth--; // subtract muon step itself
                                                break;
                                            }}
                                            child_of_muon = curr;
                                            curr = anc->GetMotherId();
                                        }}
                                        
                                        if (child_of_muon >= 0 && child_of_muon < n_tracks) {{
                                            auto* ch = static_cast<ShipMCTrack*>(mc_tracks.At(child_of_muon));
                                            if (ch) em_proc = ch->GetProcID();
                                        }}
                                        
                                        // Direct bremsstrahlung check (depth == 1)
                                        is_dir_brem = (m_pdg == 22 && mother->GetMotherId() == rad_mu_id && depth == 1);
                                    }}
                                }}
                            }}
                            
                            // Check process filter mask
                            if (proc_type >= 0 && ((1 << proc_type) & g_process_mask) == 0) {{
                                continue;
                            }}
                            
                            // Determine spatial region
                            int region_type = kRegionUnknown;
                            if (z_v < g_target_z_min) {{
                                region_type = kRegionRock;
                            }} else if (z_v < g_target_z_max) {{
                                region_type = kRegionTarget;
                            }} else {{
                                region_type = kRegionMuonSystem;
                            }}
                            
                            // Identify mu- and mu+
                            auto* tr_minus = (pdg1 == 13) ? tr1 : tr2;
                            auto* tr_plus  = (pdg1 == -13) ? tr1 : tr2;
                            int minus_id   = (pdg1 == 13) ? i : j;
                            int plus_id    = (pdg1 == -13) ? i : j;

                            // Radiating/Primary Muon track pointer
                            const ShipMCTrack* rad_tr = nullptr;
                            if (rad_mu_id >= 0 && rad_mu_id < n_tracks) {{
                                rad_tr = static_cast<ShipMCTrack*>(mc_tracks.At(rad_mu_id));
                            }} else if (n_tracks > 0) {{
                                rad_tr = static_cast<ShipMCTrack*>(mc_tracks.At(0));
                            }}

                            // Check fiducial area containment at Z = g_fid_z for all 3 muons
                            double xm = -9999.0, ym = -9999.0, xp = -9999.0, yp = -9999.0, xrad = -9999.0, yrad = -9999.0;
                            bool fid_m = check_muon_fiducial(tr_minus, g_fid_z, g_fid_xmin, g_fid_xmax, g_fid_ymin, g_fid_ymax, xm, ym);
                            bool fid_p = check_muon_fiducial(tr_plus, g_fid_z, g_fid_xmin, g_fid_xmax, g_fid_ymin, g_fid_ymax, xp, yp);
                            bool fid_rad = check_muon_fiducial(rad_tr, g_fid_z, g_fid_xmin, g_fid_xmax, g_fid_ymin, g_fid_ymax, xrad, yrad);

                            bool is_fid = (fid_m && fid_p && fid_rad);

                            // Pack boolean bitmask flags
                            unsigned int flags = 0;
                            if (proc_type == kMuonToMuonPair) flags |= kIsGenuine;
                            else if (proc_type == kGammaToMuPair) flags |= kIsGammaConv;
                            else if (proc_type == kAnnihiToMuPair) flags |= kIsPositronAnn;
                            
                            if (is_prim_init) flags |= kIsPrimaryMuon;
                            else flags |= kIsSecondaryMuon;
                            
                            if (is_dir_brem) flags |= kIsDirectBrem;
                            
                            if (region_type == kRegionRock) flags |= kInRock;
                            else if (region_type == kRegionTarget) flags |= kInTarget;
                            else if (region_type == kRegionMuonSystem) flags |= kInMuonSystem;

                            if (is_fid) flags |= kInFiducial;
                            
                            double px_m = tr_minus->GetPx(), py_m = tr_minus->GetPy(), pz_m = tr_minus->GetPz(), p_m = tr_minus->GetP();
                            double px_p = tr_plus->GetPx(), py_p = tr_plus->GetPy(), pz_p = tr_plus->GetPz(), p_p = tr_plus->GetP();
                            
                            double e_m = std::sqrt(p_m * p_m + m_mu * m_mu);
                            double e_p = std::sqrt(p_p * p_p + m_mu * m_mu);
                            
                            double pt_m = std::sqrt(px_m * px_m + py_m * py_m);
                            double pt_p = std::sqrt(px_p * px_p + py_p * py_p);
                            
                            double eta_m = (p_m - pz_m > 1e-9 && p_m + pz_m > 1e-9) ? 0.5 * std::log((p_m + pz_m) / (p_m - pz_m)) : 0.0;
                            double eta_p = (p_p - pz_p > 1e-9 && p_p + pz_p > 1e-9) ? 0.5 * std::log((p_p + pz_p) / (p_p - pz_p)) : 0.0;
                            
                            double phi_m = std::atan2(py_m, px_m);
                            double phi_p = std::atan2(py_p, px_p);
                            
                            // Pair composite kinematics
                            double e_tot = e_m + e_p;
                            double px_tot = px_m + px_p;
                            double py_tot = py_m + py_p;
                            double pz_tot = pz_m + pz_p;
                            double p_tot2 = px_tot * px_tot + py_tot * py_tot + pz_tot * pz_tot;
                            double m2 = e_tot * e_tot - p_tot2;
                            double inv_mass = (m2 > 0) ? std::sqrt(m2) : 0.0;
                            double pt_pair = std::sqrt(px_tot * px_tot + py_tot * py_tot);
                            double e_asym = (e_tot > 0) ? (e_p - e_m) / e_tot : 0.0;
                            
                            double cos_th = (p_m * p_p > 0) ? (px_m * px_p + py_m * py_p + pz_m * pz_p) / (p_m * p_p) : 1.0;
                            cos_th = std::clamp(cos_th, -1.0, 1.0);
                            double op_angle_mrad = std::acos(cos_th) * 1000.0;
                            
                            // Unbias weight:
                            double raw_w = tr1->GetWeight();
                            if (proc_type == kMuonToMuonPair && g_boost_factor > 0.0) {{
                                raw_w *= g_boost_factor;
                            }}
                            double pt_in = std::sqrt(rad_px * rad_px + rad_py * rad_py);
                            
                            return {{
                                true, is_fid, proc_type, region_type, flags, raw_w,
                                tr1->GetStartX(), tr1->GetStartY(), z_v,
                                rad_mu_id, minus_id, plus_id,
                                rad_p, rad_px, rad_py, rad_pz, pt_in,
                                p_m, px_m, py_m, pz_m, pt_m, eta_m, phi_m,
                                p_p, px_p, py_p, pz_p, pt_p, eta_p, phi_p,
                                inv_mass, op_angle_mrad, pt_pair, e_asym,
                                depth, em_proc,
                                xm, ym, xp, yp, xrad, yrad
                            }};
                        }}
                    }}
                }}
            }}
        }}

        TridentInfo empty{{}};
        empty.is_signal = false;
        empty.is_fiducial = false;
        empty.proc_type = kUnknown;
        empty.region_type = kRegionUnknown;
        empty.trident_flags = 0;
        empty.mc_weight = (n_tracks > 0) ? static_cast<ShipMCTrack*>(mc_tracks.At(0))->GetWeight() : 0.0;
        empty.vtx_x = 0.0;
        empty.vtx_y = 0.0;
        empty.vtx_z = -9999.0;
        empty.rad_muon_id = -1;
        empty.mu_minus_id = -1;
        empty.mu_plus_id = -1;
        empty.p_mu_in = 0.0;
        empty.px_mu_in = 0.0;
        empty.py_mu_in = 0.0;
        empty.pz_mu_in = 0.0;
        empty.pt_mu_in = 0.0;
        empty.p_mu_minus = 0.0;
        empty.px_mu_minus = 0.0;
        empty.py_mu_minus = 0.0;
        empty.pz_mu_minus = 0.0;
        empty.pt_mu_minus = 0.0;
        empty.eta_mu_minus = 0.0;
        empty.phi_mu_minus = 0.0;
        empty.p_mu_plus = 0.0;
        empty.px_mu_plus = 0.0;
        empty.py_mu_plus = 0.0;
        empty.pz_mu_plus = 0.0;
        empty.pt_mu_plus = 0.0;
        empty.eta_mu_plus = 0.0;
        empty.phi_mu_plus = 0.0;
        empty.inv_mass_2mu = 0.0;
        empty.opening_angle_mrad = 0.0;
        empty.pt_2mu = 0.0;
        empty.energy_asym = 0.0;
        empty.cascade_depth = -1;
        empty.emission_proc = -1;
        empty.fid_x_minus = -9999.0;
        empty.fid_y_minus = -9999.0;
        empty.fid_x_plus = -9999.0;
        empty.fid_y_plus = -9999.0;
        empty.fid_x_rad = -9999.0;
        empty.fid_y_rad = -9999.0;
        return empty;
    }}

    // Thread-local cached evaluation to maximize RDataFrame performance
    thread_local const TClonesArray* tls_last_tracks = nullptr;
    thread_local int tls_last_size = -1;
    thread_local TridentInfo tls_cached_info;

    inline TridentInfo check_trident(const TClonesArray& mc_tracks) {{
        if (&mc_tracks == tls_last_tracks && mc_tracks.GetEntriesFast() == tls_last_size) {{
            return tls_cached_info;
        }}
        tls_last_tracks = &mc_tracks;
        tls_last_size = mc_tracks.GetEntriesFast();
        tls_cached_info = compute_trident_info(mc_tracks);
        return tls_cached_info;
    }}

    bool is_trident_event(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).is_signal; }}
    bool is_fiducial_event(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).is_fiducial; }}
    bool is_selected_event(const TClonesArray& mc_tracks) {{
        TridentInfo info = check_trident(mc_tracks);
        if (!info.is_signal) return false;
        if (g_use_fiducial && !info.is_fiducial) return false;
        return true;
    }}

    double get_mc_weight(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).mc_weight; }}
    int get_proc_type(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).proc_type; }}
    int get_region_type(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).region_type; }}
    unsigned int get_trident_flags(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).trident_flags; }}
    double get_vtx_x(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).vtx_x; }}
    double get_vtx_y(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).vtx_y; }}
    double get_vtx_z(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).vtx_z; }}
    int get_rad_muon_id(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).rad_muon_id; }}
    int get_mu_minus_id(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).mu_minus_id; }}
    int get_mu_plus_id(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).mu_plus_id; }}
    double get_p_mu_in(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).p_mu_in; }}
    double get_px_mu_in(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).px_mu_in; }}
    double get_py_mu_in(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).py_mu_in; }}
    double get_pz_mu_in(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).pz_mu_in; }}
    double get_pt_mu_in(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).pt_mu_in; }}
    double get_p_mu_minus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).p_mu_minus; }}
    double get_px_mu_minus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).px_mu_minus; }}
    double get_py_mu_minus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).py_mu_minus; }}
    double get_pz_mu_minus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).pz_mu_minus; }}
    double get_pt_mu_minus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).pt_mu_minus; }}
    double get_eta_mu_minus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).eta_mu_minus; }}
    double get_phi_mu_minus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).phi_mu_minus; }}
    double get_p_mu_plus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).p_mu_plus; }}
    double get_px_mu_plus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).px_mu_plus; }}
    double get_py_mu_plus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).py_mu_plus; }}
    double get_pz_mu_plus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).pz_mu_plus; }}
    double get_pt_mu_plus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).pt_mu_plus; }}
    double get_eta_mu_plus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).eta_mu_plus; }}
    double get_phi_mu_plus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).phi_mu_plus; }}
    double get_inv_mass_2mu(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).inv_mass_2mu; }}
    double get_opening_angle_mrad(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).opening_angle_mrad; }}
    double get_pt_2mu(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).pt_2mu; }}
    double get_energy_asym(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).energy_asym; }}
    int get_cascade_depth(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).cascade_depth; }}
    int get_emission_proc(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).emission_proc; }}
    
    double get_fid_x_minus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).fid_x_minus; }}
    double get_fid_y_minus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).fid_y_minus; }}
    double get_fid_x_plus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).fid_x_plus; }}
    double get_fid_y_plus(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).fid_y_plus; }}
    double get_fid_x_rad(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).fid_x_rad; }}
    double get_fid_y_rad(const TClonesArray& mc_tracks) {{ return check_trident(mc_tracks).fid_y_rad; }}
}}
#endif
    """
    ROOT.gInterpreter.Declare(cxx_code)
    ROOT.TridentFilter.configure(z_min, z_max, target_z_min, target_z_max,
                                 process_mask, boost_factor,
                                 use_fiducial, fid_z, fid_xmin, fid_xmax, fid_ymin, fid_ymax)

def extract_file_tag(filename):
    """Extracts tag such as digCPP-200 from filename."""
    base = os.path.basename(filename)
    if "digCPP-" in base:
        parts = base.split("digCPP-")
        if len(parts) > 1:
            tag = "digCPP-" + parts[1].replace(".root", "")
            return tag
    name_no_ext = os.path.splitext(base)[0]
    return name_no_ext

def process_single_file(input_file, output_file, store_all_events=False):
    """
    Processes a single ROOT file using RDataFrame, calculating kinematics and
    summary branches, filtering (or storing all), and saving 'cbmsim' + 'ShipGeo'.
    """
    t0 = time.time()

    f_test = ROOT.TFile.Open(input_file, "READ")
    if not f_test or f_test.IsZombie():
        print(f"Error: Could not open '{input_file}'")
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    tree = f_test.Get("cbmsim")
    if not tree:
        print(f"Error: 'cbmsim' tree not found in '{input_file}'")
        f_test.Close()
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    total_events = tree.GetEntries()
    f_test.Close()

    ROOT.TridentFilter.reset_progress(total_events)

    df = ROOT.RDataFrame("cbmsim", input_file)
    df_with_progress = df.Filter("TridentFilter::print_progress()")

    if store_all_events:
        df_base = df_with_progress
    else:
        df_base = df_with_progress.Filter("TridentFilter::is_selected_event(MCTrack)")

    df_filtered = (
        df_base
          .Define("is_signal", "TridentFilter::is_trident_event(MCTrack)")
          .Define("is_fiducial", "TridentFilter::is_fiducial_event(MCTrack)")
          .Define("mc_weight", "TridentFilter::get_mc_weight(MCTrack)")
          .Define("proc_type", "TridentFilter::get_proc_type(MCTrack)")
          .Define("region_type", "TridentFilter::get_region_type(MCTrack)")
          .Define("trident_flags", "TridentFilter::get_trident_flags(MCTrack)")
          .Define("vtx_x", "TridentFilter::get_vtx_x(MCTrack)")
          .Define("vtx_y", "TridentFilter::get_vtx_y(MCTrack)")
          .Define("vtx_z", "TridentFilter::get_vtx_z(MCTrack)")
          .Define("rad_muon_id", "TridentFilter::get_rad_muon_id(MCTrack)")
          .Define("mu_minus_id", "TridentFilter::get_mu_minus_id(MCTrack)")
          .Define("mu_plus_id", "TridentFilter::get_mu_plus_id(MCTrack)")
          .Define("p_mu_in", "TridentFilter::get_p_mu_in(MCTrack)")
          .Define("px_mu_in", "TridentFilter::get_px_mu_in(MCTrack)")
          .Define("py_mu_in", "TridentFilter::get_py_mu_in(MCTrack)")
          .Define("pz_mu_in", "TridentFilter::get_pz_mu_in(MCTrack)")
          .Define("pt_mu_in", "TridentFilter::get_pt_mu_in(MCTrack)")
          .Define("p_mu_minus", "TridentFilter::get_p_mu_minus(MCTrack)")
          .Define("px_mu_minus", "TridentFilter::get_px_mu_minus(MCTrack)")
          .Define("py_mu_minus", "TridentFilter::get_py_mu_minus(MCTrack)")
          .Define("pz_mu_minus", "TridentFilter::get_pz_mu_minus(MCTrack)")
          .Define("pt_mu_minus", "TridentFilter::get_pt_mu_minus(MCTrack)")
          .Define("eta_mu_minus", "TridentFilter::get_eta_mu_minus(MCTrack)")
          .Define("phi_mu_minus", "TridentFilter::get_phi_mu_minus(MCTrack)")
          .Define("p_mu_plus", "TridentFilter::get_p_mu_plus(MCTrack)")
          .Define("px_mu_plus", "TridentFilter::get_px_mu_plus(MCTrack)")
          .Define("py_mu_plus", "TridentFilter::get_py_mu_plus(MCTrack)")
          .Define("pz_mu_plus", "TridentFilter::get_pz_mu_plus(MCTrack)")
          .Define("pt_mu_plus", "TridentFilter::get_pt_mu_plus(MCTrack)")
          .Define("eta_mu_plus", "TridentFilter::get_eta_mu_plus(MCTrack)")
          .Define("phi_mu_plus", "TridentFilter::get_phi_mu_plus(MCTrack)")
          .Define("inv_mass_2mu", "TridentFilter::get_inv_mass_2mu(MCTrack)")
          .Define("opening_angle_mrad", "TridentFilter::get_opening_angle_mrad(MCTrack)")
          .Define("pt_2mu", "TridentFilter::get_pt_2mu(MCTrack)")
          .Define("energy_asym", "TridentFilter::get_energy_asym(MCTrack)")
          .Define("cascade_depth", "TridentFilter::get_cascade_depth(MCTrack)")
          .Define("emission_proc", "TridentFilter::get_emission_proc(MCTrack)")
          .Define("fid_x_minus", "TridentFilter::get_fid_x_minus(MCTrack)")
          .Define("fid_y_minus", "TridentFilter::get_fid_y_minus(MCTrack)")
          .Define("fid_x_plus", "TridentFilter::get_fid_x_plus(MCTrack)")
          .Define("fid_y_plus", "TridentFilter::get_fid_y_plus(MCTrack)")
          .Define("fid_x_rad", "TridentFilter::get_fid_x_rad(MCTrack)")
          .Define("fid_y_rad", "TridentFilter::get_fid_y_rad(MCTrack)")
    )

    c_total = df_filtered.Count()

    # Process breakdown (only for signal events)
    df_sig = df_filtered.Filter("is_signal == true")
    c_sig = df_sig.Count()
    c_fid = df_sig.Filter("is_fiducial == true").Count()
    c_genuine = df_sig.Filter("proc_type == 0").Count()
    c_gamma = df_sig.Filter("proc_type == 1").Count()
    c_annihil = df_sig.Filter("proc_type == 2").Count()
    c_sec = df_sig.Filter("(trident_flags & 0x10) != 0").Count()

    # Spatial region breakdown (only for signal events)
    c_rock = df_sig.Filter("region_type == 1").Count()
    c_target = df_sig.Filter("region_type == 2").Count()
    c_muon = df_sig.Filter("region_type == 3").Count()

    opts = ROOT.RDF.RSnapshotOptions()
    opts.fMode = "RECREATE"
    df_filtered.Snapshot("cbmsim", output_file, "", opts)

    stored_count = c_total.GetValue()
    sig_count = c_sig.GetValue()
    fid_count = c_fid.GetValue()
    gen_count = c_genuine.GetValue()
    gam_count = c_gamma.GetValue()
    ann_count = c_annihil.GetValue()
    sec_count = c_sec.GetValue()
    r_count = c_rock.GetValue()
    t_count = c_target.GetValue()
    m_count = c_muon.GetValue()

    # Re-open input and output files to copy all metadata objects (e.g. ShipGeo)
    f_in = ROOT.TFile.Open(input_file, "READ")
    metadata_keys = []
    if f_in and not f_in.IsZombie():
        for k in f_in.GetListOfKeys():
            kname = k.GetName()
            kclass = k.GetClassName()
            if kclass != "TTree" and kname != "cbmsim":
                obj = f_in.Get(kname)
                if obj:
                    metadata_keys.append((kname, obj))

    if metadata_keys:
        f_out = ROOT.TFile.Open(output_file, "UPDATE")
        if f_out and not f_out.IsZombie():
            f_out.cd()
            for kname, obj in metadata_keys:
                obj.Write(kname, ROOT.TObject.kSingleKey | ROOT.TObject.kOverwrite)
            f_out.Close()
    if f_in:
        f_in.Close()

    elapsed = time.time() - t0
    rate = total_events / elapsed if elapsed > 0 else 0
    print(f"  Processed {total_events:,} events in {elapsed:.1f}s ({rate:.0f} ev/s)")
    print(f"  Output Tree Entries: {stored_count:,} (Signal: {sig_count:,} | Fiducial: {fid_count:,})")
    print(f"    - Processes : Genuine: {gen_count} | Gamma: {gam_count} | Annihil: {ann_count} | Sec: {sec_count}")
    print(f"    - Regions   : Rock: {r_count} | Target: {t_count} | MuonSystem: {m_count}")

    return (total_events, stored_count, sig_count, fid_count,
            gen_count, gam_count, ann_count, sec_count,
            r_count, t_count, m_count)

def main():
    default_input = "/eos/experiment/sndlhc/MonteCarlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-2*.root"
    default_output = "trimuon_filtered_%s.root"

    parser = argparse.ArgumentParser(
        description="Extract and save trident signal events into ROOT files with comprehensive physics observables and fiducial area constraints."
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_pattern",
        nargs="?",
        default=default_input,
        help="Input ROOT file path or wildcard pattern (default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_pattern",
        default=default_output,
        help="Output ROOT file path or pattern with %%s for tag (default: %(default)s)"
    )
    parser.add_argument(
        "--store-all-events", "--keep-all",
        dest="store_all_events",
        action="store_true",
        default=False,
        help="If set, stores all events in the input files (both signal and non-signal) with the extra observables attached without filtering out non-signal events (default: False)"
    )
    parser.add_argument(
        "--fiducial", "--fiducial-area",
        dest="fiducial_mode",
        choices=["unconstrained", "default", "custom"],
        default="unconstrained",
        help="Fiducial area constraint for 3-muon containment: 'unconstrained' (default, no fiducial constraint), 'default' (z=320, x in [-42, -11], y in [18, 49]), or 'custom' (uses --fiducial-box) (default: %(default)s)"
    )
    parser.add_argument(
        "--fiducial-box",
        dest="fiducial_box",
        nargs=5,
        type=float,
        default=[320.0, -42.0, -11.0, 18.0, 49.0],
        metavar=("Z", "X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Custom fiducial plane and rectangular boundaries [Z, X_MIN, X_MAX, Y_MIN, Y_MAX] in cm (default: 320.0 -42.0 -11.0 18.0 49.0)"
    )
    parser.add_argument(
        "--z-range",
        dest="target_z_range",
        nargs=2,
        type=float,
        default=[260.0, 355.0],
        metavar=("Z_MIN", "Z_MAX"),
        help="Target region Z boundaries [Z_MIN, Z_MAX] in cm (default: 260.0 355.0)"
    )
    parser.add_argument(
        "--region",
        dest="region",
        choices=["all", "rock", "target", "muon_system", "custom"],
        default="all",
        help="Volume region filter: 'all' (all Z), 'rock' (Z < target_z_min), 'target' (target_z_min <= Z < target_z_max), 'muon_system' (Z >= target_z_max), or 'custom' (default: %(default)s)"
    )
    parser.add_argument(
        "--z-min",
        dest="z_min",
        type=float,
        default=None,
        help="Minimum Z vertex coordinate in cm (overrides region preset if set)"
    )
    parser.add_argument(
        "-z", "--z-max", "--rock_boundary",
        dest="z_max",
        type=float,
        default=None,
        help="Maximum Z vertex coordinate in cm (overrides region preset if set)"
    )
    parser.add_argument(
        "--processes",
        dest="processes",
        choices=["all", "genuine", "gamma", "annihil"],
        default="all",
        help="Filter process: 'all', 'genuine' (MuonToMuonPair), 'gamma' (GammaToMuPair), 'annihil' (Positron Annihilation) (default: %(default)s)"
    )
    parser.add_argument(
        "-b", "--boost", "--boost-factor",
        dest="boost_factor",
        type=float,
        default=100.0,
        help="Geant4 cross section boost factor applied during MC generation (default: 100.0, use 1000.0 for boost1000, 1.0 for unboosted)"
    )
    parser.add_argument(
        "-j", "--threads",
        dest="num_threads",
        type=int,
        default=6,
        help="Number of worker threads for RDataFrame (default: %(default)s)"
    )

    args = parser.parse_args()

    # Unpack target Z boundaries
    target_z_min, target_z_max = args.target_z_range

    # Determine Z boundaries for filtering based on target_z_min, target_z_max and region
    if args.region == "rock":
        z_min = -1e9
        z_max = target_z_min
    elif args.region == "target":
        z_min = target_z_min
        z_max = target_z_max
    elif args.region == "muon_system":
        z_min = target_z_max
        z_max = 1e9
    elif args.region == "all":
        z_min = -1e9
        z_max = 1e9
    else:  # custom
        z_min = -1e9
        z_max = target_z_min

    if args.z_min is not None:
        z_min = args.z_min
    if args.z_max is not None:
        z_max = args.z_max

    # Determine process bitmask: bit 0: genuine, bit 1: gamma, bit 2: annihil
    if args.processes == "genuine":
        process_mask = 1 << 0  # 1
    elif args.processes == "gamma":
        process_mask = 1 << 1  # 2
    elif args.processes == "annihil":
        process_mask = 1 << 2  # 4
    else:
        process_mask = (1 << 0) | (1 << 1) | (1 << 2)  # 7

    # Determine Fiducial settings
    use_fiducial = (args.fiducial_mode != "unconstrained")
    if args.fiducial_mode == "custom" or args.fiducial_box != [320.0, -42.0, -11.0, 18.0, 49.0]:
        fid_z, fid_xmin, fid_xmax, fid_ymin, fid_ymax = args.fiducial_box
        use_fiducial = True
    else:
        fid_z, fid_xmin, fid_xmax, fid_ymin, fid_ymax = 320.0, -42.0, -11.0, 18.0, 49.0

    # Find matching files
    matched_files = sorted(glob.glob(args.input_pattern))
    if not matched_files:
        if os.path.exists(args.input_pattern):
            matched_files = [args.input_pattern]
        else:
            print(f"Error: No files found matching pattern: {args.input_pattern}")
            sys.exit(1)

    print("=" * 65)
    print("SND@LHC MC Trident Signal Event Extractor")
    print("=" * 65)
    print(f"Input Pattern     : {args.input_pattern}")
    print(f"Files Found       : {len(matched_files)}")
    print(f"Store All Events  : {args.store_all_events}")
    print(f"Target Z Region   : [{target_z_min:.1f}, {target_z_max:.1f}] cm")
    print(f"Region Filter     : {args.region} ({z_min:.1f} cm <= Z < {z_max:.1f} cm)")
    print(f"Process Filter    : {args.processes}")
    print(f"Boost Factor      : {args.boost_factor:.1f}x")
    print(f"Fiducial Area     : {'ENABLED' if use_fiducial else 'UNCONSTRAINED'}")
    if use_fiducial:
        print(f"  * Fiducial Plane: Z = {fid_z:.1f} cm")
        print(f"  * X Boundary    : [{fid_xmin:.1f}, {fid_xmax:.1f}] cm")
        print(f"  * Y Boundary    : [{fid_ymin:.1f}, {fid_ymax:.1f}] cm")
    print(f"Worker Threads    : {args.num_threads}")
    print("=" * 65)

    init_root_jit(
        z_min=z_min,
        z_max=z_max,
        target_z_min=target_z_min,
        target_z_max=target_z_max,
        num_threads=args.num_threads, 
        process_mask=process_mask,
        boost_factor=args.boost_factor,
        use_fiducial=use_fiducial,
        fid_z=fid_z,
        fid_xmin=fid_xmin,
        fid_xmax=fid_xmax,
        fid_ymin=fid_ymin,
        fid_ymax=fid_ymax
    )

    overall_t0 = time.time()
    grand_total_events = 0
    grand_total_stored = 0
    grand_total_signal = 0
    grand_total_fiducial = 0

    grand_total_genuine = 0
    grand_total_gamma = 0
    grand_total_annihil = 0
    grand_total_sec = 0

    grand_total_rock = 0
    grand_total_target = 0
    grand_total_muon = 0

    created_files = []

    for idx, input_file in enumerate(matched_files, 1):
        tag = extract_file_tag(input_file)
        if "%s" in args.output_pattern:
            out_file = args.output_pattern % tag
        elif len(matched_files) > 1:
            base_name, ext = os.path.splitext(args.output_pattern)
            out_file = f"{base_name}_{tag}{ext}"
        else:
            out_file = args.output_pattern

        print(f"\n[{idx}/{len(matched_files)}] Filtering: '{input_file}' -> '{out_file}'")
        (t_evts, stored_evts, sig_evts, fid_evts,
         gen_evts, gam_evts, ann_evts, sec_evts,
         r_evts, tgt_evts, mu_evts) = process_single_file(
             input_file, out_file, store_all_events=args.store_all_events
         )

        grand_total_events += t_evts
        grand_total_stored += stored_evts
        grand_total_signal += sig_evts
        grand_total_fiducial += fid_evts

        grand_total_genuine += gen_evts
        grand_total_gamma += gam_evts
        grand_total_annihil += ann_evts
        grand_total_sec += sec_evts

        grand_total_rock += r_evts
        grand_total_target += tgt_evts
        grand_total_muon += mu_evts

        created_files.append(out_file)

    overall_elapsed = time.time() - overall_t0

    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE SUMMARY")
    print("=" * 65)
    print(f"Total Input Files Processed : {len(matched_files)}")
    print(f"Total Output Files Created   : {len(created_files)}")
    print(f"Total Events Scanned        : {grand_total_events:,}")
    print(f"Total Events Stored         : {grand_total_stored:,}")
    print(f"Total Signal Events Found   : {grand_total_signal:,}")
    print(f"Total Fiducial 3-Muon Signal: {grand_total_fiducial:,} ({grand_total_fiducial/max(1,grand_total_signal)*100:.1f}% acceptance)")
    print(f"\n--- Physical Process Breakdown (Signal Events) ---")
    print(f"  * Genuine Tridents (MuonToMuonPair)  : {grand_total_genuine:,} events")
    print(f"  * Photon Conversions (GammaToMuPair) : {grand_total_gamma:,} events")
    print(f"  * Positron Annihilations             : {grand_total_annihil:,} events")
    print(f"  * Secondary Muon Induced             : {grand_total_sec:,} events")
    print(f"\n--- Spatial Region Breakdown (Signal Events) ---")
    print(f"  * Upstream Rock (< {target_z_min:.1f} cm)         : {grand_total_rock:,} events")
    print(f"  * Target Region [{target_z_min:.1f}, {target_z_max:.1f}] cm : {grand_total_target:,} events")
    print(f"  * Muon System (>= {target_z_max:.1f} cm)         : {grand_total_muon:,} events")
    print(f"\n============================================================")
    print(f"Total Processing Time        : {overall_elapsed:.1f} s")
    print("=" * 65)

if __name__ == "__main__":
    main()

