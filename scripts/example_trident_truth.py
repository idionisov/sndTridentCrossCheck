#!/usr/bin/env python3
"""
example_trident_truth.py
------------------------
Example demonstrating how to use the compiled C++ library `libtrident_analysis.so`
with ROOT's RDataFrame in Python.

Style notes:
  - Python variables follow snake_case (e.g. input_file, total_candidates).
  - C++ classes, methods, and properties follow camelCase (e.g. isRockTrident(), invMass2Mu).
"""

import os
import ROOT

# --- 1. Load the compiled C++ library ---
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_lib = os.path.join(repo_root, "build", "lib")
analysis_inc = os.path.join(repo_root, "analysis")

ROOT.gInterpreter.AddIncludePath(analysis_inc)
ROOT.gInterpreter.ProcessLine('#include "TridentTruthProcessor.h"')
ROOT.gSystem.AddDynamicPath(build_lib)
ROOT.gSystem.Load("libtrident_analysis.so")

# --- 2. Configure the C++ Processor using camelCase C++ setters ---
config = ROOT.snd.trident.TridentTruthConfig()
config.targetZMin = -100.0  # Upstream target boundary (cm)
config.targetZMax = 600.0   # Downstream target boundary (cm)
config.processMask = 7      # Accept all 3 processes: genuine (1), gamma (2), annihil (4)
config.useFiducial = False

processor = ROOT.snd.trident.TridentTruthProcessor(config)

# --- 3. Run with RDataFrame ---
input_file = "/eos/experiment/sndlhc/MonteCarlo/ThreeMuons/sndLHC.Ntuple-TGeant4_boost100LHC_-160urad_magfield_2022TCL6_muons_rock_2e8pr_filteredAtScoringPlane_digCPP-200.root"
print(f"Opening: {input_file}")

df = ROOT.RDataFrame("cbmsim", input_file).Range(2000)

# Evaluate C++ processor on MCTrack once per event.
# Downstream nodes query rich semantic predicates (e.g. isRockTrident(), inTarget())
df_annotated = (
    df.Define("truth", processor, ["MCTrack"])
      .Define("inv_mass", "truth.invMass2Mu")
      .Define("scaled_weight", "truth.scaledWeight")
      .Define("vtx_z", "truth.vtxZ")
)

# Different analysis definitions of 'signal':
# A. Genuine tridents in the rock:
df_rock_trident = df_annotated.Filter("truth.isRockTrident()")

# B. Any 3-muon process in the rock:
df_rock_trimuon = df_annotated.Filter("truth.isRockTriMuon()")

# C. Genuine tridents in the target:
df_target_trident = df_annotated.Filter("truth.isTargetTrident()")

# --- 4. Book Actions and Compute ---
total_rock_tridents = df_rock_trident.Count()
yield_rock_tridents = df_rock_trident.Sum("scaled_weight")

total_rock_trimuon = df_rock_trimuon.Count()
yield_rock_trimuon = df_rock_trimuon.Sum("scaled_weight")

total_target_tridents = df_target_trident.Count()
yield_target_tridents = df_target_trident.Sum("scaled_weight")

h_mass_rock = df_rock_trident.Histo1D(
    ("h_mass_rock", "Rock Trident Invariant Mass;M_{2#mu} [GeV/c^{2}];Yield", 40, 0.2, 10.0),
    "inv_mass", "scaled_weight"
)

print(f"\nResults for 2000 events:")
print(f"  Rock Genuine Tridents:  count = {total_rock_tridents.GetValue():3d}, yield = {yield_rock_tridents.GetValue():.4e}")
print(f"  Rock All Tri-Muons:     count = {total_rock_trimuon.GetValue():3d}, yield = {yield_rock_trimuon.GetValue():.4e}")
print(f"  Target Genuine Tridents: count = {total_target_tridents.GetValue():3d}, yield = {yield_target_tridents.GetValue():.4e}")
print(f"  Histogram integral:     {h_mass_rock.Integral():.4e}")
