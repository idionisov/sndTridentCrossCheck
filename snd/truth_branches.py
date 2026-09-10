"""
truth_branches.py
-----------------
Helper utilities to configure TridentTruthProcessor and unpack truth
observables into flat RDataFrame branches.
"""

from __future__ import annotations

from typing import Optional, Sequence, Any, List
import ROOT
from .data_manager import load_trident_libraries

TRIDENT_ANALYSIS_COLUMNS: List[str] = [
    "is_signal", "is_fiducial", "proc_type", "region_type", "trident_flags",
    "proc_name", "region_name", "emission_proc_name",
    "mc_weight", "raw_mc_weight",
    "vtx_x", "vtx_y", "vtx_z",
    "rad_muon_id", "mu_minus_id", "mu_plus_id",
    "p_mu_in", "px_mu_in", "py_mu_in", "pz_mu_in", "pt_mu_in",
    "p_mu_minus", "px_mu_minus", "py_mu_minus", "pz_mu_minus", "pt_mu_minus", "eta_mu_minus", "phi_mu_minus",
    "p_mu_plus", "px_mu_plus", "py_mu_plus", "pz_mu_plus", "pt_mu_plus", "eta_mu_plus", "phi_mu_plus",
    "inv_mass_2mu", "opening_angle_mrad", "pt_2mu", "energy_asym",
    "delta_phi_2mu", "delta_eta_2mu", "delta_r_2mu",
    "inv_mass_3mu", "pt_3mu",
    "opening_angle_rad_minus_mrad", "opening_angle_rad_plus_mrad",
    "energy_ratio_pair", "momentum_fraction_minus", "momentum_fraction_plus", "pt_asymmetry",
    "cascade_depth", "emission_proc",
    "fid_x_minus", "fid_y_minus", "fid_x_plus", "fid_y_plus", "fid_x_rad", "fid_y_rad",
    "dist_to_fiducial_edge"
]


def create_truth_config(
    z_min: float = -1e9,
    z_max: float = 1e9,
    target_z_min: float = 260.0,
    target_z_max: float = 355.0,
    process_mask: int = 7,
    weight_scale: float = 1.0,
    use_fiducial: bool = False,
    fid_box: Optional[Sequence[float]] = None,
) -> ROOT.snd.trident.TridentTruthConfig:
    """
    Creates and populates a C++ TridentTruthConfig instance.

    Args:
        z_min: Minimum vertex Z coordinate in cm.
        z_max: Maximum vertex Z coordinate in cm.
        target_z_min: Target region lower Z boundary in cm.
        target_z_max: Target region upper Z boundary in cm.
        process_mask: Process bitmask (1=genuine, 2=gamma, 4=annihil).
        weight_scale: Overall weight normalization scale.
        use_fiducial: If True, evaluates fiducial containment flag.
        fid_box: Optional sequence of [z, x_min, x_max, y_min, y_max].
    """
    load_trident_libraries()
    config = ROOT.snd.trident.TridentTruthConfig()
    config.zMin = z_min
    config.zMax = z_max
    config.targetZMin = target_z_min
    config.targetZMax = target_z_max
    config.processMask = process_mask
    config.weightScale = weight_scale
    config.useFiducial = use_fiducial

    if fid_box is not None and len(fid_box) == 5:
        config.fidZ = float(fid_box[0])
        config.fidXMin = float(fid_box[1])
        config.fidXMax = float(fid_box[2])
        config.fidYMin = float(fid_box[3])
        config.fidYMax = float(fid_box[4])
    else:
        config.fidZ = 320.0
        config.fidXMin = -42.0
        config.fidXMax = -11.0
        config.fidYMin = 18.0
        config.fidYMax = 49.0

    return config


def attach_truth_branches(df: ROOT.RDataFrame, truth_col: str = "truth") -> ROOT.RDataFrame:
    """
    Defines flattened scalar branches from a TridentTruthInfo column to RDataFrame,
    ensuring 100% backward-compatibility with downstream ntuples and analysis scripts.

    Args:
        df: RDataFrame with the truth column already defined.
        truth_col: Column name of the TridentTruthInfo struct. Default: 'truth'.
    """
    return (
        df
        .Define("is_signal", f"{truth_col}.hasCandidate")
        .Define("is_fiducial", f"{truth_col}.isFiducial")
        .Define("mc_weight", f"{truth_col}.scaledWeight")
        .Define("raw_mc_weight", f"{truth_col}.mcWeight")
        .Define("proc_type", f"static_cast<int>({truth_col}.procType)")
        .Define("region_type", f"static_cast<int>({truth_col}.regionType)")
        .Define("trident_flags", f"{truth_col}.tridentFlags")
        .Define("vtx_x", f"{truth_col}.vtxX")
        .Define("vtx_y", f"{truth_col}.vtxY")
        .Define("vtx_z", f"{truth_col}.vtxZ")
        .Define("rad_muon_id", f"{truth_col}.radMuonId")
        .Define("mu_minus_id", f"{truth_col}.muMinusId")
        .Define("mu_plus_id", f"{truth_col}.muPlusId")
        .Define("p_mu_in", f"{truth_col}.pMuIn")
        .Define("px_mu_in", f"{truth_col}.pxMuIn")
        .Define("py_mu_in", f"{truth_col}.pyMuIn")
        .Define("pz_mu_in", f"{truth_col}.pzMuIn")
        .Define("pt_mu_in", f"{truth_col}.ptMuIn")
        .Define("p_mu_minus", f"{truth_col}.pMuMinus")
        .Define("px_mu_minus", f"{truth_col}.pxMuMinus")
        .Define("py_mu_minus", f"{truth_col}.pyMuMinus")
        .Define("pz_mu_minus", f"{truth_col}.pzMuMinus")
        .Define("pt_mu_minus", f"{truth_col}.ptMuMinus")
        .Define("eta_mu_minus", f"{truth_col}.etaMuMinus")
        .Define("phi_mu_minus", f"{truth_col}.phiMuMinus")
        .Define("p_mu_plus", f"{truth_col}.pMuPlus")
        .Define("px_mu_plus", f"{truth_col}.pxMuPlus")
        .Define("py_mu_plus", f"{truth_col}.pyMuPlus")
        .Define("pz_mu_plus", f"{truth_col}.pzMuPlus")
        .Define("pt_mu_plus", f"{truth_col}.ptMuPlus")
        .Define("eta_mu_plus", f"{truth_col}.etaMuPlus")
        .Define("phi_mu_plus", f"{truth_col}.phiMuPlus")
        .Define("inv_mass_2mu", f"{truth_col}.invMass2Mu")
        .Define("opening_angle_mrad", f"{truth_col}.openingAngleMrad")
        .Define("pt_2mu", f"{truth_col}.pt2Mu")
        .Define("energy_asym", f"{truth_col}.energyAsym")
        .Define("delta_phi_2mu", f"{truth_col}.deltaPhi2Mu")
        .Define("delta_eta_2mu", f"{truth_col}.deltaEta2Mu")
        .Define("delta_r_2mu", f"{truth_col}.deltaR2Mu")
        .Define("inv_mass_3mu", f"{truth_col}.invMass3Mu")
        .Define("pt_3mu", f"{truth_col}.pt3Mu")
        .Define("opening_angle_rad_minus_mrad", f"{truth_col}.openingAngleRadMinusMrad")
        .Define("opening_angle_rad_plus_mrad", f"{truth_col}.openingAngleRadPlusMrad")
        .Define("energy_ratio_pair", f"{truth_col}.energyRatioPair")
        .Define("momentum_fraction_minus", f"{truth_col}.momentumFractionMinus")
        .Define("momentum_fraction_plus", f"{truth_col}.momentumFractionPlus")
        .Define("pt_asymmetry", f"{truth_col}.ptAsymmetry")
        .Define("cascade_depth", f"{truth_col}.cascadeDepth")
        .Define("emission_proc", f"static_cast<int>({truth_col}.emissionProc)")
        .Define("fid_x_minus", f"{truth_col}.fidXMinus")
        .Define("fid_y_minus", f"{truth_col}.fidYMinus")
        .Define("fid_x_plus", f"{truth_col}.fidXPlus")
        .Define("fid_y_plus", f"{truth_col}.fidYPlus")
        .Define("fid_x_rad", f"{truth_col}.fidXRad")
        .Define("fid_y_rad", f"{truth_col}.fidYRad")
        .Define("dist_to_fiducial_edge", f"{truth_col}.distToFiducialEdge")
        .Define("proc_name", f"std::string({truth_col}.getProcessName())")
        .Define("region_name", f"std::string({truth_col}.getRegionName())")
        .Define("emission_proc_name", f"std::string({truth_col}.getEmissionProcessName())")
    )
