#!/usr/bin/env python3
"""
================================================================================
SND@LHC Calibration Grid Configuration Interface & Combination Generator
================================================================================
Provides programmatic access to the calibration grid configuration:
- Loads and parses `calibration_grid_config.yaml`
- Generates the full Cartesian product of parameter combinations (ts, s_qdc, b0)
- Exports formatted parameter tables and execution manifests
================================================================================
"""

import os
import sys
import itertools
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    import yaml
except ImportError:
    yaml = None

_config_dir = Path(__file__).resolve().parent
DEFAULT_YAML_PATH = _config_dir / "calibration_grid_config.yaml"


def load_grid_config(yaml_path: str = "") -> Dict[str, Any]:
    """Loads calibration grid configuration from YAML file."""
    path = Path(yaml_path) if yaml_path else DEFAULT_YAML_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if yaml is None:
        raise ImportError("PyYAML is required to parse YAML configs. Please install pyyaml or use python dict.")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_all_combinations(yaml_path: str = "") -> List[Dict[str, Any]]:
    """
    Generates the complete Cartesian product list of all parameter combinations:
    each entry contains {combo_id, ts, s_qdc, b0, tag}.
    """
    cfg = load_grid_config(yaml_path)
    p = cfg.get("parameters", {})
    thresholds = p.get("thresholds", [3.5])
    qdc_scales = p.get("qdc_scales", [4.5])
    qdc_offsets = p.get("qdc_offsets", [0.0])

    combinations = []
    combo_id = 1
    for ts, s_qdc, b0 in itertools.product(thresholds, qdc_scales, qdc_offsets):
        tag = f"ts_{ts:.2f}_s_{s_qdc:.2f}_b_{b0:.2f}".replace(".", "p").replace("-", "m")
        combinations.append({
            "combo_id": combo_id,
            "ts": float(ts),
            "s_qdc": float(s_qdc),
            "b0": float(b0),
            "tag": tag,
            "ts_tag": f"ts_{ts:.2f}".replace(".", "p")
        })
        combo_id += 1

    return combinations


def print_grid_summary(yaml_path: str = ""):
    """Prints a clean overview of the grid configuration and combinations count."""
    cfg = load_grid_config(yaml_path)
    p = cfg["parameters"]
    combos = get_all_combinations(yaml_path)

    print("=" * 80)
    print("SND@LHC CALIBRATION PARAMETER GRID CONFIGURATION")
    print("=" * 80)
    print(f"Config File           : {yaml_path or DEFAULT_YAML_PATH}")
    print(f"SciFi Thresholds (ts) : {p['thresholds']} p.e. (Count = {len(p['thresholds'])})")
    print(f"QDC Scales (s_QDC)    : {p['qdc_scales']} (Count = {len(p['qdc_scales'])})")
    print(f"QDC Offsets (b0)      : {p['qdc_offsets']} a.u. (Count = {len(p['qdc_offsets'])})")
    print("-" * 80)
    print(f"TOTAL PARAMETER POINTS: {len(combos)} combinations")
    print(f"Loss Weights          : w_hits = {cfg['loss']['weight_hits']}, w_qdc = {cfg['loss']['weight_qdc']}")
    print(f"Data References       : Hits = '{cfg['data_reference']['hits_file']}', QDC = '{cfg['data_reference']['qdc_file']}'")
    print(f"Simulation File       : {cfg['simulation']['input_reco_file']}")
    print("=" * 80)


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else ""
    print_grid_summary(cfg_path)
