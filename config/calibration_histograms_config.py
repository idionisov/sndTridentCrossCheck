"""
================================================================================
SND@LHC SciFi Muon Calibration Histograms & Master Config Interface
================================================================================
Loads configuration from config/scifi_calibration_config.yaml and provides:
1. Job IO and argparse execution defaults
2. Physical cut parameters and tracking algorithms
3. Ordered cut pipeline with enable/disable statuses
4. 1D, 2D, and Profile histogram specifications
================================================================================
"""

import os
from typing import Dict, Any, List, Tuple
import yaml

_current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML_PATH = os.path.join(_current_dir, "scifi_calibration_config.yaml")


def load_scifi_calibration_config(yaml_path: str = None) -> Dict[str, Any]:
    """Loads the master SciFi calibration configuration from YAML."""
    if not yaml_path:
        yaml_path = DEFAULT_YAML_PATH

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"SciFi calibration YAML config not found at: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


# Load the default configuration
_cfg = load_scifi_calibration_config(DEFAULT_YAML_PATH)

JOB_DEFAULTS = _cfg.get("job", {})
PHYSICS_PARAMS = _cfg.get("parameters", {})
CUT_PIPELINE = _cfg.get("cuts", {}).get("pipeline", [])

# Convert lists to tuples for compatibility with legacy code
CALIB_CONFIGS_1D: List[Tuple] = [tuple(item) for item in _cfg.get("histograms_1d", [])]
CALIB_CONFIGS_2D: List[Tuple] = [tuple(item) for item in _cfg.get("histograms_2d", [])]
CALIB_PROFILE_CONFIGS: List[Tuple] = [tuple(item) for item in _cfg.get("profiles", [])]
