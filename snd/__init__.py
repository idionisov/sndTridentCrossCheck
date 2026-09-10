"""
snd
---
SND@LHC Trident Analysis Framework & Tools.
"""

from .data_manager import DataManager, load_trident_libraries
from .yield_summary import print_yield_header, print_yield_summary
from .truth_branches import attach_truth_branches, create_truth_config

get_geo_path = DataManager.get_geo_path_for_run
get_geometry = DataManager.fetch_geometry
get_tchain = DataManager.fetch_tchain
get_data_base_path = DataManager.get_data_base_path

__all__ = [
    "DataManager",
    "load_trident_libraries",
    "print_yield_header",
    "print_yield_summary",
    "attach_truth_branches",
    "create_truth_config",
    "get_geo_path",
    "get_geometry",
    "get_tchain",
    "get_data_base_path",
]
