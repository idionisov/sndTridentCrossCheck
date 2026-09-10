"""
yield_summary.py
----------------
Modular reporting functions for SND@LHC trident yield calculations.
Provides reusable banner and tabular summary printers for command-line scripts,
notebooks, and batch analysis jobs.
"""

from typing import List, Tuple, Dict, Any, Optional
from .data_manager import DataManager


def print_yield_header(
    data: DataManager,
    target_z_min: float,
    target_z_max: float,
    lumi_mc: float,
    lumi_target: float,
    num_threads: int,
    input_pattern: Optional[str] = None,
):
    """
    Prints a formatted configuration banner for the yield calculation run.

    Args:
        data: Initialized DataManager instance.
        target_z_min: Upstream target boundary in cm.
        target_z_max: Downstream target boundary in cm.
        lumi_mc: Equivalent integrated luminosity of the MC dataset in fb^-1.
        lumi_target: Target integrated luminosity to scale to in fb^-1.
        num_threads: Number of RDataFrame worker threads.
        input_pattern: Optional string pattern override for display.
    """
    weight_scale = lumi_target / lumi_mc if lumi_mc > 0 else 1.0
    pattern_str = input_pattern or (data.files[0] if len(data.files) == 1 else f"{len(data.files)} files")

    print("=" * 70)
    print("SND@LHC Trident Physical Yield Calculator")
    print("=" * 70)
    print(f"Input Pattern     : {pattern_str}")
    print(f"Files Found       : {data.num_files:,}")
    print(f"Tree Name         : {data.tree_name} (MC: {data.isMC()})")
    print(f"Total Entries     : {data.entries:,}")
    print(f"Target Z Region   : [{target_z_min:.1f}, {target_z_max:.1f}] cm")
    print(f"Dataset MC Lumi   : {lumi_mc:.4f} fb^-1")
    print(f"Target Lumi       : {lumi_target:.2f} fb^-1")
    print(f"Lumi Multiplier   : {weight_scale:.4f}x")
    print(f"Worker Threads    : {num_threads}")
    print("=" * 70)


def print_yield_summary(
    results: Dict[str, Any],
    processes: List[Tuple[int, str]],
    regions: List[Tuple[int, str]],
    lumi_target: float,
    elapsed_time: Optional[float] = None,
):
    """
    Prints comprehensive tabular breakdown tables for calculated trident yields:
      1) Process breakdown table (counts, percentages, physical yields)
      2) Spatial region breakdown table (Rock, Target, Muon System)
      3) 2D Process x Region cross-tabulation matrix

    Args:
        results: Dictionary mapping action keys to values (e.g. 'total_count', 'p_0_cnt', etc.).
        processes: List of (proc_id, proc_name) tuples.
        regions: List of (region_id, region_name) tuples.
        lumi_target: Target integrated luminosity in fb^-1.
        elapsed_time: Optional wall-clock runtime in seconds.
    """
    tot_cnt = int(results.get("total_count", 0))
    tot_yld = float(results.get("total_yield", 0.0)) if tot_cnt > 0 else 0.0

    if elapsed_time is not None:
        print(f"\nCalculation completed in {elapsed_time:.2f} seconds.")

    print("\n" + "=" * 70)
    print(f"PHYSICAL YIELD SUMMARY (Target Lumi: {lumi_target:.2f} fb^-1)")
    print("=" * 70)
    print(f"Total Raw Signal Events Selected : {tot_cnt:,}")
    print(f"Total Physical Expected Yield    : {tot_yld:,.2f} events")
    print("-" * 70)

    # [1] Process breakdown table
    print("\n[1] BREAKDOWN BY PROCESS:")
    print(f"{'Process':<36} | {'Raw MC Events':>13} | {'Raw %':>7} | {'Yield (' + str(lumi_target) + ' fb^-1)':>18} | {'Yield %':>7}")
    print("-" * 92)
    for p_id, p_name in processes:
        pcnt = int(results.get(f"p_{p_id}_cnt", 0))
        pyld = float(results.get(f"p_{p_id}_yld", 0.0)) if pcnt > 0 else 0.0
        p_pct_raw = (pcnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
        p_pct_yld = (pyld / tot_yld * 100.0) if tot_yld > 0 else 0.0
        print(f"{p_name:<36} | {pcnt:>13,} | {p_pct_raw:>6.2f}% | {pyld:>18.2f} | {p_pct_yld:>6.2f}%")
    print("-" * 92)
    print(f"{'Total All Processes':<36} | {tot_cnt:>13,} | 100.00% | {tot_yld:>18.2f} | 100.00%")

    # [2] Spatial region breakdown table
    print("\n[2] BREAKDOWN BY SPATIAL REGION:")
    print(f"{'Spatial Region':<36} | {'Raw MC Events':>13} | {'Raw %':>7} | {'Yield (' + str(lumi_target) + ' fb^-1)':>18} | {'Yield %':>7}")
    print("-" * 92)
    for r_id, r_name in regions:
        rcnt = int(results.get(f"r_{r_id}_cnt", 0))
        ryld = float(results.get(f"r_{r_id}_yld", 0.0)) if rcnt > 0 else 0.0
        r_pct_raw = (rcnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
        r_pct_yld = (ryld / tot_yld * 100.0) if tot_yld > 0 else 0.0
        print(f"{r_name:<36} | {rcnt:>13,} | {r_pct_raw:>6.2f}% | {ryld:>18.2f} | {r_pct_yld:>6.2f}%")
    print("-" * 92)
    print(f"{'Total All Regions':<36} | {tot_cnt:>13,} | 100.00% | {tot_yld:>18.2f} | 100.00%")

    # [3] 2D Cross Table (Process x Region)
    print("\n[3] CROSS MATRIX: PROCESS x REGION YIELDS:")
    col_w = 20
    header_str = f"{'Process':<28} | " + " | ".join([f"{r_name[:18]:^{col_w}}" for _, r_name in regions]) + f" | {'Total':^{col_w}}"
    print(header_str)
    print("-" * len(header_str))

    for p_id, p_name in processes:
        row_str = f"{p_name[:28]:<28} | "
        p_row_tot = 0.0
        for r_id, _ in regions:
            pr_cnt = int(results.get(f"pr_{p_id}_{r_id}_cnt", 0))
            pr_yld = float(results.get(f"pr_{p_id}_{r_id}_yld", 0.0)) if pr_cnt > 0 else 0.0
            p_row_tot += pr_yld
            row_str += f"{pr_yld:^{col_w}.2f} | "
        row_str += f"{p_row_tot:^{col_w}.2f}"
        print(row_str)

    print("-" * len(header_str))
    tot_row_str = f"{'Total':<28} | "
    for r_id, _ in regions:
        r_col_tot = float(results.get(f"r_{r_id}_yld", 0.0))
        tot_row_str += f"{r_col_tot:^{col_w}.2f} | "
    tot_row_str += f"{tot_yld:^{col_w}.2f}"
    print(tot_row_str)
    print("=" * len(header_str) + "\n")
