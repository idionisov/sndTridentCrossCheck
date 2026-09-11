"""
Progress reporting utilities for ROOT RDataFrame workflows in SND@LHC.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any
import ROOT
from .data_manager import load_trident_libraries


class ProgressPrinter:
    """
    Python wrapper around high-performance C++ snd::trident::ProgressPrinter.

    Thread-safe, lock-free counter with configurable time-based (e.g. 30s)
    and/or event-based periodic status logging.
    """

    def __init__(
        self,
        total_events: int = 0,
        every_seconds: float = 30.0,
        every_events: int = 0,
        label: str = "Progress",
        log_mode: bool = True,
    ):
        load_trident_libraries()
        self._cpp = ROOT.snd.trident.ProgressPrinter(
            int(every_events),
            int(total_events),
            float(every_seconds),
            str(label),
            bool(log_mode),
        )

    @property
    def cpp(self) -> ROOT.snd.trident.ProgressPrinter:
        return self._cpp

    def attach(self, df: ROOT.RDataFrame) -> ROOT.RDataFrame:
        """
        Attaches this progress printer as an identity filter to the RDataFrame graph.
        """
        df_out = df.Filter(self._cpp, ["rdfentry_"])
        df_out._progress_printer = self
        return df_out

    def print_summary(self) -> None:
        """
        Prints the final completion summary line.
        """
        self._cpp.PrintSummary()

    def reset(self) -> None:
        """
        Resets counters and timer.
        """
        self._cpp.Reset()

    @property
    def processed(self) -> int:
        return int(self._cpp.GetProcessed())

    @property
    def total(self) -> int:
        return int(self._cpp.GetTotal())

    @total.setter
    def total(self, val: int) -> None:
        self._cpp.SetTotal(int(val))

    @property
    def rate(self) -> float:
        return float(self._cpp.GetRate())

    @property
    def elapsed_seconds(self) -> float:
        return float(self._cpp.GetElapsedSeconds())


def add_progress_printer(
    df: ROOT.RDataFrame,
    total_events: int = 0,
    every_seconds: float = 30.0,
    every_events: int = 0,
    label: str = "Progress",
    log_mode: bool = True,
) -> Tuple[ROOT.RDataFrame, ROOT.snd.trident.ProgressPrinter]:
    """
    Attaches a thread-safe ProgressPrinter to the RDataFrame.

    Returns:
        Tuple of (modified_df, cpp_printer_instance)
    """
    load_trident_libraries()
    printer = ROOT.snd.trident.ProgressPrinter(
        int(every_events),
        int(total_events),
        float(every_seconds),
        str(label),
        bool(log_mode),
    )
    df_out = df.Filter(printer, ["rdfentry_"])
    df_out._progress_printer = printer
    return df_out, printer


def attach_progress_printer(
    df: ROOT.RDataFrame,
    total_events: int = 0,
    every_seconds: float = 30.0,
    every_events: int = 0,
    label: str = "Progress",
    log_mode: bool = True,
) -> ROOT.RDataFrame:
    """
    Convenience helper that attaches a progress printer and directly returns
    the modified RDataFrame for clean call chaining.
    """
    df_out, _ = add_progress_printer(
        df,
        total_events=total_events,
        every_seconds=every_seconds,
        every_events=every_events,
        label=label,
        log_mode=log_mode,
    )
    return df_out
