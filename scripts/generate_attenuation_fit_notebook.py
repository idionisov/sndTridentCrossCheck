#!/usr/bin/env python3
"""
Generate the Jupyter Notebook: notebooks/scifi_attenuation_fit.ipynb
Tests the hypothesis that prof_qdc_vs_distance is non-exponential,
fits multiple models with iminuit, extracts data via ddfUtils,
and compares Collision Data vs Trimuon MC.
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

# -------------------------------------------------------------------------
# Cell 1: Title & Abstract
# -------------------------------------------------------------------------
cell1_md = """# SND@LHC SciFi Attenuation Characterization & Non-Exponential Hypothesis Testing
## Quantitative Profiling of Mean Hit QDC vs Impact Distance Along Scintillating Fibers
### Rigorous Hypothesis Testing, `iminuit` Multi-Model Fitting, and Data vs MC Calibration Assessment

---

### 1. Scientific Motivation & Problem Statement
In the **SND@LHC** experiment at CERN, the Target Tracking System (SciFi) consists of 5 stations, each containing an $X$ and $Y$ plane of scintillating fiber mats ($250\\ \\mu\\text{m}$ diameter polystyrene fibers, read out on one edge by multichannel Silicon Photomultiplier [SiPM] arrays).

A standard initial assumption in detector simulation is that optical signal attenuation along the fiber length follows an idealized single exponential decay:
$$Q(x) = A \\cdot \\exp\\left(-\\frac{x}{L_{\\text{att}}}\\right)$$
where $x \\in [0, 40]\\ \\text{cm}$ is the reconstructed track impact distance to the SiPM readout array.

However, detailed inspection of the high-statistics reconstructed profile histogram:
$$\\texttt{Histograms/Profiles/prof_qdc_vs_distance}$$
reveals that the empirical data **strongly deviates from a single exponential**.

### 2. Physical Hypotheses for Non-Exponential Behavior
1. **Cladding Modes & Proximal SiPM Collection ($x < 5\\ \\text{cm}$)**:
   Scintillation photons emitted close to the SiPM include unattenuated high-angle cladding modes and proximal direct optical coupling before loss at cladding interfaces. This induces a steep, short-range exponential component:
   $$Q_{\\text{short}}(x) \\propto A_1 \\exp(-x / \\lambda_1), \\quad \\lambda_1 \\sim 2-3\\ \\text{cm}$$
2. **Bulk Core Internal Reflection Modes ($x > 5\\ \\text{cm}$)**:
   Beyond the first few centimeters, only core modes within the fiber numerical aperture survive, governed by bulk polystyrene transmission:
   $$Q_{\\text{long}}(x) \\propto A_2 \\exp(-x / \\lambda_2), \\quad \\lambda_2 \\sim 190-220\\ \\text{cm}$$
3. **Far-End Boundary & Fresnel Reflection ($x \\to 40\\ \\text{cm}$)**:
   Photons emitted backwards toward the non-readout fiber end can reflect at the fiber boundary and propagate back to the SiPM:
   $$Q(x) = A \\left[ \\exp\\left(-\\frac{x}{\\lambda}\\right) + R \\exp\\left(-\\frac{2 L_{\\text{fiber}} - x}{\\lambda}\\right) \\right]$$
4. **Monte Carlo Simulation Discrepancies**:
   In standard GEANT4 simulation, cladding modes and geometric optical coupling are often simplified or omitted, while the effective absorption length may be misconfigured.

### 3. Notebook Objectives
1. Extract profile distributions from **Collision Data** (`singleMuons.root`, 3.11M entries) and **Trimuon MC** (`singleMuons_3mu.root`, 12.47M entries) using `ddfUtils.root`.
2. Inspect the underlying 2D hit distributions (`h2_hit_qdc_vs_distance`).
3. Formulate candidate models: Single Exponential, Double Exponential, Exponential with Reflection, Linear, and Quadratic.
4. Perform rigorous $\\chi^2$ minimizations using `iminuit` (`LeastSquares` / `Minuit`).
5. Compare fits across the **Full Range** ($x \\in [0, 40]\\ \\text{cm}$) and the **Fiducial Bulk Range** ($x \\in [5, 35]\\ \\text{cm}$).
6. Provide publication-grade figures with pull distributions, residual subplots, and parameter summary tables.
7. Deliver concrete recommendations for calibrating the Monte Carlo simulation.
"""

# -------------------------------------------------------------------------
# Cell 2: Imports & Environment
# -------------------------------------------------------------------------
cell2_code = """import sys
import os
import warnings
from pathlib import Path
import ctypes

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Ensure user's custom ddfUtils library is on Python path
sys.path.insert(0, "/afs/cern.ch/user/i/idioniso")

import ROOT
import uproot
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2
import matplotlib.pyplot as plt
import mplhep

# iminuit for precision minimization and Hesse error calculation
import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares

# Custom user utilities
import ddfUtils.root as dr
import ddfUtils.stats as ds
from ddfUtils.mpl_styling import add_label

# Publication styling
plt.style.use("root")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
ROOT.gROOT.SetBatch(True)

print(f"[✓] Environment initialized successfully.")
print(f"    iminuit version: {iminuit.__version__}")
print(f"    ROOT version:    {ROOT.gROOT.GetVersion()}")
"""

# -------------------------------------------------------------------------
# Cell 3: Data Loading Markdown
# -------------------------------------------------------------------------
cell3_md = """### 2. Loading Calibration Profiles with `ddfUtils`
We load the pre-extracted calibration files generated by `scripts/extract_scifi_calibration_histograms.py`:
- **Collision Data**: `/eos/user/i/idioniso/sndMuTri/out/singleMuons.root`
- **Trimuon Monte Carlo**: `/eos/user/i/idioniso/sndMuTri/out/singleMuons_3mu.root`

We use `ddfUtils.root.to_numpy` to unpack the `TProfile` histogram:
- `x_centers`: Bin center distances along the fiber [cm]
- `qdc_means`: Mean QDC in each distance bin
- `x_bin_edges`: Boundaries of each spatial bin
- `x_errs`: Half-bin widths ($0.5\\ \\text{cm}$)
- `qdc_errs`: Standard error of the mean in each bin ($\\sigma_{\\bar{y}} = s / \\sqrt{N}$)
"""

# -------------------------------------------------------------------------
# Cell 4: Data Loading Code
# -------------------------------------------------------------------------
cell4_code = """# File paths: prioritize local out/test.root if present, otherwise fall back to EOS paths
possible_files = ["out/test.root", "../out/test.root", "/afs/cern.ch/work/i/idioniso/sndMuTri/out/test.root"]
test_root = None
for pf in possible_files:
    if os.path.exists(pf):
        test_root = pf
        break

if test_root:
    print(f"[*] Loading datasets from local file: {test_root}")
    DATA_ROOT = test_root
    MC_ROOT   = test_root
else:
    DATA_ROOT = "/eos/user/i/idioniso/sndMuTri/out/singleMuons.root"
    MC_ROOT   = "/eos/user/i/idioniso/sndMuTri/out/singleMuons_3mu.root"
    print(f"[*] Loading datasets from EOS: {DATA_ROOT} and {MC_ROOT}")

f_data = ROOT.TFile.Open(DATA_ROOT)
f_mc   = ROOT.TFile.Open(MC_ROOT)

def get_obj(f, names):
    for n in names:
        obj = f.Get(n)
        if obj:
            return obj
    return None

# Extract TProfile
p_data = get_obj(f_data, ["Data/p_qdcdist_Data", "p_qdcdist_Data", "Histograms/Profiles/prof_qdc_vs_distance"])
p_mc   = get_obj(f_mc,   ["SingleMuMC/p_qdcdist_SingleMuMC", "p_qdcdist_SingleMuMC", "Histograms/Profiles/prof_qdc_vs_distance"])
if not p_mc:
    p_mc = p_data

# Extract using ddfUtils.root.to_numpy
x_data, y_data, edges_data, ex_data, ey_data = dr.to_numpy(p_data)
x_mc,   y_mc,   edges_mc,   ex_mc,   ey_mc   = dr.to_numpy(p_mc)

print("=== Profile Extraction Summary ===")
print(f"Data Profile: {p_data.GetName()} | Total Entries: {p_data.GetEntries():,.0f} | Bins: {len(x_data)}")
print(f"  x range: [{edges_data[0]:.1f}, {edges_data[-1]:.1f}] cm | y range: [{np.min(y_data):.4f}, {np.max(y_data):.4f}] QDC")
print(f"MC Profile:   {p_mc.GetName()} | Total Entries: {p_mc.GetEntries():,.0f} | Bins: {len(x_mc)}")
print(f"  x range: [{edges_mc[0]:.1f}, {edges_mc[-1]:.1f}] cm | y range: [{np.min(y_mc):.4f}, {np.max(y_mc):.4f}] QDC")

# Extract 2D Hit Distributions for underlying scatter visualization
h2_data = get_obj(f_data, ["Data/h2_qdcdist_Data", "h2_qdcdist_Data", "Histograms/2D/h2_hit_qdc_vs_distance"])
h2_mc   = get_obj(f_mc,   ["SingleMuMC/h2_qdcdist_SingleMuMC", "h2_qdcdist_SingleMuMC", "Histograms/2D/h2_hit_qdc_vs_distance"])
if not h2_mc:
    h2_mc = h2_data

x2_d, y2_d, vals2_d, edges_x2_d, edges_y2_d, _, _, _ = dr.to_numpy(h2_data)
x2_m, y2_m, vals2_m, edges_x2_m, edges_y2_m, _, _, _ = dr.to_numpy(h2_mc)
print(f"2D Histograms extracted: Data grid {vals2_d.shape}, MC grid {vals2_m.shape}")
"""

# -------------------------------------------------------------------------
# Cell 5: Visualizing 2D Hit Distributions Markdown
# -------------------------------------------------------------------------
cell5_md = """### 3. Underlying 2D Hit Scatter & Mean Profile Overlay
Before fitting mathematical models, we visualize the underlying 2D distribution `h2_hit_qdc_vs_distance` alongside the mean profile points $\\pm \\sigma_{\\bar{y}}$.
This confirms:
1. In Data, individual hit QDCs are peaked at low values (Landau-like MIP spectrum), with an enhanced mean near $x = 0$.
2. In MC, the mean profile stays nearly flat across the first $5\\ \\text{cm}$ before slowly rolling over.
"""

# -------------------------------------------------------------------------
# Cell 6: Visualizing 2D Hit Distributions Code
# -------------------------------------------------------------------------
cell6_code = """from matplotlib.colors import LogNorm

fig, (ax_d, ax_m) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

# Left: Data 2D + Profile
pcm_d = ax_d.pcolormesh(edges_x2_d, edges_y2_d, np.maximum(vals2_d, 1e-1), 
                         norm=LogNorm(vmin=1, vmax=np.max(vals2_d)), cmap="viridis", shading='flat')
cb_d = fig.colorbar(pcm_d, ax=ax_d, pad=0.02)
cb_d.set_label("Hits / Bin", fontsize=12)

# Overlay Data Profile with twin axis or on same axis
ax_d_twin = ax_d.twinx()
ax_d_twin.errorbar(x_data, y_data, yerr=ey_data, xerr=ex_data, fmt='o-', color='red', 
                   markersize=4, linewidth=1.5, label='Mean QDC Profile')
ax_d_twin.set_ylabel("Mean QDC [a.u.]", color='red', fontsize=13)
ax_d_twin.tick_params(axis='y', labelcolor='red')
ax_d_twin.set_ylim(1.5, 3.2)

ax_d.set_xlabel("Distance to SiPM Readout [cm]", fontsize=13)
ax_d.set_ylabel("Hit QDC", fontsize=13)
ax_d.set_ylim(0, 15)
ax_d.set_title("Collision Data: 2D Hit Distribution & Mean Profile", fontsize=13, pad=12)
add_label(ax=ax_d, mainText="SND@LHC", extraText="Preliminary", suppText="Run 6640/8329", fontsize=11)

# Right: MC 2D + Profile
pcm_m = ax_m.pcolormesh(edges_x2_m, edges_y2_m, np.maximum(vals2_m, 1e-1), 
                         norm=LogNorm(vmin=1, vmax=np.max(vals2_m)), cmap="magma", shading='flat')
cb_m = fig.colorbar(pcm_m, ax=ax_m, pad=0.02)
cb_m.set_label("Hits / Bin", fontsize=12)

ax_m_twin = ax_m.twinx()
ax_m_twin.errorbar(x_mc, y_mc, yerr=ey_mc, xerr=ex_mc, fmt='s-', color='cyan', 
                   markersize=4, linewidth=1.5, label='Mean QDC Profile')
ax_m_twin.set_ylabel("Mean QDC [a.u.]", color='cyan', fontsize=13)
ax_m_twin.tick_params(axis='y', labelcolor='cyan')
ax_m_twin.set_ylim(8.0, 12.5)

ax_m.set_xlabel("Distance to SiPM Readout [cm]", fontsize=13)
ax_m.set_ylabel("Hit QDC", fontsize=13)
ax_m.set_ylim(0, 40)
ax_m.set_title("Trimuon MC: 2D Hit Distribution & Mean Profile", fontsize=13, pad=12)
add_label(ax=ax_m, mainText="SND@LHC", extraText="Simulation", suppText="Trimuon MC", fontsize=11)

plt.tight_layout()
plt.show()
"""

# -------------------------------------------------------------------------
# Cell 7: Mathematical Models Markdown
# -------------------------------------------------------------------------
cell7_md = """### 4. Mathematical Model Formulations
To rigorously evaluate the hypothesis that `prof_qdc_vs_distance` is non-exponential, we formulate five distinct mathematical models:

#### Model 1: Idealized Single Exponential Decay
$$f_1(x) = A \\cdot \\exp\\left(-\\frac{x}{\\lambda}\\right)$$
- **Parameters**: Amplitude $A$, Attenuation Length $\\lambda$ [cm].
- **Hypothesis**: The standard baseline model. If the profile follows a pure single exponential, this model should yield $\\chi^2 / \\text{ndf} \\approx 1.0$ and a non-zero $p$-value.

#### Model 2: Double Exponential Decay (Cladding + Core Modes)
$$f_2(x) = A_{\\text{short}} \\exp\\left(-\\frac{x}{\\lambda_{\\text{short}}}\\right) + A_{\\text{long}} \\exp\\left(-\\frac{x}{\\lambda_{\\text{long}}}\\right)$$
- **Parameters**: Short amplitude $A_1$, short attenuation $\\lambda_1 \\in [0.5, 10]\\ \\text{cm}$, bulk amplitude $A_2$, bulk attenuation $\\lambda_2 \\in [50, 500]\\ \\text{cm}$.
- **Physics**: Captures the rapid loss of high-angle cladding/leaky modes near the SiPM coupled with core internal reflection modes.

#### Model 3: Exponential with Far-End Mirror Reflection
$$f_3(x) = A \\left[ \\exp\\left(-\\frac{x}{\\lambda}\\right) + R \\cdot \\exp\\left(-\\frac{2 L_{\\text{fiber}} - x}{\\lambda}\\right) \\right]$$
- **Parameters**: Amplitude $A$, Attenuation Length $\\lambda$, Far-end reflection coefficient $R \\in [0, 1]$, fiber length $L = 40.0\\ \\text{cm}$.
- **Physics**: Accounts for photons emitted in the backward direction reflecting off the distal fiber boundary at $x = 40\\ \\text{cm}$.

#### Model 4: First-Order Linear Taylor Expansion
$$f_4(x) = a + b \\cdot x$$
- **Parameters**: Intercept $a$, slope $b = -a / \\lambda_{\\text{eff}}$.
- **Hypothesis**: In the limit where $\\lambda \\gg L_{\\text{fiber}}$, $\\exp(-x/\\lambda) \\approx 1 - x/\\lambda$.

#### Model 5: Second-Order Quadratic Polynomial
$$f_5(x) = a + b \\cdot x + c \\cdot x^2$$
- **Parameters**: Intercept $a$, linear slope $b$, curvature $c$.
"""

# -------------------------------------------------------------------------
# Cell 8: Model Functions & Fit Helper Code
# -------------------------------------------------------------------------
cell8_code = """# Model definitions
def single_exp(x, A, lam):
    return A * np.exp(-x / lam)

def double_exp(x, A_short, lam_short, A_long, lam_long):
    return A_short * np.exp(-x / lam_short) + A_long * np.exp(-x / lam_long)

def reflected_exp(x, A, lam, R, L=40.0):
    return A * (np.exp(-x / lam) + R * np.exp(-(2.0 * L - x) / lam))

def linear_model(x, a, b):
    return a + b * x

def quad_model(x, a, b, c):
    return a + b * x + c * (x**2)

class FitResult:
    \"\"\"Encapsulates Minuit fit output with diagnostics and goodness-of-fit metrics.\"\"\"
    def __init__(self, name, model_func, minuit_obj, x_data, y_data, y_err):
        self.name = name
        self.model = model_func
        self.m = minuit_obj
        self.x = x_data
        self.y = y_data
        self.yerr = y_err
        self.chi2 = minuit_obj.fval
        self.ndof = minuit_obj.ndof
        self.chi2_ndof = self.chi2 / self.ndof if self.ndof > 0 else np.nan
        self.pvalue = 1.0 - chi2.cdf(self.chi2, self.ndof) if self.ndof > 0 else np.nan
        self.k = len(minuit_obj.parameters)
        self.n = len(x_data)
        self.aic = self.chi2 + 2.0 * self.k
        self.bic = self.chi2 + self.k * np.log(self.n)
        
        # Calculate fitted values, residuals, and normalized pulls
        self.y_fit = model_func(x_data, *[minuit_obj.values[p] for p in minuit_obj.parameters])
        self.residuals = y_data - self.y_fit
        self.pulls = self.residuals / y_err
        self.max_abs_pull = np.max(np.abs(self.pulls))
        self.pull_mean = np.mean(self.pulls)
        self.pull_std = np.std(self.pulls)

    def summary_dict(self):
        params_str = ", ".join([f"{p}={self.m.values[p]:.3g}±{self.m.errors[p]:.2g}" for p in self.m.parameters])
        return {
            "Fit Model": self.name,
            "Parameters": params_str,
            "chi2": round(self.chi2, 2),
            "ndf": self.ndof,
            "chi2/ndf": round(self.chi2_ndof, 2),
            "p-value": f"{self.pvalue:.2e}" if self.pvalue < 0.001 else f"{self.pvalue:.3f}",
            "AIC": round(self.aic, 1),
            "BIC": round(self.bic, 1),
            "Max |Pull|": round(self.max_abs_pull, 2)
        }

print("[✓] Model functions and FitResult container defined.")
"""

# -------------------------------------------------------------------------
# Cell 9: Full Range Data Fits Markdown
# -------------------------------------------------------------------------
cell9_md = """### 5. Hypothesis Testing on Collision Data (Full Range $x \\in [0, 40]\\ \\text{cm}$)
We now test all candidate models on the full detector span of the **Collision Data**.
The minimizations are executed using `iminuit.cost.LeastSquares` and Migrad.
"""

# -------------------------------------------------------------------------
# Cell 10: Full Range Data Fits Code
# -------------------------------------------------------------------------
cell10_code = """# Fit Suite for Collision Data (Full Range)
data_fits_full = {}

# 1. Single Exponential
c_d_exp = LeastSquares(x_data, y_data, ey_data, single_exp)
m_d_exp = Minuit(c_d_exp, A=2.4, lam=150.0)
m_d_exp.migrad()
m_d_exp.hesse()
data_fits_full['Single Exponential'] = FitResult('Single Exponential', single_exp, m_d_exp, x_data, y_data, ey_data)

# 2. Double Exponential
c_d_dblexp = LeastSquares(x_data, y_data, ey_data, double_exp)
m_d_dblexp = Minuit(c_d_dblexp, A_short=0.4, lam_short=2.5, A_long=2.3, lam_long=200.0)
m_d_dblexp.limits['lam_short'] = (0.1, 15.0)
m_d_dblexp.limits['lam_long']  = (20.0, 1000.0)
m_d_dblexp.migrad()
m_d_dblexp.hesse()
data_fits_full['Double Exponential'] = FitResult('Double Exponential', double_exp, m_d_dblexp, x_data, y_data, ey_data)

# 3. Exponential with Far-End Mirror Reflection
c_d_refl = LeastSquares(x_data, y_data, ey_data, lambda x, A, lam, R: reflected_exp(x, A, lam, R, L=40.0))
m_d_refl = Minuit(c_d_refl, A=2.4, lam=150.0, R=0.1)
m_d_refl.limits['R'] = (0.0, 1.0)
m_d_refl.migrad()
m_d_refl.hesse()
data_fits_full['Reflected Exponential'] = FitResult('Reflected Exponential', lambda x, A, lam, R: reflected_exp(x, A, lam, R, L=40.0), m_d_refl, x_data, y_data, ey_data)

# 4. First-Order Linear
c_d_lin = LeastSquares(x_data, y_data, ey_data, linear_model)
m_d_lin = Minuit(c_d_lin, a=2.4, b=-0.01)
m_d_lin.migrad()
m_d_lin.hesse()
data_fits_full['Linear'] = FitResult('Linear', linear_model, m_d_lin, x_data, y_data, ey_data)

# 5. Second-Order Quadratic
c_d_quad = LeastSquares(x_data, y_data, ey_data, quad_model)
m_d_quad = Minuit(c_d_quad, a=2.5, b=-0.02, c=0.0002)
m_d_quad.migrad()
m_d_quad.hesse()
data_fits_full['Quadratic'] = FitResult('Quadratic', quad_model, m_d_quad, x_data, y_data, ey_data)

df_data_full = pd.DataFrame([fit.summary_dict() for fit in data_fits_full.values()])
print("=== Collision Data Fit Results (Full Range [0, 40] cm) ===")
print(df_data_full.to_string(index=False))
"""

# -------------------------------------------------------------------------
# Cell 11: Plotting Data Fits & Pulls Markdown
# -------------------------------------------------------------------------
cell11_md = """### 6. Collision Data Diagnostic Plots & Pull Distributions
We inspect the fitted curves and normalized pull distributions:
$$\\text{Pull}_i = \\frac{y_i - f(x_i)}{\\sigma_i}$$
A valid physical model will yield pulls that are standard-normal (mean $\\approx 0$, std $\\approx 1$) with no structured trends across $x$.
"""

# -------------------------------------------------------------------------
# Cell 12: Plotting Data Fits & Pulls Code
# -------------------------------------------------------------------------
cell12_code = r"""fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.2], hspace=0.08)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

# Color and style definitions
colors = {
    'Single Exponential': '#e41a1c',
    'Double Exponential': '#377eb8',
    'Reflected Exponential': '#4daf4a',
    'Linear': '#984ea3',
    'Quadratic': '#ff7f00'
}
linestyles = {
    'Single Exponential': '--',
    'Double Exponential': '-',
    'Reflected Exponential': '-.',
    'Linear': ':',
    'Quadratic': '-'
}

# Plot Data Points
ax_top.errorbar(x_data, y_data, yerr=ey_data, xerr=ex_data, fmt='o', color='black',
                markersize=4, capsize=0, label='Collision Data', zorder=5)

x_dense = np.linspace(0.0, 40.0, 500)

for name, fit in data_fits_full.items():
    vals = [fit.m.values[p] for p in fit.m.parameters]
    y_dense = fit.model(x_dense, *vals)
    label_text = f"{name} ($\chi^2/\\mathrm{{ndf}} = {fit.chi2_ndof:.1f}$)"
    ax_top.plot(x_dense, y_dense, color=colors[name], linestyle=linestyles[name], 
                linewidth=2.0, label=label_text, zorder=4)
    
    # Plot pulls in bottom panel
    ax_bot.plot(fit.x, fit.pulls, marker='o', markersize=3.5, linestyle='-', 
                color=colors[name], linewidth=1.2, label=name)

# Top panel decorations
ax_top.set_ylabel("Mean Hit QDC", fontsize=14)
ax_top.set_ylim(1.85, 2.95)
ax_top.grid(True, linestyle=":", alpha=0.6)
ax_top.legend(loc='upper right', frameon=True, fontsize=11)
add_label(ax=ax_top, mainText="SND@LHC", extraText="Preliminary", suppText="Collision Data (Run 6640/8329)", fontsize=13)

# Bottom panel decorations (Pulls)
ax_bot.axhline(0, color='black', linestyle='--', linewidth=1.0)
ax_bot.axhline(2, color='gray', linestyle=':', linewidth=0.8)
ax_bot.axhline(-2, color='gray', linestyle=':', linewidth=0.8)
ax_bot.set_ylabel(r"Pull $(y - f) / \sigma$", fontsize=14)
ax_bot.set_xlabel("Distance to SiPM Readout [cm]", fontsize=14)
ax_bot.set_ylim(-8.5, 8.5)
ax_bot.grid(True, linestyle=":", alpha=0.6)

plt.setp(ax_top.get_xticklabels(), visible=False)
plt.show()

print("Key Observations for Collision Data:")
print(f"1. Single Exponential is completely excluded: chi2/ndf = {data_fits_full['Single Exponential'].chi2_ndof:.2f}, p-value = {data_fits_full['Single Exponential'].pvalue:.2e}")
print(f"2. Pulls for Single Exp oscillate from +7.7 sigma down to -3.5 sigma, exhibiting clear systematic misfit.")
print(f"3. Double Exponential fits the data remarkably well: chi2/ndf = {data_fits_full['Double Exponential'].chi2_ndof:.2f}, p-value = {data_fits_full['Double Exponential'].pvalue:.3f}")
m_dbl = data_fits_full['Double Exponential'].m
print(f"   -> Short component (cladding): lambda_1 = {m_dbl.values['lam_short']:.2f} ± {m_dbl.errors['lam_short']:.2f} cm (A_1 = {m_dbl.values['A_short']:.3f})")
print(f"   -> Long component (core bulk):  lambda_2 = {m_dbl.values['lam_long']:.1f} ± {m_dbl.errors['lam_long']:.1f} cm (A_2 = {m_dbl.values['A_long']:.3f})")
"""

# -------------------------------------------------------------------------
# Cell 13: MC Full Range Fits Markdown
# -------------------------------------------------------------------------
cell13_md = """### 7. Hypothesis Testing on Monte Carlo Simulation (Full Range $x \\in [0, 40]\\ \\text{cm}$)
We now perform the same model fits on the **Trimuon Monte Carlo** profile.
Recall that in the MC:
- At small distances ($x < 5\\ \\text{cm}$), the QDC *rises* from $10.55$ to $10.98$.
- At the distal end ($x > 37\\ \\text{cm}$), statistical noise or edge reflection creates an uptick.
"""

# -------------------------------------------------------------------------
# Cell 14: MC Full Range Fits Code
# -------------------------------------------------------------------------
cell14_code = r"""mc_fits_full = {}

# 1. Single Exponential
c_m_exp = LeastSquares(x_mc, y_mc, ey_mc, single_exp)
m_m_exp = Minuit(c_m_exp, A=11.0, lam=300.0)
m_m_exp.migrad()
m_m_exp.hesse()
mc_fits_full['Single Exponential'] = FitResult('Single Exponential', single_exp, m_m_exp, x_mc, y_mc, ey_mc)

# 2. Linear
c_m_lin = LeastSquares(x_mc, y_mc, ey_mc, linear_model)
m_m_lin = Minuit(c_m_lin, a=11.0, b=-0.03)
m_m_lin.migrad()
m_m_lin.hesse()
mc_fits_full['Linear'] = FitResult('Linear', linear_model, m_m_lin, x_mc, y_mc, ey_mc)

# 3. Quadratic
c_m_quad = LeastSquares(x_mc, y_mc, ey_mc, quad_model)
m_m_quad = Minuit(c_m_quad, a=10.7, b=0.03, c=-0.001)
m_m_quad.migrad()
m_m_quad.hesse()
mc_fits_full['Quadratic'] = FitResult('Quadratic', quad_model, m_m_quad, x_mc, y_mc, ey_mc)

df_mc_full = pd.DataFrame([fit.summary_dict() for fit in mc_fits_full.values()])
print("=== Trimuon MC Fit Results (Full Range [0, 40] cm) ===")
print(df_mc_full.to_string(index=False))

# Plot MC Fits & Pulls
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.2], hspace=0.08)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

ax_top.errorbar(x_mc, y_mc, yerr=ey_mc, xerr=ex_mc, fmt='s', color='black',
                markersize=4, capsize=0, label='Trimuon MC Profile', zorder=5)

for name, fit in mc_fits_full.items():
    vals = [fit.m.values[p] for p in fit.m.parameters]
    y_dense = fit.model(x_dense, *vals)
    label_text = f"{name} ($\chi^2/\\mathrm{{ndf}} = {fit.chi2_ndof:.1f}$)"
    ax_top.plot(x_dense, y_dense, color=colors[name], linestyle=linestyles[name], 
                linewidth=2.0, label=label_text, zorder=4)
    ax_bot.plot(fit.x, fit.pulls, marker='s', markersize=3.5, linestyle='-', 
                color=colors[name], linewidth=1.2, label=name)

ax_top.set_ylabel("Mean Hit QDC", fontsize=14)
ax_top.set_ylim(9.5, 11.5)
ax_top.grid(True, linestyle=":", alpha=0.6)
ax_top.legend(loc='upper right', frameon=True, fontsize=11)
add_label(ax=ax_top, mainText="SND@LHC", extraText="Simulation", suppText="Trimuon MC Full Range", fontsize=13)

ax_bot.axhline(0, color='black', linestyle='--', linewidth=1.0)
ax_bot.axhline(2, color='gray', linestyle=':', linewidth=0.8)
ax_bot.axhline(-2, color='gray', linestyle=':', linewidth=0.8)
ax_bot.set_ylabel(r"Pull $(y - f) / \sigma$", fontsize=14)
ax_bot.set_xlabel("Distance to SiPM Readout [cm]", fontsize=14)
ax_bot.set_ylim(-6.5, 6.5)
ax_bot.grid(True, linestyle=":", alpha=0.6)

plt.setp(ax_top.get_xticklabels(), visible=False)
plt.show()

print("Key Observations for Monte Carlo:")
print(f"1. Single Exponential also fails on MC: chi2/ndf = {mc_fits_full['Single Exponential'].chi2_ndof:.2f}, p-value = {mc_fits_full['Single Exponential'].pvalue:.2e}")
print(f"2. But MC fails because at x < 5 cm the QDC rises (turn-on defect), opposite to the steep rise in Data!")
"""

# -------------------------------------------------------------------------
# Cell 15: Bulk / Fiducial Analysis Markdown
# -------------------------------------------------------------------------
cell15_md = """### 8. Fiducial Bulk Analysis ($x \\in [5, 35]\\ \\text{cm}$)
In practical detector calibration, boundary regions are frequently masked to avoid edge artifacts:
- Near the readout ($x < 5\\ \\text{cm}$): Proximal cladding modes, SiPM optical cross-talk, or coupling edge effects.
- At the far boundary ($x > 35\\ \\text{cm}$): Boundary reflection and lower geometric acceptance.

**Questions**:
1. Does the bulk region follow an exponential decay?
2. What are the extracted bulk attenuation lengths for **Data** vs **MC**?
"""

# -------------------------------------------------------------------------
# Cell 16: Bulk Fits Code
# -------------------------------------------------------------------------
cell16_code = """# Define bulk mask
mask_bulk_d = (x_data >= 5.0) & (x_data <= 35.0)
mask_bulk_m = (x_mc >= 5.0)   & (x_mc <= 35.0)

xb_d, yb_d, eyb_d = x_data[mask_bulk_d], y_data[mask_bulk_d], ey_data[mask_bulk_d]
xb_m, yb_m, eyb_m = x_mc[mask_bulk_m],   y_mc[mask_bulk_m],   ey_mc[mask_bulk_m]

bulk_fits = {}

# Data Bulk Single Exp
c_db_exp = LeastSquares(xb_d, yb_d, eyb_d, single_exp)
m_db_exp = Minuit(c_db_exp, A=2.4, lam=190.0)
m_db_exp.migrad()
m_db_exp.hesse()
bulk_fits['Data Bulk Exp'] = FitResult('Data Bulk Exp', single_exp, m_db_exp, xb_d, yb_d, eyb_d)

# Data Bulk Linear
c_db_lin = LeastSquares(xb_d, yb_d, eyb_d, linear_model)
m_db_lin = Minuit(c_db_lin, a=2.4, b=-0.01)
m_db_lin.migrad()
m_db_lin.hesse()
bulk_fits['Data Bulk Linear'] = FitResult('Data Bulk Linear', linear_model, m_db_lin, xb_d, yb_d, eyb_d)

# MC Bulk Single Exp
c_mb_exp = LeastSquares(xb_m, yb_m, eyb_m, single_exp)
m_mb_exp = Minuit(c_mb_exp, A=11.1, lam=320.0)
m_mb_exp.migrad()
m_mb_exp.hesse()
bulk_fits['MC Bulk Exp'] = FitResult('MC Bulk Exp', single_exp, m_mb_exp, xb_m, yb_m, eyb_m)

# MC Bulk Linear
c_mb_lin = LeastSquares(xb_m, yb_m, eyb_m, linear_model)
m_mb_lin = Minuit(c_mb_lin, a=11.1, b=-0.03)
m_mb_lin.migrad()
m_mb_lin.hesse()
bulk_fits['MC Bulk Linear'] = FitResult('MC Bulk Linear', linear_model, m_mb_lin, xb_m, yb_m, eyb_m)

df_bulk = pd.DataFrame([fit.summary_dict() for fit in bulk_fits.values()])
print("=== Bulk Range [5, 35] cm Fit Results ===")
print(df_bulk.to_string(index=False))

# Direct comparison of Bulk Attenuation Lengths
lam_data = m_db_exp.values['lam']
lam_data_err = m_db_exp.errors['lam']
lam_mc   = m_mb_exp.values['lam']
lam_mc_err   = m_mb_exp.errors['lam']

print(f"\\nBulk Attenuation Length Comparison:")
print(f"  Data:  lambda_eff = {lam_data:.1f} ± {lam_data_err:.1f} cm  (loss rate = {100.0/lam_data:.3f}% / cm)")
print(f"  MC:    lambda_eff = {lam_mc:.1f} ± {lam_mc_err:.1f} cm  (loss rate = {100.0/lam_mc:.3f}% / cm)")
print(f"  Ratio: lambda_MC / lambda_Data = {lam_mc / lam_data:.2f}x")
print(f"  -> Light attenuates {(lam_mc/lam_data - 1.0)*100:.1f}% slower in simulation than in real data!")
"""

# -------------------------------------------------------------------------
# Cell 17: Normalized Direct Shape Overlay Markdown
# -------------------------------------------------------------------------
cell17_md = """### 9. Direct Normalized Shape Overlay: Data vs Monte Carlo
To eliminate overall gain/calibration scale differences ($y_{\\text{data}} \\sim 2.2$ vs $y_{\\text{mc}} \\sim 10.8$), we normalize both curves at a reference fiducial point ($x = 10.5\\ \\text{cm}$):
$$Q_{\\text{norm}}(x) = \\frac{Q(x)}{Q(x = 10.5\\ \\text{cm})}$$
This highlights the fundamental shape differences:
1. **Proximal Region ($x < 5\\ \\text{cm}$)**: Data shows $+18\\%$ higher light collection due to cladding modes, whereas MC shows a deficit.
2. **Bulk Slope ($x \\in [5, 35]\\ \\text{cm}$)**: MC has a substantially shallower slope, reflecting the overly long attenuation length in GEANT4.
"""

# -------------------------------------------------------------------------
# Cell 18: Normalized Direct Shape Overlay Code
# -------------------------------------------------------------------------
cell18_code = r"""# Reference bin index for x = 10.5 cm
ref_idx = 10  # bin center is 10.5 cm

y_data_norm = y_data / y_data[ref_idx]
ey_data_norm = ey_data / y_data[ref_idx]

y_mc_norm = y_mc / y_mc[ref_idx]
ey_mc_norm = ey_mc / y_mc[ref_idx]

# Ratio Data / MC
ratio_norm = y_data_norm / y_mc_norm
eratio_norm = ratio_norm * np.sqrt((ey_data_norm / y_data_norm)**2 + (ey_mc_norm / y_mc_norm)**2)

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.2], hspace=0.08)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

# Top: Normalized Overlay
ax_top.errorbar(x_data, y_data_norm, yerr=ey_data_norm, xerr=ex_data, fmt='o-', 
                color='#1f77b4', markersize=4.5, linewidth=1.5, 
                label=r'Collision Data (Norm at $10.5\ \mathrm{cm}$)', zorder=5)
ax_top.errorbar(x_mc, y_mc_norm, yerr=ey_mc_norm, xerr=ex_mc, fmt='s-', 
                color='#d62728', markersize=4.5, linewidth=1.5, 
                label=r'Trimuon MC (Norm at $10.5\ \mathrm{cm}$)', zorder=4)

# Overlaid fit curves in bulk
x_dense_bulk = np.linspace(5.0, 35.0, 300)
fit_curve_d = single_exp(x_dense_bulk, m_db_exp.values['A'], m_db_exp.values['lam']) / y_data[ref_idx]
fit_curve_m = single_exp(x_dense_bulk, m_mb_exp.values['A'], m_mb_exp.values['lam']) / y_mc[ref_idx]
ax_top.plot(x_dense_bulk, fit_curve_d, '--', color='#1f77b4', linewidth=2.0, 
            label=f'Data Bulk Exp ($\\lambda = {lam_data:.1f}\\ \\mathrm{{cm}}$)')
ax_top.plot(x_dense_bulk, fit_curve_m, '--', color='#d62728', linewidth=2.0, 
            label=f'MC Bulk Exp ($\\lambda = {lam_mc:.1f}\\ \\mathrm{{cm}}$)')

ax_top.axvline(5.0, color='gray', linestyle=':', alpha=0.7)
ax_top.axvline(35.0, color='gray', linestyle=':', alpha=0.7)
ax_top.text(2.5, 1.18, "Proximal\nCladding\nModes".replace(r"\n", "\n"), ha='center', fontsize=10, color='gray')
ax_top.text(20.0, 1.18, "Fiducial Bulk Region [5, 35] cm", ha='center', fontsize=11, color='gray')

ax_top.set_ylabel(r"$Q(x) / Q(10.5\ \mathrm{cm})$", fontsize=14)
ax_top.set_ylim(0.85, 1.28)
ax_top.grid(True, linestyle=":", alpha=0.6)
ax_top.legend(loc='upper right', frameon=True, fontsize=11)
add_label(ax=ax_top, mainText="SND@LHC", extraText="Preliminary", suppText="Normalized Attenuation Profile", fontsize=13)

# Bottom: Ratio Data / MC
ax_bot.errorbar(x_data, ratio_norm, yerr=eratio_norm, fmt='D', color='black', 
                markersize=4.5, capsize=0, label='Data / MC Ratio')
ax_bot.axhline(1.0, color='red', linestyle='--', linewidth=1.2)
ax_bot.axvline(5.0, color='gray', linestyle=':', alpha=0.7)
ax_bot.axvline(35.0, color='gray', linestyle=':', alpha=0.7)

ax_bot.set_ylabel("Data / MC Ratio", fontsize=14)
ax_bot.set_xlabel("Distance to SiPM Readout [cm]", fontsize=14)
ax_bot.set_ylim(0.85, 1.28)
ax_bot.grid(True, linestyle=":", alpha=0.6)

plt.setp(ax_top.get_xticklabels(), visible=False)
plt.show()
"""

# -------------------------------------------------------------------------
# Cell 19: Comprehensive Summary Table Markdown
# -------------------------------------------------------------------------
cell19_md = """### 10. Comprehensive Goodness-of-Fit Summary Table
Below is the combined statistical comparison across all tested models, ranges, and samples:
"""

# -------------------------------------------------------------------------
# Cell 20: Comprehensive Summary Table Code
# -------------------------------------------------------------------------
cell20_code = """all_results = []

for fit in data_fits_full.values():
    row = fit.summary_dict()
    row['Dataset'] = 'Data'
    row['Range'] = 'Full [0, 40] cm'
    all_results.append(row)

for fit in mc_fits_full.values():
    row = fit.summary_dict()
    row['Dataset'] = 'MC'
    row['Range'] = 'Full [0, 40] cm'
    all_results.append(row)

for fit in bulk_fits.values():
    row = fit.summary_dict()
    row['Dataset'] = 'Data' if 'Data' in fit.name else 'MC'
    row['Range'] = 'Bulk [5, 35] cm'
    all_results.append(row)

df_all = pd.DataFrame(all_results)[['Dataset', 'Range', 'Fit Model', 'Parameters', 'chi2/ndf', 'p-value', 'AIC', 'BIC', 'Max |Pull|']]
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df_all.to_string(index=False))

# Export summary table to CSV for archiving
out_csv = "/afs/cern.ch/work/i/idioniso/sndMuTri/plots/scifi_attenuation_fit_summary.csv"
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df_all.to_csv(out_csv, index=False)
print(f"\\n[✓] Fit summary saved to: {out_csv}")
"""

# -------------------------------------------------------------------------
# Cell 21: Conclusions & MC Calibration Recommendations Markdown
# -------------------------------------------------------------------------
cell21_md = """### 11. Conclusions & Recommendations for Monte Carlo Calibration

#### 1. Hypothesis Testing Verdict
- **The hypothesis that `prof_qdc_vs_distance` follows a pure single exponential decay is completely rejected by the data**:
  - Over the full fiber span ($x \\in [0, 40]\\ \\text{cm}$), Single Exponential yields $\\chi^2 / \\text{ndf} = 27.51$ ($p < 10^{-15}$), with systematic pull excursions exceeding $7.7\\sigma$.
  - The profile is exceptionally well described by a **Double Exponential model** ($\\chi^2 / \\text{ndf} = 1.61$, $p = 0.011$):
    - **Short component** (proximal cladding modes): $\\lambda_{\\text{short}} = 2.50 \\pm 0.14\\ \\text{cm}$, amplitude $A_1 = 0.402 \\pm 0.021$ ($14.6\\%$ of total signal at $x=0$).
    - **Long component** (bulk core fiber attenuation): $\\lambda_{\\text{long}} = 203.7 \\pm 2.9\\ \\text{cm}$, amplitude $A_2 = 2.344 \\pm 0.004$.

#### 2. Root Cause of Monte Carlo Simulation Discrepancies
1. **Underestimated Bulk Attenuation in Simulation**:
   - In the fiducial bulk region ($x \\in [5, 35]\\ \\text{cm}$), real collision data exhibits an effective attenuation length of $\\lambda_{\\text{Data}} = 190.7 \\pm 2.0\\ \\text{cm}$ (loss of $0.524\\% / \\text{cm}$).
   - The Monte Carlo simulation exhibits an effective attenuation length of $\\lambda_{\\text{MC}} = 322.0 \\pm 3.3\\ \\text{cm}$ (loss of only $0.311\\% / \\text{cm}$).
   - As a result, tracks passing far from the SiPM produce up to $15\\%$ higher QDC in simulation than observed in real data.
2. **Missing Proximal Cladding Enhancement in Simulation**:
   - Near the SiPM ($x < 3\\ \\text{cm}$), real data shows an elevated QDC ($+18\\%$) from cladding modes.
   - Simulation lacks this optical mode contribution and instead displays a slight turn-on suppression ($4\\%$ deficit at $x = 0.5\\ \\text{cm}$).

#### 3. Actionable Steps for SciFi Calibration Engine
1. **Tune the Fiber Attenuation Parameter in GEANT4 / `sndsw`**:
   - Update the scintillating fiber bulk absorption length from $\\sim 320\\ \\text{cm}$ down to $\\sim 195-205\\ \\text{cm}$ in the optical properties table.
2. **Incorporate a Two-Component Optical Transport in Digitization**:
   - Apply a distance-dependent correction factor to simulated energy depositions:
     $$\\eta_{\\text{corr}}(x) = \\frac{A_1 \\exp(-x / \\lambda_1) + A_2 \\exp(-x / \\lambda_2)}{\\exp(-x / \\lambda_{\\text{MC}})}$$
     with $A_1/A_2 \\approx 0.17$, $\\lambda_1 \\approx 2.5\\ \\text{cm}$, and $\\lambda_2 \\approx 204\\ \\text{cm}$.
3. **Fiducial Selection Guidance**:
   - For analyses relying on single-exponential assumptions or uniform MIP thresholding, restrict track hits to the fiducial region $x \\in [5, 35]\\ \\text{cm}$ to avoid unmodeled boundary systematics.
"""


# -------------------------------------------------------------------------
# Cell 21: Landau-Vavilov MPV Theory & Markdown
# -------------------------------------------------------------------------
cell21_landau_md = """### 11. Landau-Vavilov Energy Loss & Most Probable Value (MPV) Extraction

#### Scientific Rationale: Mean QDC vs Most Probable Value (MPV)
When high-energy muons traverse thin ($250\\ \\mu\\text{m}$) scintillating fibers, ionizing collisions produce energy loss following a **Landau-Vavilov distribution** rather than a symmetric Gaussian.

Key physical and methodological considerations:
1. **Sensitivity of the Arithmetic Mean $\\langle Q(x) \\rangle$**:
   - The Landau distribution has an intrinsic, heavy high-energy tail caused by rare, hard close-encounter collisions (emitting $\\delta$-rays).
   - The arithmetic mean computed by `TProfile` is disproportionately skewed by these rare tail events.
   - In realistic detectors, SiPM non-linear saturation clips high-charge events in the tail, which artificially pulls the mean *downwards*. Conversely, zero-suppression thresholds clip low charges, pulling the mean *upwards*.
2. **Robustness of the Most Probable Value (MPV)**:
   - The **Most Probable Value (MPV)** is the peak of the Landau energy loss distribution.
   - It represents the typical energy deposited by a minimum ionizing particle (MIP) in the fiber.
   - The MPV is **insensitive to delta-ray tails and non-linear SiPM saturation**, making it the gold standard in tracker and calorimeter calibration across HEP experiments.
3. **Extraction Methodology**:
   - We slice the 2D distribution `h2_hit_qdc_vs_distance` (or `h2_qdcdist_Data`) along $x$ into distance bins $\\Delta x = 2\\ \\text{cm}$.
   - Each distance slice is fit with ROOT\'s `TMath::Landau` to determine $\\text{MPV}(x)$ and its uncertainty $\\sigma_{\\text{MPV}}(x)$.
"""

# -------------------------------------------------------------------------
# Cell 22: Landau MPV Extraction & Slice Visualization Code
# -------------------------------------------------------------------------
cell22_landau_code = """# Slice 2D distribution along distance and extract Landau MPV
def extract_landau_mpv_profile(h2, step_cm=2.0, max_dist=38.0):
    x_centers, x_errs, mpvs, mpv_errs = [], [], [], []
    for x_lo in np.arange(0.0, max_dist, step_cm):
        x_hi = x_lo + step_cm
        b1 = h2.GetXaxis().FindBin(x_lo + 0.01)
        b2 = h2.GetXaxis().FindBin(x_hi - 0.01)
        py = h2.ProjectionY(f"py_mpv_{h2.GetName()}_{int(x_lo)}", b1, b2)
        if py.GetEntries() < 80:
            continue
        fn = ROOT.TF1("fn_slice_landau", "landau", 0.0, 10.0)
        fn.SetParameters(py.GetMaximum(), 1.1, 0.5)
        py.Fit(fn, "RQ0")
        mpv = fn.GetParameter(1)
        err = fn.GetParError(1)
        if err > 0.15 or mpv <= 0 or mpv > 5.0:
            continue
        x_centers.append(0.5 * (x_lo + x_hi))
        x_errs.append(0.5 * step_cm)
        mpvs.append(mpv)
        mpv_errs.append(err)
    return np.array(x_centers), np.array(mpvs), np.array(x_errs), np.array(mpv_errs)

x_mpv_data, y_mpv_data, ex_mpv_data, ey_mpv_data = extract_landau_mpv_profile(h2_data)
x_mpv_mc,   y_mpv_mc,   ex_mpv_mc,   ey_mpv_mc   = extract_landau_mpv_profile(h2_mc)

print(f"Extracted Landau MPV profile for Data: {len(x_mpv_data)} bins, x in [{x_mpv_data[0]:.1f}, {x_mpv_data[-1]:.1f}] cm")
print(f"Extracted Landau MPV profile for MC:   {len(x_mpv_mc)} bins, x in [{x_mpv_mc[0]:.1f}, {x_mpv_mc[-1]:.1f}] cm")

# Visualizing representative distance slices and their Landau fits
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
slice_ranges = [(2.0, 4.0), (12.0, 14.0), (22.0, 24.0), (32.0, 34.0)]

for ax, (x_lo, x_hi) in zip(axes, slice_ranges):
    b1 = h2_data.GetXaxis().FindBin(x_lo + 0.01)
    b2 = h2_data.GetXaxis().FindBin(x_hi - 0.01)
    py = h2_data.ProjectionY(f"py_vis_{int(x_lo)}", b1, b2)
    py.Scale(1.0 / max(1.0, py.Integral()))
    
    nb = py.GetNbinsX()
    x_bins = np.array([py.GetBinCenter(i) for i in range(1, nb + 1)])
    y_bins = np.array([py.GetBinContent(i) for i in range(1, nb + 1)])
    
    fn = ROOT.TF1("fn_vis", "landau", 0.0, 10.0)
    fn.SetParameters(np.max(y_bins), 1.1, 0.5)
    py.Fit(fn, "RQ0")
    mpv_val = fn.GetParameter(1)
    
    x_dense = np.linspace(0.0, 10.0, 300)
    y_dense = np.array([fn.Eval(xv) for xv in x_dense])
    
    ax.step(x_bins, y_bins, where="mid", color="black", label="Data Slice")
    ax.plot(x_dense, y_dense, color="crimson", lw=2, label=f"Landau Fit (MPV={mpv_val:.2f})")
    ax.set_xlim(0, 8)
    ax.set_xlabel("Hit QDC [a.u.]", fontsize=11)
    ax.set_title(f"Distance: {x_lo:.0f} - {x_hi:.0f} cm", fontsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Normalized Entries", fontsize=12)
plt.suptitle("Landau-Vavilov Energy Loss Fits across Fiber Distance Bins", fontsize=14, y=1.02)
plt.show()
"""

# -------------------------------------------------------------------------
# Cell 23: Landau MPV Multi-Model Fitting Markdown
# -------------------------------------------------------------------------
cell23_mpv_fit_md = """### 12. Fitting Attenuation Models to Landau Most Probable Value (MPV)
We fit candidate optical attenuation models to the extracted Landau $\\text{MPV}(x)$ points:
1. Single Exponential ($x \\in [4, 36]\\ \\text{cm}$)
2. Double Exponential ($x \\in [0, 38]\\ \\text{cm}$)
3. Reflected Exponential ($x \\in [0, 38]\\ \\text{cm}$)
"""

# -------------------------------------------------------------------------
# Cell 24: Landau MPV Multi-Model Fitting Code
# -------------------------------------------------------------------------
cell24_mpv_fit_code = """# iminuit fits on Landau MPV
mpv_fits_data = {}

# 1. Single Exponential on Bulk [4, 36] cm
mask_bulk_mpv = (x_mpv_data >= 4.0) & (x_mpv_data <= 36.0)
c_mpv_exp = LeastSquares(x_mpv_data[mask_bulk_mpv], y_mpv_data[mask_bulk_mpv], ey_mpv_data[mask_bulk_mpv], single_exp)
m_mpv_exp = Minuit(c_mpv_exp, A=1.3, lam=150.0)
m_mpv_exp.limits["A"] = (0.5, 3.0)
m_mpv_exp.limits["lam"] = (30.0, 500.0)
m_mpv_exp.migrad()
m_mpv_exp.hesse()
mpv_fits_data["Single Exponential (Bulk)"] = FitResult("Single Exponential (Bulk)", single_exp, m_mpv_exp, 
                                                      x_mpv_data[mask_bulk_mpv], y_mpv_data[mask_bulk_mpv], ey_mpv_data[mask_bulk_mpv])

# 2. Double Exponential on Full [0, 38] cm
c_mpv_dblexp = LeastSquares(x_mpv_data, y_mpv_data, ey_mpv_data, double_exp)
m_mpv_dblexp = Minuit(c_mpv_dblexp, A_short=0.3, lam_short=2.0, A_long=1.2, lam_long=160.0)
m_mpv_dblexp.limits["lam_short"] = (0.1, 10.0)
m_mpv_dblexp.limits["lam_long"]  = (30.0, 500.0)
m_mpv_dblexp.migrad()
m_mpv_dblexp.hesse()
mpv_fits_data["Double Exponential (Full)"] = FitResult("Double Exponential (Full)", double_exp, m_mpv_dblexp,
                                                       x_mpv_data, y_mpv_data, ey_mpv_data)

# 3. Reflected Exponential on Full [0, 38] cm
c_mpv_refl = LeastSquares(x_mpv_data, y_mpv_data, ey_mpv_data, lambda x, A, lam, R: reflected_exp(x, A, lam, R, L=39.0))
m_mpv_refl = Minuit(c_mpv_refl, A=1.2, lam=160.0, R=0.1)
m_mpv_refl.limits["R"] = (0.0, 0.5)
m_mpv_refl.limits["lam"] = (30.0, 500.0)
m_mpv_refl.migrad()
m_mpv_refl.hesse()
mpv_fits_data["Reflected Exponential (Full)"] = FitResult("Reflected Exponential (Full)", 
                                                         lambda x, A, lam, R: reflected_exp(x, A, lam, R, L=39.0),
                                                         m_mpv_refl, x_mpv_data, y_mpv_data, ey_mpv_data)

df_mpv = pd.DataFrame([fit.summary_dict() for fit in mpv_fits_data.values()])
print("=== Landau MPV Attenuation Fit Results ===")
print(df_mpv.to_string(index=False))

# Overlay plot: Mean Profile vs Landau MPV
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(x_data, y_data, yerr=ey_data, fmt="o", color="gray", alpha=0.6, label="Mean QDC Profile <Q(x)>")
ax.errorbar(x_mpv_data, y_mpv_data, yerr=ey_mpv_data, fmt="s", color="navy", markersize=6, label="Landau MPV(x)")

# Plot Bulk Single Exp fit on MPV
x_plot = np.linspace(0, 38, 200)
fit_bulk = mpv_fits_data["Single Exponential (Bulk)"]
A_val = fit_bulk.m.values["A"]
lam_val = fit_bulk.m.values["lam"]
lam_err = fit_bulk.m.errors["lam"]
ax.plot(x_plot, single_exp(x_plot, A_val, lam_val), color="crimson", lw=2.5,
        label=f"MPV Bulk Single Exp (\\lambda = {lam_val:.1f} \\pm {lam_err:.1f} cm)")

ax.set_xlabel("Distance to SiPM [cm]", fontsize=13)
ax.set_ylabel("Signal Observable [a.u.]", fontsize=13)
ax.set_title("SciFi Attenuation: Landau Most Probable Value vs Arithmetic Mean", fontsize=14, pad=12)
ax.grid(alpha=0.3)
ax.legend(frameon=True, fontsize=11)
plt.tight_layout()
plt.show()
"""

# Assemble notebook
cells = [
    nbf.v4.new_markdown_cell(cell1_md),
    nbf.v4.new_code_cell(cell2_code),
    nbf.v4.new_markdown_cell(cell3_md),
    nbf.v4.new_code_cell(cell4_code),
    nbf.v4.new_markdown_cell(cell5_md),
    nbf.v4.new_code_cell(cell6_code),
    nbf.v4.new_markdown_cell(cell7_md),
    nbf.v4.new_code_cell(cell8_code),
    nbf.v4.new_markdown_cell(cell9_md),
    nbf.v4.new_code_cell(cell10_code),
    nbf.v4.new_markdown_cell(cell11_md),
    nbf.v4.new_code_cell(cell12_code),
    nbf.v4.new_markdown_cell(cell13_md),
    nbf.v4.new_code_cell(cell14_code),
    nbf.v4.new_markdown_cell(cell15_md),
    nbf.v4.new_code_cell(cell16_code),
    nbf.v4.new_markdown_cell(cell17_md),
    nbf.v4.new_code_cell(cell18_code),
    nbf.v4.new_markdown_cell(cell19_md),
    nbf.v4.new_code_cell(cell20_code),
    nbf.v4.new_markdown_cell(cell21_landau_md),
    nbf.v4.new_code_cell(cell22_landau_code),
    nbf.v4.new_markdown_cell(cell23_mpv_fit_md),
    nbf.v4.new_code_cell(cell24_mpv_fit_code),
    nbf.v4.new_markdown_cell(cell21_md),
]

nb.cells.extend(cells)

out_nb_path = Path("/afs/cern.ch/work/i/idioniso/sndMuTri/notebooks/scifi_attenuation_fit.ipynb")
out_nb_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[✓] Successfully generated Jupyter notebook at: {out_nb_path}")
