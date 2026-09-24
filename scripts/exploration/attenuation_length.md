# SciFi Optical Attenuation Length & Energy Loss Characterization

This document provides a detailed explanation of the 4-pad canvas (`landau_vavilov_energy_loss.png`), justifying each step of the analysis pipeline, explaining the physics of individual hits versus clusters, and demonstrating how the fiber light attenuation length ($\lambda_{\text{att}}$) is measured in SND@LHC.

---

## 1. Overview of the Four Pads

```
+------------------------------------------+------------------------------------------+
|  Pad 1: Single Hit QDC                   |  Pad 2: Cluster Integrated QDC          |
|  (Individual SciFi Channels, 250 µm)     |  (Track-Matched Clusters, LanGaus Fit)   |
+------------------------------------------+------------------------------------------+
|  Pad 3: Cluster Distance Slices          |  Pad 4: Attenuation Extraction           |
|  (LanGaus Peak vs Fiber Distance)        |  (Exponential Fit: Peak vs Distance)    |
+------------------------------------------+------------------------------------------+
```

---

## 2. Pad 1: Single Hit QDC (Upper-Left)

### What is shown?
The charge ($Q_{\text{DC}}$) recorded by individual single SciFi channels (250 $\mu\text{m}$ diameter scintillating fibers) matched within $\Delta r \le 1.0\text{ mm}$ of an extrapolated muon track.
- **SingleMuMC (Blue)** clearly separates into two distinct structures:
  1. A sharp **pedestal / noise peak** centered around $Q_{\text{DC}} \approx 0$ ($\sigma \approx 0.61$).
  2. A distinct **Landau ionization MIP peak** at $Q_{\text{DC}} \approx 1.79$.
- **Data (Black)** shows a broad, single-envelope distribution with no dip separating noise from signal.

### Why does Data look like one broad peak while MC has two?
1. **Photoelectron Yield per Fiber**:
   - A single fiber has a diameter of only $250\text{ }\mu\text{m}$.
   - A passing minimum ionizing particle (MIP) deposits relatively little energy in an individual fiber, creating on average only **$1$ to $4$ photoelectrons (p.e.)** at the SiPM photocathode.
   - For a single fiber, a $1\text{--}2\text{ p.e.}$ signal is very close in charge to electronics pedestal noise fluctuations ($\sigma_{\text{noise}} \approx 1.4\text{ a.u.}$).
2. **Cluster Sharing**:
   - In actual data, scintillation light and ionization are distributed across several adjacent fibers ($2\text{--}4$ fibers per cluster).
   - Consequently, peripheral channels in the cluster often receive only a single photon or electronic cross-talk, blurring the boundary between the true zero-charge pedestal and a real single-fiber MIP signal.
3. **Is Pad 1 used for attenuation extraction?**
   - **No.** Pad 1 is solely a diagnostic check for single-channel behavior. Single-hit QDC cannot reliably measure attenuation because the noise pedestal and threshold effects distort the true energy loss.

---

## 3. Pad 2: Cluster Integrated QDC (Upper-Right)

### What is shown?
The total cluster charge:
$$Q_{\text{cluster}} = \sum_{i \in \text{cluster}} Q_{\text{DC}, i}$$
for clusters within $\Delta r \le 1.0\text{ mm}$ of the reconstructed SciFi track.

### Why switch to Clusters?
1. **Full Energy Collection**:
   Summing across all fibers in the track cluster collects the full energy deposit of the muon ($\approx 15\text{--}30\text{ p.e.}$ total).
2. **Pedestal Noise Elimination**:
   Because the total deposit is well above the multi-channel pedestal, the noise peak around zero disappears entirely, revealing a clean, continuous ionization distribution.
3. **The LanGaus Model (Landau $\otimes$ Gaussian)**:
   - Energy loss by ionization in thin absorbers follows a **Landau-Vavilov distribution** (asymmetric tail towards high energy due to rare, energetic $\delta$-ray knock-on electrons).
   - In a real detector, the pure Landau distribution is smeared by experimental resolution:
     - Poisson statistics of primary scintillation photons and photoelectrons ($1/\sqrt{N_{\text{p.e.}}}$).
     - SiPM pixel gain dispersion and optical cross-talk.
     - Multi-channel front-end electronics noise.
   - Therefore, the data is fitted with a numerical convolution of a Landau density with a Gaussian resolution function:
     $$(f_L \otimes f_G)(x) = \frac{1}{\sqrt{2\pi}\sigma_G} \int_{-\infty}^{+\infty} f_{\text{Landau}}(x'; \text{MPV}_L, \sigma_L) \exp\left( -\frac{(x - x')^2}{2\sigma_G^2} \right) dx'$$
   - This convolved model matches both Data ($\text{Peak} = 1.32\text{ QDC}, \sigma_L = 0.63, \sigma_G = 1.80$) and MC ($\text{Peak} = 1.32\text{ QDC}, \sigma_L = 0.32, \sigma_G = 0.86$) across the entire charge range.

---

## 4. Pad 3: Cluster QDC Slices along Fiber Length (Lower-Left)

### What is shown?
Data cluster QDC distributions sliced into 4-cm windows along the fiber length:
- $d \in [2, 6]\text{ cm}$ (closest to the SiPM readout, Blue)
- $d \in [12, 16]\text{ cm}$ (Teal)
- $d \in [22, 26]\text{ cm}$ (Orange)
- $d \in [32, 36]\text{ cm}$ (farthest from the SiPM readout, Red)

Each slice is individually fitted with the LanGaus convolution model.

### Key Observation: Direct Visualization of Light Attenuation
- As the muon impact point moves farther from the SiPM ($2\text{ cm} \to 34\text{ cm}$), scintillation photons travel greater distances through the polystyrene fiber core, undergoing bulk absorption and cladding reflections.
- The fitted peak shifts continuously and monotonically toward lower charge:
  $$1.97\text{ a.u.} \longrightarrow 1.63\text{ a.u.} \longrightarrow 1.45\text{ a.u.} \longrightarrow 1.20\text{ a.u.}$$
- The LanGaus fit curve precisely overlays the empirical peak of each slice without any artificial offsets.

---

## 5. Pad 4: SciFi Optical Attenuation Length Extraction (Lower-Right)

### What is shown?
The LanGaus peak position extracted for each distance slice plotted against distance to the SiPM, fitted with Beer-Lambert exponential attenuation:
$$Q_{\text{peak}}(x) = A_0 \exp\left(-\frac{x}{\lambda_{\text{att}}}\right)$$

### Results:
| Dataset | Initial Amplitude $A_0$ [a.u.] | Attenuation Length $\lambda_{\text{att}}$ [cm] | $\chi^2 / \text{ndf}$ |
| :--- | :---: | :---: | :---: |
| **Data (2023 Run 8329)** | $2.04 \pm 0.04$ | **$60.8 \pm 4.5\text{ cm}$** | $4.8 / 7$ ($p = 0.68$) |
| **SingleMuMC (FLUKA + Geant4)** | $1.64 \pm 0.07$ | **$64.3 \pm 10.2\text{ cm}$** | $4.2 / 7$ ($p = 0.75$) |

### Physical Interpretation & Agreement:
1. **Data vs MC Consistency**:
   - The fitted attenuation lengths ($\lambda_{\text{att}}^{\text{Data}} \approx 60.8\text{ cm}$ vs $\lambda_{\text{att}}^{\text{MC}} \approx 64.3\text{ cm}$) are in excellent statistical agreement within $1\sigma$.
2. **Detector Technology Comparison**:
   - Kuraray SCSF-78 / Nol-lux scintillating fibers typically exhibit effective attenuation lengths of $50\text{--}80\text{ cm}$ when read out by SiPMs over short active lengths ($\approx 40\text{ cm}$), where both short-attenuation (cladding/UV modes) and long-attenuation (core transmission) components contribute.
   - An effective single-exponential attenuation length of $\sim 61\text{ cm}$ accurately characterizes the light collection response across the SND@LHC tracker fiducial volume.

---

## 6. Summary of Methodology

1. **Track-Cluster Association**:
   Only clusters within a narrow residual distance ($\Delta r \le 1.0\text{ mm}$) to an isolated reconstructed SciFi track are used, rejecting noise hits and unrelated secondary tracks.
2. **LanGaus Fitting**:
   Pure Landau models fail to describe the rounded peak caused by SiPM and electronic smearing. LanGaus provides an unbiased and stable extraction of the true most probable energy deposit.
3. **Attenuation Extraction**:
   Slicing cluster QDC across fiber coordinates and tracking the peak yields a robust, clean measurement of $\lambda_{\text{att}}$, free from threshold distortion and zero-pedestal contamination.
