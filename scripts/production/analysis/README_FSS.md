# Finite-Size Scaling Analysis — `full_fss_analysis.py`

Self-consistent FSS pipeline for the 2D Ising model (square lattice, PBC, Metropolis).
All critical exponents are extracted as **free parameters**; Onsager exact values are used only for validation.

## Quick Start

```bash
pip install numpy scipy matplotlib
cd scripts/production/analysis
python3 full_fss_analysis.py
```

**Input:** JSON files produced by `parse_final.py`, located at
`results/production/L{L}/parsed/observables_L{L}.json`.

**Lattice sizes:** $L \in \{24, 32, 48, 64, 96, 128\}$.

---

## Pipeline Overview

### Phase 1 — $\beta_{pc}$ (Pseudo-Critical Temperature)

`find_beta_pc()` — Binder cumulant $U_4(\beta, L)$ crossings.

- Weighted cubic splines ($k=3$) of $U_4(\beta)$ for each $L$
- Brent root-finding on consecutive $(L_1, L_2)$ pairs
- Bootstrap resampling ($N=250$) for crossing errors
- Inverse-variance weighted mean → $\beta_{pc} \pm \sigma$, $U^* \pm \sigma$

### Phase 2 — $\nu$ (Correlation Length Exponent)

`extract_nu()` — Binder slope scaling $S(L) = |dU_4/d\beta|_{\beta_{pc}}$.

- Spline derivative at fixed $\beta_{pc}$ for each $L$; parametric bootstrap errors
- **Global bootstrap** (`bootstrap_global_slopes`): builds the full covariance matrix $C_{ij}$ across all $L$ simultaneously
- **Correlated $\chi^2$ fit** (`fit_nu_correlated`): $S(L) = A \cdot L^{1/\nu}$ using SVD pseudo-inverse of $C_{ij}$ → yields $\nu_{\text{final}}$, used in all subsequent phases
- Uncorrelated fits with correction ansatz $S = A \cdot L^{1/\nu} [1 + B \cdot L^{-\omega}]$ ($\omega = 2$, CFT) for $L_{\min}$ cuts at $\{24, 32, 48\}$, used as cross-check

### Phase 3 — $\gamma/\nu$, $\beta/\nu$ (Magnetic Exponents)

`extract_magnetic_exponents()` — Free power-law fits of peak observables.

- **$\chi'_{\max}(L)$**: 3-point parabolic interpolation around the raw peak (Rummukainen standard), errors from `pcov`; spline cross-check
- **$M(\beta_{pc}, L)$**: spline interpolation at $\beta_{pc}$, parametric bootstrap
- Fit models ($L \geq 32$):
  - $\chi'_{\max} = A \cdot L^{\gamma/\nu} \cdot [1 + B \cdot L^{-2}]$ (3-param, exponent free)
  - $M = A \cdot L^{-\beta/\nu} \cdot [1 + B \cdot L^{-2}]$ (3-param, exponent free)
- **Derived exponents**:
  $\gamma = (\gamma/\nu)\cdot\nu$, $\beta = (\beta/\nu)\cdot\nu$,
  $\eta = 2 - \gamma/\nu$, $\delta = 1 + (\gamma/\nu)/(\beta/\nu)$ (Griffiths–Rushbrooke)
- **Hyperscaling check**: $2(\beta/\nu) + \gamma/\nu = d = 2$

### Phase 3b — Sub-Leading Corrections

`extract_all_corrections()` — Diagnostic analysis (does not modify Phase 3 results).

Two-step procedure + F-test per observable ($\chi'_{\max}$, $M$, $S_{\text{Binder}}$, $C_{\max}$):
1. Simple power law: $O = A \cdot L^\alpha$ (2 params)
2. Fix $\alpha$ → $O = A \cdot L^\alpha [1 + B \cdot L^{-\omega}]$ (2 params)
3. F-test (2-param vs 3-param all-free) for model selection

### Phase 3c — Corrections Scan

`extract_corrections_scan()` — Repeats Phase 3b for multiple $L_{\min}$ cuts $\{24, 32\}$, producing a comparison table.

### Phase 4 — $\alpha$ (Specific Heat Exponent)

`extract_alpha()` — $C_{\max}(L)$ via parabolic fit.

- **Primary**: $C_{\max} = A \cdot L^g \cdot (1 + q \ln L)$ with $g = \alpha/\nu$ free
- **Cross-check**: $C_{\max} = a + b \ln L$ (pure logarithmic)
- F-test to decide whether the power-law correction is needed
- Derives $\alpha = g \cdot \nu$

### Phase 5 — Data Collapse

`data_collapse_plots()` — Visual validation using **measured** exponents.

Four collapse plots: $\chi'/L^{\gamma/\nu}$, $M \cdot L^{\beta/\nu}$, $U_4$, $C_v / \ln L$ vs $(\beta - \beta_{pc}) L^{1/\nu}$.

### Robustness

`plot_robustness_vs_Lmin()` — Three-panel stability plot of $\beta_{pc}$, $1/\nu$, $\gamma/\nu$ vs $L_{\min}$ cutoff.
Uses simplified spline-peak extraction; results are qualitative.

---

## Output

All output is written to `results/production/`:

| Type | Path | Content |
|------|------|---------| 
| Plots | `plots/fss/*.pdf` | 14 analysis figures (see below) |
| Results | `analysis/fss_results_complete.dat` | Full numerical results |


### Plots

| File | Content |
|------|---------| 
| `binder_crossing_with_inset.pdf` | $U_4$ crossing + zoom inset |
| `beta_pc_vs_L_convergence.pdf` | $\beta_{pc}$ convergence vs $L$ |
| `nu_extraction_loglog.pdf` | $\nu$ extraction (log–log) |
| `exponent_fits_magnetic.pdf` | $\gamma/\nu$ and $\beta/\nu$ (two-panel) |
| `specific_heat_log.pdf` | $C_{\max}$ vs $\ln L$ |
| `collapse_{susceptibility,magnetization,binder,cv_log}_inset.pdf` | Data collapse (4 plots) |
| `robustness_vs_Lmin.pdf` | Exponent stability vs $L_{\min}$ |
| `chi_parabolic_fit_example.pdf` | $\chi(\beta)$ parabolic fit, $L=96$ |
| `C_parabolic_fit_example.pdf` | $C(\beta)$ parabolic fit, $L=96$ |
| `specific_heat_scaling_fit.pdf` | $C_{\max}(L)$ generalized fit |
| `magnetization_vs_L.pdf` | $M(\beta_{pc})$ vs $L$ (log–log) |



## Configuration

Key parameters at the top of the script:

```python
L_LIST         = [24, 32, 48, 64, 96, 128]
BETA_RANGE_FIT = (0.43, 0.45)     # beta window for spline fits
N_BOOTSTRAP    = 250              # bootstrap replicas
OMEGA_EXACT    = 2.0              # CFT correction-to-scaling exponent (fixed)
```

